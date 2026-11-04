"""Extension-free DraftMap scoring and fixed-budget ragged routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RaggedRoutePlan:
    """Compact NVFP4-then-MXFP8-then-FP16 native execution lists."""

    block_ids: Any
    nvfp4_counts: Any
    fp8_counts: Any
    fp16_counts: Any
    fp16_blocks_per_head: int
    fp8_blocks_per_head: int
    nvfp4_blocks_per_head: int
    packed_offsets: Any = None
    logical_columns: int | None = None


@dataclass(frozen=True)
class CachedExactRoute:
    """One layer's route derived from dense INT8 QK tile maxima."""

    signature: tuple[Any, ...]
    plan: RaggedRoutePlan


class AttentionRouteCache:
    """Request-local per-layer routes captured by ``first_step=True``."""

    def __init__(self, capture_mode: str = "qk_max") -> None:
        if capture_mode not in ("qk_max", "softmax_mean", "softmax_mean_fused"):
            raise ValueError("capture_mode must be qk_max or softmax_mean")
        self._capture_mode = capture_mode
        self._routes: dict[int, CachedExactRoute] = {}

    @property
    def capture_mode(self) -> str:
        return self._capture_mode

    def clear(self) -> None:
        self._routes.clear()

    def store(
        self,
        layer: int,
        signature: tuple[Any, ...],
        plan: RaggedRoutePlan,
    ) -> None:
        if type(layer) is not int or layer < 0:
            raise ValueError("layer must be a nonnegative integer")
        self._routes[layer] = CachedExactRoute(signature, plan)

    def load(self, layer: int, signature: tuple[Any, ...]) -> RaggedRoutePlan:
        cached = self._routes.get(layer)
        if cached is None:
            raise RuntimeError(
                f"no exact DraftMap route is cached for layer {layer}; "
                "run the layer once with first_step=True"
            )
        if cached.signature != signature:
            raise RuntimeError(
                f"cached exact DraftMap geometry/policy mismatch for layer {layer}"
            )
        return cached.plan

    def __len__(self) -> int:
        return len(self._routes)

    @property
    def allocated_bytes(self) -> int:
        total = 0
        seen: set[int] = set()
        for cached in self._routes.values():
            plan = cached.plan
            for tensor in (
                plan.block_ids,
                plan.packed_offsets,
                plan.nvfp4_counts,
                plan.fp8_counts,
                plan.fp16_counts,
            ):
                identity = id(tensor)
                if (
                    identity not in seen
                    and tensor is not None
                    and hasattr(tensor, "numel")
                    and hasattr(tensor, "element_size")
                ):
                    seen.add(identity)
                    total += tensor.numel() * tensor.element_size()
        return total


def _route_counts(
    items: int,
    *,
    sparsity_ratio: float,
    nvfp4_ratio: float,
    fp8_ratio: float,
    fp16_ratio: float,
) -> tuple[int, int, int]:
    """Hamilton-allocate the retained budget as ``(FP16, MXFP8, NVFP4)``."""

    if type(items) is not int or items <= 0:
        raise ValueError("items must be a positive integer")
    values = (sparsity_ratio, nvfp4_ratio, fp8_ratio, fp16_ratio)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise TypeError("route ratios must be finite real scalars")
    sparsity, nvfp4, fp8, fp16 = map(float, values)
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1)")
    if min(nvfp4, fp8, fp16) < 0.0:
        raise ValueError("precision ratios must be nonnegative")
    if not math.isclose(
        nvfp4 + fp8 + fp16, 1.0, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError("NVFP4/MXFP8/FP16 ratios must sum to one")

    retained = min(items, max(1, math.floor((1.0 - sparsity) * items + 0.5)))
    quotas = (retained * fp16, retained * fp8, retained * nvfp4)
    counts = [math.floor(value) for value in quotas]
    order = sorted(
        range(3), key=lambda index: (-(quotas[index] - counts[index]), index)
    )
    for index in order[: retained - sum(counts)]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def draft_probability(
    q_pool: Any,
    k_pool: Any,
    *,
    q_second: Any | None = None,
    k_second: Any | None = None,
) -> Any:
    """Return pooled-QK probability with an optional diagonal-Jensen correction."""

    import torch

    if q_pool.ndim != 4 or k_pool.ndim != 4:
        raise ValueError("q_pool/k_pool must be [B,H,blocks,D]")
    if q_pool.shape[0] != k_pool.shape[0] or q_pool.shape[2:] != k_pool.shape[2:]:
        raise ValueError("q_pool/k_pool geometries are incompatible")
    if q_pool.shape[1] != k_pool.shape[1]:
        if q_pool.shape[1] % k_pool.shape[1]:
            raise ValueError("query heads must be divisible by key heads")
        groups = q_pool.shape[1] // k_pool.shape[1]
        k_pool = k_pool.repeat_interleave(groups, dim=1)
        if k_second is not None:
            k_second = k_second.repeat_interleave(groups, dim=1)
    head_dim = q_pool.shape[-1]
    logits = torch.matmul(q_pool.float(), k_pool.float().transpose(-2, -1)).mul_(
        1.0 / math.sqrt(head_dim)
    )
    if (q_second is None) != (k_second is None):
        raise ValueError("diag-Jensen requires both Q and K second moments")
    if q_second is not None:
        if q_second.shape != q_pool.shape or k_second.shape != k_pool.shape:
            raise ValueError("second moments must match expanded pool shapes")
        pair_variance = torch.matmul(q_second.float(), k_second.float().transpose(-2, -1))
        pair_variance.sub_(
            torch.matmul(
                q_pool.float().square(),
                k_pool.float().square().transpose(-2, -1),
            )
        ).div_(head_dim).clamp_min_(0.0)
        logits.add_(pair_variance, alpha=0.5)
    return torch.softmax(logits, dim=-1).to(torch.float16)


def route_probability(
    probability: Any,
    anchors: Any | None,
    *,
    anchor_count: int,
    prefix_blocks: int,
    sparsity_ratio: float,
    nvfp4_ratio: float = 0.0,
    fp8_ratio: float,
    fp16_ratio: float,
) -> RaggedRoutePlan:
    """Stable global top-k with missing anchors charged to the lowest precision."""

    import torch

    if probability.ndim != 4 or probability.shape[-1] != probability.shape[-2]:
        raise ValueError("probability must be square [B,H,R,R]")
    batch, heads, rows, columns = probability.shape
    if anchors is None:
        if anchor_count != 0:
            raise ValueError("anchor_count must be zero when anchors are disabled")
    elif (
        anchors.shape != (rows, columns)
        or anchors.dtype != torch.bool
        or anchors.device != probability.device
    ):
        raise ValueError("anchors must be a bool matrix matching DraftMap")
    if type(anchor_count) is not int or not 0 <= anchor_count <= rows * columns:
        raise ValueError("anchor_count must be a valid static adjacency count")
    if type(prefix_blocks) is not int or prefix_blocks < 0:
        raise ValueError("prefix_blocks must be nonnegative")
    nominal_fp16, nominal_fp8, nominal_nvfp4 = _route_counts(
        rows * columns,
        sparsity_ratio=sparsity_ratio,
        nvfp4_ratio=nvfp4_ratio,
        fp8_ratio=fp8_ratio,
        fp16_ratio=fp16_ratio,
    )
    retained = nominal_fp16 + nominal_fp8 + nominal_nvfp4
    fp16_per_head = nominal_fp16
    fp8_per_head = nominal_fp8
    nvfp4_per_head = nominal_nvfp4
    if nominal_nvfp4:
        lowest_code, lowest_count = 1, nominal_nvfp4
    elif nominal_fp8:
        lowest_code, lowest_count = 2, nominal_fp8
    else:
        lowest_code, lowest_count = 3, nominal_fp16
    higher_count = retained - lowest_count
    if anchors is not None and anchor_count > lowest_count:
        raise ValueError("anchor_count exceeds the configured lowest-precision budget")

    selected = torch.argsort(
        probability.reshape(batch, heads, -1),
        dim=-1,
        descending=True,
        stable=True,
    )[..., :retained]
    precision = torch.zeros(
        (batch, heads, rows * columns),
        device=probability.device,
        dtype=torch.uint8,
    )
    precision.scatter_(
        -1,
        selected[..., :fp16_per_head],
        torch.full_like(selected[..., :fp16_per_head], 3, dtype=torch.uint8),
    )
    precision.scatter_(
        -1,
        selected[..., fp16_per_head : fp16_per_head + fp8_per_head],
        torch.full_like(
            selected[..., fp16_per_head : fp16_per_head + fp8_per_head],
            2,
            dtype=torch.uint8,
        ),
    )
    precision.scatter_(
        -1,
        selected[..., fp16_per_head + fp8_per_head :],
        torch.ones_like(
            selected[..., fp16_per_head + fp8_per_head :], dtype=torch.uint8
        ),
    )
    if anchors is not None:
        anchor_mask = anchors.reshape(1, 1, -1).expand(batch, heads, -1)
        missing = anchor_mask & precision.eq(0)
        missing_count = missing.sum(dim=-1)
        low_ids = selected[..., higher_count:]
        low_is_anchor = anchor_mask.gather(-1, low_ids)
        ordinary_reversed = (~low_is_anchor).flip(-1)
        evict = (
            ordinary_reversed
            & (ordinary_reversed.cumsum(dim=-1) <= missing_count.unsqueeze(-1))
        ).flip(-1)
        low_precision = precision.gather(-1, low_ids).masked_fill_(evict, 0)
        precision.scatter_(-1, low_ids, low_precision)
        precision.masked_fill_(missing, lowest_code)
    precision = precision.view(batch, heads, rows, columns)
    nvfp4_counts = (precision == 1).sum(dim=-1, dtype=torch.int32)
    fp8_counts = (precision == 2).sum(dim=-1, dtype=torch.int32)
    fp16_counts = (precision == 3).sum(dim=-1, dtype=torch.int32)

    column_ids = torch.arange(columns, device=probability.device, dtype=torch.int32).view(
        1, 1, 1, columns
    )
    # Per-row argsort compacts NVFP4, MXFP8, FP16, then unused ids.
    sort_key = torch.where(
        precision == 1,
        column_ids,
        torch.where(
            precision == 2,
            columns + column_ids,
            torch.where(
                precision == 3,
                2 * columns + column_ids,
                3 * columns + column_ids,
            ),
        ),
    )
    block_ids = sort_key.argsort(dim=-1).to(torch.int32).add_(prefix_blocks)
    return RaggedRoutePlan(
        block_ids=block_ids.contiguous(),
        nvfp4_counts=nvfp4_counts.contiguous(),
        fp8_counts=fp8_counts.contiguous(),
        fp16_counts=fp16_counts.contiguous(),
        fp16_blocks_per_head=fp16_per_head,
        fp8_blocks_per_head=fp8_per_head,
        nvfp4_blocks_per_head=nvfp4_per_head,
    )


__all__ = ["RaggedRoutePlan", "draft_probability", "route_probability"]
