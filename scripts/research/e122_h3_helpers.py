"""Self-contained H3 routing helpers for E122 benchmarking and validation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research")]

from h3_mixed_sage3_api import split_mixed_attention  # noqa: E402
import sageattn3  # noqa: E402
import sageattn3.api as sage_api  # noqa: E402
from sageattn3 import sageattn3_blackwell  # noqa: E402
from h3_native_fp4_attention import H3AttentionRouter  # noqa: E402
from h3_native_nvfp4 import TARGET_NAMES  # noqa: E402
from minimax_h3_svdquant_common import load_h3_pipeline, tree_device  # noqa: E402
from diffsynth.core.vram.layers import AutoWrappedLinear  # noqa: E402

CORRECTED = (ROOT / "third_party/sage3_h3_maskfix/sageattn3").resolve()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def check_backend() -> None:
    actual = Path(sageattn3.__file__).resolve()
    require(CORRECTED in actual.parents,
            f"corrected Sage3 was not loaded: {actual}")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")
    temp.replace(path)


def split_output(output):
    require(isinstance(output, tuple) and len(output) == 2 and
            all(torch.is_tensor(x) for x in output),
            f"unexpected DiT output: {type(output)!r}")
    return output


def attention_contract(kwargs) -> dict:
    return {
        "expected_cu": kwargs["packed_seq_params"]["cu_seqlens_q"].detach().cpu().tolist(),
        "expected_refiner_cu": kwargs["refiner_packed_seq_params"]["cu_seqlens_q"].detach().cpu().tolist(),
    }


def checked_linears(dit) -> dict:
    offloaded = 0
    for name in TARGET_NAMES:
        module = dit.get_submodule(name)
        require(isinstance(module, AutoWrappedLinear),
                f"target is not AutoWrappedLinear: {name}")
        require(module.computation_dtype == torch.bfloat16 and
                not module.enable_fp8 and len(module.lora_A_weights) == 0,
                f"target is not pure BF16: {name}")
        if module.disk_offload:
            offloaded += 1
    require(len(TARGET_NAMES) == 200, "target linear count changed")
    return {"count": len(TARGET_NAMES), "bf16_compute": True,
            "disk_offloaded_count": offloaded}


class Router(H3AttentionRouter):
    def __init__(self, dit, arm: str):
        require(arm in ("bf16", "sage3_fixed", "split_mixed"),
                f"invalid attention arm: {arm}")
        self.arm = arm
        super().__init__(dit, "bf16" if arm == "bf16" else "block_mean",
                         _official=None if arm == "bf16" else sage_api)
        self.manifest.update(e122_arm=arm,
                             corrected_sage3=str(Path(sageattn3.__file__).resolve()),
                             split_threshold=0.10 if arm == "split_mixed" else None)

    def _dispatch(self, q, k, v, cu_seqlens, softmax_scale):
        if self.arm == "bf16":
            return super()._dispatch(q, k, v, cu_seqlens, softmax_scale)
        log = self._active_log.get()
        block = self._active_main.get()
        require(log is not None and softmax_scale == 128 ** -0.5,
                "attention called outside route or scale changed")
        expected = log.expected_refiner_cu if block is None else log.expected_cu
        require(q.shape == k.shape == v.shape == (expected[-1], 56, 128),
                "QKV shape changed")
        require(all(x.dtype == torch.bfloat16 and x.is_cuda for x in (q, k, v)),
                "QKV dtype/device changed")
        log.remember_cu(cu_seqlens, expected, "refiner" if block is None else "main")
        if block is None:
            log.refiner_helpers += 1
            log.original_bf16_segments += 1
            return self.original_helper(q, k, v, cu_seqlens, softmax_scale)
        n, total = expected[1:]
        Q, K, V = (x[:n].transpose(0, 1).unsqueeze(0).contiguous()
                   for x in (q, k, v))
        if self.arm == "sage3_fixed":
            result = sageattn3_blackwell(Q, K.clone(), V, is_causal=False,
                                         per_block_mean=True)
        else:
            result = split_mixed_attention(Q, K, V, threshold=0.10)
        require(result.shape == Q.shape and result.dtype == torch.bfloat16,
                "attention output shape/dtype changed")
        out = torch.empty_like(q)
        out[:n] = result[0].transpose(0, 1)
        out[n:] = self.original_helper(q[n:], k[n:], v[n:],
                                       log.padding_cpu_cu, softmax_scale)
        log.main_rows.append({"block": block, "valid_length": n,
                              "padding_length": total - n, "arm": self.arm})
        log.fp4_calls += 1
        log.original_bf16_segments += 1
        return out
