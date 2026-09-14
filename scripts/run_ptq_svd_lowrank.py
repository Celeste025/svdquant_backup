#!/usr/bin/env python
"""PTQ entry with LowRankBranch using torch.svd_lowrank (process-local monkeypatch only)."""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback

import torch  # noqa: F401  # before deepcompressor
import pyarrow  # noqa: F401


def _patch_svd_lowrank(q_extra: int = 8, niter: int = 2) -> None:
    from deepcompressor.nn.patch import lowrank as lowrank_mod

    def reset_parameters(self, weight: torch.Tensor | None = None) -> None:
        if weight is None:
            if self.rank < 0:
                torch.nn.init.zeros_(self.a.weight)
            elif self.rank > 0:
                torch.nn.init.kaiming_uniform_(self.a.weight)
                torch.nn.init.zeros_(self.b.weight)
            return
        if weight.ndim >= 2:
            assert weight.shape[2:].numel() == 1, "LinearLoRAHook only supports 2D input tensor"
        weight = weight.view(weight.shape[0], -1)
        device, dtype = weight.device, weight.dtype
        self.to(device=device, dtype=dtype)
        out_features, in_features = weight.shape
        assert self.in_features == in_features, "Input features size mismatch"
        assert self.out_features == out_features, "Output features size mismatch"
        if self.rank < 0:
            self.a.weight.data.copy_(weight)
        elif self.rank > 0:
            q = min(min(out_features, in_features), max(self.rank + q_extra, self.rank))
            U, S, V = torch.svd_lowrank(weight.float(), q=q, niter=niter)
            us = U[:, : self.rank] * S[: self.rank]
            vh = V[:, : self.rank].transpose(0, 1)
            assert not us.isnan().any(), "NaN in U * S"
            assert not vh.isnan().any(), "NaN in V^T"
            assert not us.isinf().any(), "Inf in U * S"
            assert not vh.isinf().any(), "Inf in V^T"
            self.a.weight.data.copy_(vh.to(dtype))
            self.b.weight.data.copy_(us.to(dtype))

    lowrank_mod.LowRankBranch.reset_parameters = reset_parameters  # type: ignore[method-assign]
    print(
        f"[svd_lowrank_patch] LowRankBranch.reset_parameters -> torch.svd_lowrank "
        f"(q=rank+{q_extra}, niter={niter}, dtype=float32)",
        flush=True,
    )


def main() -> None:
    _patch_svd_lowrank()
    entry_path = os.path.join(os.path.dirname(__file__), "_deepcompressor_entry.py")
    spec = importlib.util.spec_from_file_location("deepcompressor_entry", entry_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    argv = sys.argv[1:]
    if not argv:
        raise SystemExit("usage: run_ptq_svd_lowrank.py [deepcompressor.app.diffusion.ptq] <ptq-args...>")
    if not (argv[0].endswith(".ptq") or argv[0].endswith("diffusion.ptq")):
        argv = ["deepcompressor.app.diffusion.ptq", *argv]
    sys.argv = ["_deepcompressor_entry.py", *argv]
    mod.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
