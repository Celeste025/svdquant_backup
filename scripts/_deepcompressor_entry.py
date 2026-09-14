#!/usr/bin/env python
"""Run deepcompressor diffusion CLIs without ``python -m`` / run_module(__main__).

``python -m deepcompressor...`` segfaults in this environment (libarrow/jemalloc);
importing the package and invoking the same main logic is stable.
"""
from __future__ import annotations

import os
import sys
import traceback

# Import torch before any FluxPipeline/diffusers path that pulls pyarrow/libarrow.
# Otherwise libarrow.so init can SIGSEGV when torch has not been loaded yet.
import torch  # noqa: F401
import pyarrow  # noqa: F401


def _run_collect_calib(argv: list[str]) -> None:
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    from deepcompressor.app.diffusion.dataset.collect.calib import CollectConfig, collect, get_dataset

    sys.argv = ["deepcompressor.app.diffusion.dataset.collect.calib", *argv]
    parser = DiffusionPtqRunConfig.get_parser()
    parser.add_config(CollectConfig, scope="collect", prefix="collect")
    configs, _, unused_cfgs, unused_args, unknown_args = parser.parse_known_args()
    ptq_config, collect_config = configs[""], configs["collect"]
    if unused_cfgs:
        print(f"Warning: unused configurations {unused_cfgs}")
    if unused_args is not None:
        print(f"Warning: unused arguments {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"

    collect_dirpath = os.path.join(
        collect_config.root,
        str(ptq_config.pipeline.dtype),
        ptq_config.pipeline.name,
        ptq_config.eval.protocol,
        collect_config.dataset_name,
        f"s{collect_config.num_samples}",
    )
    print(f"Saving caches to {collect_dirpath}")
    dataset = get_dataset(
        collect_config.data_path,
        max_dataset_size=collect_config.num_samples,
        return_gt=ptq_config.pipeline.task in ["canny-to-image"],
        repeat=1,
    )
    ptq_config.output.root = collect_dirpath
    os.makedirs(ptq_config.output.root, exist_ok=True)
    collect(ptq_config, dataset=dataset)


def _run_ptq(argv: list[str]) -> None:
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    from deepcompressor.app.diffusion.ptq import main
    from deepcompressor.utils import tools

    sys.argv = ["deepcompressor.app.diffusion.ptq", *argv]
    config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
    if unused_cfgs:
        tools.logging.warning(f"Unused configurations: {unused_cfgs}")
    if unused_args is not None:
        tools.logging.warning(f"Unused arguments: {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"
    try:
        main(config, logging_level=tools.logging.DEBUG)
    except Exception as e:
        tools.logging.Formatter.indent_reset()
        tools.logging.error("=== Error ===")
        tools.logging.error(traceback.format_exc())
        tools.logging.shutdown()
        traceback.print_exc()
        config.output.unlock(error=True)
        raise e


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: _deepcompressor_entry.py <deepcompressor.app.diffusion.(dataset.collect.calib|ptq)> [args...]"
        )
    mod = sys.argv[1]
    argv = sys.argv[2:]
    if mod.endswith("dataset.collect.calib") or mod.endswith("collect.calib"):
        _run_collect_calib(argv)
    elif mod.endswith(".ptq") or mod.endswith("diffusion.ptq"):
        _run_ptq(argv)
    else:
        raise SystemExit(f"unsupported module: {mod}")


if __name__ == "__main__":
    main()
