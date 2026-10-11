#!/usr/bin/env python3
"""Export compact H3 x and x/s panels as NumPy arrays for 3D surface plots."""
from pathlib import Path

import numpy as np
import torch

OUT = Path("/data1/models/svdquant-wjq/results/E087_h3_patterns")

for source in OUT.glob("h3_linear_pattern_data_*.pt"):
    record = torch.load(source, map_location="cpu", weights_only=False)
    prompt, step, block, suffix = record["spec"]
    np.savez_compressed(
        OUT / source.name.replace("_data", "_surface").replace(".pt", ".npz"),
        x=record["panels"]["input"].numpy(),
        xs=record["panels"]["smoothed_input"].numpy(),
        prompt=prompt, step=step, block=block, suffix=suffix,
    )
