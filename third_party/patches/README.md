# External source checkouts and local patches

These repositories are local independent Git checkouts, not submodules of this
repository. The checkouts themselves, their Git metadata and compiled binaries
are ignored by the outer repository. Their upstream revisions and local source
changes are recorded below so a fresh checkout can reconstruct the experiments.
Applying a patch and building CUDA extensions requires the corresponding toolchain
and must be performed explicitly; no setup script runs these commands automatically.

| Checkout | Upstream | Base commit | Local source changes |
|---|---|---|---|
| `third_party/anemoi-code` | https://github.com/anemoi-project/anemoi.git | `97a1c8b14f26f387214c8cfb8fc7e890390b93b4` | `anemoi-local.patch` |
| `third_party/MixAttention` | https://github.com/Jasonrjw/MixAttention.git | `ba74f1aecba83c2b4fa0c6f8c5aea76606f44961` | `mixattention-local.patch` |
| `third_party/SageAttention3_prescale` | https://github.com/Jasonrjw/SageAttention3_prescale.git | `673a3db2370aa1185ed1c789e7fc86f087bac234` | `sageattention3-prescale-local.patch` |
| `third_party/FastVideo_v2` | https://github.com/hao-ai-lab/FastVideo.git | `7203b4976eefcd70a52728ca1281797ce96734eb` | No local tracked changes observed at snapshot |

For example, from the outer repository root, reconstruct Anemoi into an absent
`third_party/anemoi-code` directory:

```bash
git clone https://github.com/anemoi-project/anemoi.git third_party/anemoi-code
git -C third_party/anemoi-code checkout 97a1c8b14f26f387214c8cfb8fc7e890390b93b4
git -C third_party/anemoi-code apply --check ../patches/anemoi-local.patch
git -C third_party/anemoi-code apply ../patches/anemoi-local.patch
```

Use the corresponding URL, commit and patch from the table for the other
checkouts. The patches are snapshots of source worktrees, including new source,
documentation and tests; CUDA build outputs and Python bytecode are excluded.
They were verified with `git apply --reverse --check` against the source
worktrees on 2026-11-02. A missing remote commit, changed dependency or toolchain
may still require additional setup. The separate `third_party/sage3_h3_maskfix`
source is already tracked directly by the outer repository; its machine-specific
`csrc/cutlass` symlink and build temporaries are ignored.
