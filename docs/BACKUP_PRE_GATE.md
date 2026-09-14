# 本地备份：gate 精确校准前

- **日期**: 2026-08-08
- **仓库**: `/home/wenjinqi.wjq/workspace/svdquant_backup`
- **Commit**: `cec628fd0bea220dc4442883bdb3c9c07c95ed7a`
- **信息**: `gate精确校准前备份`
- **本地 Tag**: `backup-pre-gate`（指向上述 commit）

## 说明

在开始「OutputsError 校准乘上 `gate_msa` / `c_gate_msa`」改造前，已在本机打好此备份。
当时尝试 `git push` 到 GitHub（`Celeste025/svdquant_backup`）因网络（Empty reply / 443 timeout）失败，**远程未必有这份 commit**；回退请以本机 tag/commit 为准。

## 如何回退

```bash
cd /home/wenjinqi.wjq/workspace/svdquant_backup
git checkout backup-pre-gate
# 或丢弃之后改动硬回退：
# git reset --hard backup-pre-gate
```

## 相关对照产物（未改，可继续对比）

- 校准 cache（可复用）: `/ssd/2/wenjinqi.wjq/datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16`
- 旧 W4A4 ckpt（无 gate 校准）: `/ssd/2/wenjinqi.wjq/ckpts/wan2.1-1.3b-int4-s16`
