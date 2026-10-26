# LiuGangGPU 项目存储迁移

用户于 2026-10-21 确认目标为 `/data1`。该路径挂载独立数据盘 `/dev/vdc`；`/data` 仍属于系统盘，未用作目标。

项目整体迁至 `/data1/users/liugang/crystalDLM`，原工作区保留为符号链接：

```text
/home/liugang/src/crystalDLM -> /data1/users/liugang/crystalDLM
```

迁移包含代码、数据、模型、全部历史检查点、独立环境、缓存和临时目录，约 64 GiB。整体迁移保留代码通过 `__file__.resolve()` 定位其他项目文件时的相对关系。环境仍用原前缀执行，runtime.sh 中原路径不变，写入实际落在数据盘。

使用 `rsync -aH` 保留权限、时间、符号链接和硬链接；复制后 `rsync -aHnc --delete --itemize-changes` 内容校验无差异。随后将原目录改为符号链接，检查原路径读取正常后删除系统盘旧副本。

已检查：mamba 环境、PyTorch/Transformers 导入、Qwen tokenizer、7272/900 数据及哈希、新检查点反序列化、32 条修正结果汇总，以及 20 项修正训练测试。

旧检查点身份中记录了绝对路径，模型加载器现仅在 `samefile` 确认新旧路径指向同一文件时沿用历史路径字段，仍严格比较全部非路径字段与内容哈希。迁移不修改已有权重、数据和历史身份记录。加载器目录 16 项测试通过；用一张 GPU 实际恢复旧 FLARE 和新修正模型，参数哈希与迁移前一致，检查后释放 GPU，未启动训练。见 [完整加载检查](model-load-check.json) 和 [迁移回执](receipt.json)。

系统同步已完成。迁移后系统盘约剩 143 GiB，数据盘约剩 1.5 TiB；该服务器同时有其他活动，系统盘总空闲增量不能全部归因于本项目的约 64 GiB 迁移。

以后继续使用原工作区路径。大文件操作前执行：

```bash
findmnt -T /home/liugang/src/crystalDLM
df -h /home/liugang/src/crystalDLM
```

目标应位于 `/data1`。若数据盘未挂载，先恢复挂载，避免在系统盘创建同名替代目录。
