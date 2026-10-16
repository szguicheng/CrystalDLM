# 持续计算会话

按用户最新要求，后续连续验证复用一个 Slurm 资源分配和计算节点内的 tmux 会话；不再为每次短验证重新提交作业。登录节点只做传输、排队和读取日志。

- 资源：stingy，1节点中的1张A100、4CPU、24G内存，最长2小时。
- tmux socket：`crystaldlm-<Slurm job id>`；session：`work`。
- 执行器只顺序执行该分配私有 `pending/` 目录里的实验脚本。
- 每个实验有独立脚本副本、SHA、开始/结束时间、退出码及stdout/stderr。
- 连续30分钟没有实验时自动释放；写入分配目录的 `STOP` 文件可在当前实验结束后释放。

## 连接与提交

本地SSH和rsync使用相同的OpenSSH ControlMaster socket，连接最多复用2小时：

```bash
rtk proxy experiments/wyckoff_tokenizer_search/hpc/ssh_burgundy.sh 'hostname'
rtk proxy experiments/wyckoff_tokenizer_search/hpc/rsync_shared.sh -az local-file Burgundy:/scratch/chenggui2/CrystalDLM/tokenizer_search_20260908/
```

远端 `hpc/active_allocation.json` 指向当前分配，详细运行状态在 `compute_allocations/<jobid>/state.json`。使用前同时核查 Slurm job 是否仍在RUNNING；文件本身不能证明分配还活着。

把已经写完的可信实验 `.sh` 文件通过rsync原子传入 `compute_allocations/<jobid>/pending/` 即可入队。脚本在项目远端根目录运行，继承分配的CUDA/CPU环境，并额外提供 `CRYSTALDLM_RUN_ID`。不要把下载的文献或未经审核的外部脚本放入执行目录。

运行中位于 `running/<runid>/`，结束后移动到 `finished/<runid>/`。先读 `receipt.json`，再读输出；若Slurm分配已终止但收据仍显示running，应作为中断调查，不能根据陈旧状态直接重新跑。

需要人工查看tmux时，可以在登录节点使用现有分配的一个重叠控制step进入计算节点，例如：

```bash
srun --jobid=<jobid> --overlap -N1 -n1 --cpus-per-task=1 --pty tmux -L crystaldlm-<jobid> attach -t work
```

这复用既有分配，不是再次申请一张GPU。日常程序化操作使用共享目录队列，无需反复进入计算节点。
