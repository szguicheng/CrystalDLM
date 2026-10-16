# GPU smoke 的节点问题

- A100 作业532764保持PENDING，调度预估等待约两天。用户不能通过scontrol修改已提交任务的partition，因此确认PENDING后撤销，再提交V100S任务。
- V100S作业532791在gpu-v100s-06启动后，超过3分钟仍无训练步或模型配置文件产生。
- sstat记录CPU约1秒、RSS约325MB；计算节点上进程3429169的状态为`Dl`，内核等待位置为`os_acquire_rwlock_write`。
- 同节点只读诊断`nvidia-smi --query-compute-apps`也没有正常返回。证据支持GPU驱动/设备层等待，不是根据日志静默猜测模型停止。
- 因为没有训练进展或检查点，撤销该smoke任务并寻找另一可用GPU节点；不把它当作模型结果，也不修改计数/数据处理逻辑来掩盖问题。

具体作业终态、替代节点与后续成功记录见Slurm日志和STATUS.json。
