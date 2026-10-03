# Crater 实验

本目录保存平台说明，以及经过脱敏的探测程序、镜像定义、作业模板和脚本。禁止提交内部域名、凭据、用户名、个人路径或真实节点名。

执行范围、顺序和验收统一由[当前计划](../../docs/plans/Mycroft_26日开发路线图.md)规定。已实现 Day14 固定源码构建、Day15 同 Pod 双 GPU workload 及其上传包；具体命令见[构建说明](../../instrumentation/nccl-2.21.5/README.md)和[workload README](../../workloads/minimal_allreduce/README.md)。M1 双节点 runner 与资源预检配置、真实运行证据及离线核对说明见 [M1 实验 README](../../workloads/minimal_allreduce/M1_README.md)；它需要 Master/Worker 各一个 rank/GPU，当前技术证据通过，最终验收等待用户确认。

`package_day16_results.py` 已能把指定布局的必要文本证据打成一个包，工具说明见[采集框架 README](../../instrumentation/nccl-2.21.5/day16/README.md)。M1 runner 已复用该归档逻辑，rank0 汇总两侧状态并自动打包，成功和失败都供用户单文件下载；后续新 runner 沿用这一行为。Day14/15 的逐文件取回说明属于历史流程。
