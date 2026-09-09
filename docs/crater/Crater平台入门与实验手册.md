# Crater 平台入门与实验手册

本手册只保留完成 Mycroft 复现所需的 Crater 知识和实验步骤。Day 02 学会通过 GUI 运行一个 CPU 容器；Day 03 再运行双 Pod DDP；Day 04 验证单 GPU 环境。

## 1. Crater 在项目中的作用

Crater 是集群作业入口。用户在网页中选择镜像、资源、启动命令和挂载目录；平台随后在 Kubernetes 集群中创建并调度 Pod。当前阶段不要求使用 `kubectl` 或理解 Kubernetes 管理细节。

```text
本地代码或共享目录
        ↓
Crater GUI 作业配置
        ↓
Job → Role 配置（职责、镜像、资源、命令）
        ↓
作业控制器按 Replica 数量创建 Pod
        ↓
Pod → Container → 启动命令创建进程
        ↓
物理 Node 上的 CPU/GPU
```

必要概念：

| 概念 | 本项目中的含义 |
|---|---|
| Job | 用户提交的一次完整作业 |
| Role | 一类逻辑职责及其 Pod 运行配置，例如 Master、Worker；Role 本身不创建 Pod |
| Replica | 作业控制器需要依据某个 Role 配置创建的 Pod 数量 |
| Pod | Kubernetes 调度的基本运行单元 |
| Container | Pod 中由镜像启动的进程环境 |
| Image | 创建容器所需的只读软件环境模板 |
| Node | Pod 最终运行的物理机或集群节点 |
| Mount | 容器访问持久化或共享文件的路径 |

这里的“一份实例”具体指实际创建出来的一个 Pod，而不是 Role 自己创建出的对象。Crater 提交作业配置后，由 Kubernetes/Volcano 作业控制器读取 Role 配置，并按照 Replica 数量创建 Pod。每个 Pod 随后执行启动命令，启动一个或多个操作系统进程，因此 Replica 数与进程数不一定相等。

两个 Pod 可能被调度到同一台 Node，因此“两个容器”不能直接证明“使用了两台物理机”。Target Node Allow List 只限制允许选择哪些节点，也不自动保证副本分散。

## 2. GUI 页面字段

| 页面字段 | 含义 | Day 02 设置 |
|---|---|---|
| 作业名称 | 作业列表中的名称 | `day02-cpu-smoke` |
| 副本数 | 作业控制器按照当前 Role 配置创建多少个 Pod | 1 |
| CPU | 每个副本申请的 CPU 核数 | 1 |
| Memory | 每个副本申请的内存 | 2 GiB |
| GPU | 每个副本申请的 GPU 数 | 0 |
| Container Image | 容器的软件环境 | 平台已有的 Ubuntu 或 Python 基础镜像 |
| 启动命令 | 容器启动后执行的命令 | 见下一节 |
| 工作目录 | 命令执行位置 | GUI 显示的持久化用户目录 |
| Data Mounting | 挂入容器的持久化空间 | 保持默认 User Space |
| Target Node Control | 限定调度节点 | 关闭 |

Day 02 使用 **Custom Job** 页面，不使用 PyTorch DDP Job。所有配置、提交、日志查看和清理都通过 GUI 完成。启动命令虽然是 Shell 文本，但它填写在 GUI 的“启动命令”输入框中，不是在本地终端提交作业。

## 3. Day 02：单容器 CPU 冒烟作业

### 3.1 任务边界

- 实际提交一个 1 CPU、2 GiB、0 GPU 的 Custom Job。
- 查看作业状态、Pod、Node 和日志。
- 验证持久化目录中的结果文件。
- 找到停止、删除或清理入口。
- 不使用 DDP、GPU、NCCL、RDMA、自定义镜像或目标节点限制。

### 3.2 GUI 操作

1. 进入 `Create New Job`，选择 `Custom Job`。
2. 按第 2 节填写资源字段。
3. 保留 GUI 显示的默认持久化挂载点，记为 `<MOUNT_POINT>`。
4. 将下列命令中的 `<MOUNT_POINT>` 替换为 GUI 显示的真实路径，再粘贴到“启动命令”：

```bash
/crater-start.sh bash -lc 'echo "hello from crater"; hostname; whoami; pwd; echo "day02 success" > <MOUNT_POINT>/day02-crater-smoke.txt'
```

5. 提交前检查 GPU 为 0、Target Node Control 关闭。
6. 提交后进入作业详情，依次观察状态、Pod、Node 和日志。
7. 作业结束后检查 `<MOUNT_POINT>/day02-crater-smoke.txt`。
8. 找到停止、删除或清理入口；完成状态的作业按平台规则清理。

### 3.3 预期现象

- 作业从排队或 Pending 进入 Running，最后正常完成。
- 日志出现 `hello from crater`、hostname、普通用户名和工作目录。
- 结果文件内容为 `day02 success`，容器退出后仍然存在。
- 用户能指出 Job、Pod、Node、日志和清理入口。

### 3.4 排错顺序

| 现象 | 首先检查 |
|---|---|
| 长时间 Pending | CPU/内存资源、配额、节点限制 |
| 镜像无法启动 | 镜像选择和访问权限 |
| 容器立即退出 | 启动命令、引号、工作目录和权限 |
| 日志正常但没有文件 | 挂载点、输出路径和写权限 |

每次只修改一个字段。CPU 冒烟失败时，不要通过申请 GPU、启用 RDMA 或限制节点来绕过问题。

## 4. Day 03：双 Pod CPU DDP

### 4.1 程序任务

在 `cluster/crater/probes/cpu_ddp_probe.py` 中完成 `run_all_reduce()`：使用平台注入的 `WORLD_SIZE`、`RANK`、`MASTER_ADDR` 和 `MASTER_PORT` 初始化 Gloo 进程组，以 `rank + 1` 作为本 rank 初始值，执行一次求和 AllReduce，最后销毁进程组。

本地测试只验证环境配置接口，不模拟多 Pod 通信。真正的分布式验收必须在 Crater 完成。

### 4.2 GUI 配置

使用 PyTorch DDP Job，并为 Master 和 Worker 选择同一个包含 PyTorch 的稳定镜像：

| Role | Replica | CPU | Memory | GPU |
|---|---:|---:|---:|---:|
| Master | 1 | 1 | 2 GiB | 0 |
| Worker | 1 | 1 | 2 GiB | 0 |

- 两个 Role 使用相同的持久化挂载点和代码路径。
- 启动命令为 `/crater-start.sh python <REPOSITORY_ROOT>/cluster/crater/probes/cpu_ddp_probe.py`，执行前替换真实路径。
- Target Node Control 关闭；Day 03 不要求两个 Pod 位于不同物理 Node。

### 4.3 预期现象

- 作业详情中出现一个 Master Pod 和一个 Worker Pod。
- 两份日志的 `WORLD_SIZE` 都是 2，`RANK` 分别为 0 和 1。
- rank 0 初始值为 1，rank 1 初始值为 2；两份日志最终都显示 `result` 为 3。
- 两个进程正常退出，整个 Job 成功完成。

## 5. Day 04 预告

Day 04 使用单 Pod、单 GPU，验证 `nvidia-smi`、PyTorch CUDA、GPU tensor 运算和版本信息。

## 6. 公开仓库规则

真实用户名、个人挂载路径、内部域名、镜像仓库地址、节点名、账户 ID、凭据和原始日志不得直接提交。GUI 导出的原始配置先保存在仓库外；确认真实格式并完成脱敏后，才在 `cluster/crater/jobs/` 中新增可复查配置。

## 7. Day 02 验收问题

完成实验后，应能用自己的话回答：

1. Image 与 Container 有什么区别？
2. Pod 与物理 Node 有什么区别？
3. Role、Replica 和进程是什么关系？
4. 为什么两个 Pod 不一定运行在两台物理机？
5. 作业失败后，依次从哪里查看状态和日志？

## 8. Day 02 实验结果

- 使用 Crater GUI 成功运行 1 CPU、2 GiB、0 GPU 的单容器 Custom Job。
- 容器成功执行 `/crater-start.sh` 和 Bash 启动命令。
- 日志、身份、工作目录及持久化结果文件均符合第 3.3 节预期。
- 已完成 Job、Pod、Container、Image、Node、Role 和 Replica 的基础概念验收。
