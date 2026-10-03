# M1：双节点 NET/RDMA 路径验证

范围、顺序和完成定义见[当前计划 M1](../../docs/plans/Mycroft_26日开发路线图.md)。本页说明运行入口与证据核对方式。Crater 首轮缺少 verbs 动态库；补齐依赖后的重试已完成双节点 NET/IB AllReduce，原作业因应用日志与 Proxy TRACE 交错而核对失败，修复后的离线复核通过。证据齐全后用户明确进入 M2；M1 已验证并提交到 main（`f333d6d`）。

## 这次实验提供什么证据

Day15 的同机 P2P/CUMEM 基线证明项目 NCCL 能正确完成 AllReduce，但没有验证论文依赖的 NET Proxy/RDMA 路径。M1 用两个物理节点、每节点一个 rank/GPU，确认项目库真的走 NET/IB，供 M2 采集 readiness、transmitted 和 done。

两个 Pod 各只暴露一张 GPU，故 rank0 和 rank1 都使用各自容器的 `cuda device=0`。原程序新增显式 `--single-gpu-node` 模式；默认 Day15 模式仍要求同 Pod 至少两张可见 GPU、`device=rank`，原离线核对约束保持不变。

共享目录只传递原始 `ncclUniqueId` 字节、就绪标记和日志；AllReduce 数组不经过这个目录。`ncclCommInitRank` 后 NCCL 自行建立控制和数据连接。TCP bootstrap 出现不等于数据经 Socket；数据路径要看 `Channel ... [send/receive] via NET/IB/...`，并结合真实结果验证。

## 文件与行为

- `cluster/crater/probes/rdma_probe.py`：检查 GPU inventory、verbs 动态库、实际打开设备 context、IB/RoCE active port、memlock、网络接口及 host 身份摘要。不注册 MR、不建立 QP、不传数据；`RESOURCES_VISIBLE` 仅是本地预检。
- `cluster/crater/scripts/run_m1.py`：每侧先探测资源、核对独立构建 hash、编译 workload，再等两侧就绪。rank0 创建新目录和唯一运行 ID；rank1 等待配置。各 rank 的进程有超时，失败状态写入结果。
- `cluster/crater/scripts/package_m1.py`：生成一个上传包，包含上述工具和 workload，不包含 CUDA、NCCL 库或源码；本次复用已验证的 TRACE 库。
- `verify_m1.py`：核对两侧来源、数值、应用阶段、真实网络连接、身份和 rank0 的实际计划。它复用 Day15 的选择日志解析器，不放宽旧单节点验收。
- 复用 `package_day16_results.py` 的归档逻辑，以 `m1-results/` 为根，只打包必要文本证据。失败或缺失项写入 manifest，不打包 NCCL ID、库或中间文件。

当前 runner 固定 `NCCL_NET=IB`、`NCCL_NET_PLUGIN=none`、`NCCL_IB_DISABLE=0`、`NCCL_ALGO=RING`、`NCCL_PROTO=SIMPLE`、`NCCL_DEBUG=TRACE`。channel 配置上限和下限均设为 2；逐操作实际 channel 数仍以 TRACE 为准，小消息可能只使用一个。仅 rank0 有上游逐操作选择日志，不给 rank1 补造对应字段。

`NCCL_NET=IB` 指定内部 verbs 网络，IB 不可用时应失败；核对脚本也拒绝实际 Channel 的 Socket/P2P 回退。NET/IB 支持 InfiniBand 或 RoCE；是否使用 GPUDirect RDMA 另看 GDR 标记，本实验不把 NET/IB 自动等同于 GDR。[官方环境变量说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/env.html#nccl-net)

## Crater r1 已确认的环境缺口

失败包已通过 SHA256 核对，两个 rank 的结果一致：GPU inventory 命令成功，CUDA 12.5.82/G++ 11.4 可用，RDMA device files 已挂载，两个 mlx5 端口为 ACTIVE/LinkUp；`libibverbs.so.1` 加载失败。探针的“没有可用 context”和“没有 opened device 上的 active port”由这个缺库问题连带产生，不能据此说物理端口未激活。

DMI/boot ID 摘要分别不同，支持节点分离判断；rank0 的归档包含两侧探针，证明本轮共享结果收集已工作。尚无 workload 编译、communicator 初始化、AllReduce 或 NET 数据传输证据。

Ubuntu 22.04 下需要补齐 `libibverbs1` 和 `ibverbs-providers`；`ibverbs-utils` 提供 `ibv_devinfo` 等诊断工具。用户随后使用 Crater“基于现有镜像构建”的 APT 字段准备派生镜像，重试中两侧已能加载 verbs 并打开两个 mlx5 device context。该方式生成新镜像，保留基础镜像；不要求修改平台的公共镜像。若改用运行中容器安装，仅在允许 root 安装且软件源可达时使用以下命令，两侧均需补齐；新 Pod 需要再次准备依赖。当前缺库无需重建已验证 NCCL 产物。[Ubuntu 运行库](https://packages.ubuntu.com/en/jammy/libibverbs1)、[provider](https://packages.ubuntu.com/ibverbs-providers)、[诊断工具](https://packages.ubuntu.com/jammy/amd64/ibverbs-utils)、[Crater 镜像构建说明](https://raids-lab.github.io/crater/en/docs/user/image/imageCreate/)

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libibverbs1 ibverbs-providers ibverbs-utils
```

首轮两侧 memlock 软、硬上限均为 65536 字节。它可能影响后续 MR 注册，但首轮只做预检、尚未尝试注册，不能当作已发生的注册错误。须结合进程锁内存权限和真实运行结果判断；硬限制受权限控制，不能保证容器内直接执行 `ulimit -l unlimited` 就能解决。[NCCL IB 排障](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/troubleshooting.html#infiniband)

## 补库后的重试与核对器修复

用户清理远端首轮目录和同名归档后复用了 `m1-rdma-r1`。本地分别保存首轮与重试，不能仅凭文件名区分；重试归档 SHA256 为 `22262421a9dfdb765ac3e3e0fc64a7c0d5c6e6d336e25bd1f29e045f24feb59d`。

重试两侧的 probe/build/workload 退出码均为 0，各完成九次 AllReduce。加载库与独立 TRACE 构建 hash 一致；两台不同物理节点、两张不同 GPU、同一 communicator、`nNodes 2 localRanks 1` 和 NET/IB 双向连接证据齐全。rank0 的实际 RING/SIMPLE 计划如下，channel 数以真实记录为准：

| 每次逻辑字节数 | 操作数 | 实际 channels |
|---|---|---|
| 16 B | 3 | 1 |
| 16 KiB | 3 | 1 |
| 4 MiB | 3 | 2 |

原 `analysis.txt` 报 `operation stages differ`。原因是 `std::unitbuf` 下的一串 `operator<<` 分多次写 stdout，NCCL Proxy 后台线程的完整 TRACE 行插入了应用记录的字段之间。它不是通信或数值失败；实际 rank0 在 Master、rank1 在 Worker，未发现角色参数填反。

`verify_m1.py` 现在仅恢复由探针精确 hostname 标识、由 `ncclIbTest` / `sendProxyProgress` / `recvProxyProgress` 完整 TRACE 插入的应用行。按原片段拼接，不补造任何时间、阶段或结果；非法、缺失、未知插入与未结束片段仍失败。插入的后台 TRACE 保留在派生视图的恢复行之后，不把派生视图用于 Proxy 时序或 M2 状态分析，也不移动 API/计划记录。原始归档、运行日志、失败状态和核对报告不修改。

本次 rank0 恢复两行、rank1 恢复五行；完整离线复核通过，18 项结果检查全部正确。可直接对同一批原始日志重新核对，无需 GPU 重跑：

```bash
python3 workloads/minimal_allreduce/verify_m1.py \
  .build/m1-download/r1-retry/unpacked/m1-results
```

原作业总退出码仍为 1，离线复核结果单独保存；技术证据通过与用户最终验收分别判断。两侧 memlock 仍为 64 KiB，但本轮实际通信未被阻塞，不为本轮验收新增修改平台限制的前置任务。后续 workload 已改为先组装完整应用行，再以一次 `fwrite` 写同一个 stdout FILE，避免在字段间交错；该 C++ 输出修复尚未在 Crater 编译运行，本次通过证据来自原运行和离线复核。

### 公开的真实摘录

仓库保存 [M1 脱敏摘录](../../results/samples/e06/m1/)，可从干净 checkout 离线复查：

```bash
python3 workloads/minimal_allreduce/verify_m1.py results/samples/e06/m1
```

`provenance.json` 记录原归档 SHA256、筛选和脱敏规则。摘录保留两侧原始退出码与失败报告、库来源、必要节点/连接记录、全部应用阶段与数值记录、NCCL 调用及 rank0 计划；保留被 TRACE 插入的原应用片段，不预先修正为完整行。无关的后台 TRACE、编译输出和设备环境细节未公开；路径、hostname、进程/线程 ID、GPU UUID、节点身份摘要、communicator ID、指针和 peer 地址使用占位符，保留必要相等/不同关系。operation、count、结果、阶段时间与实际算法/协议/channel 数未改写。该摘录是真实日志的筛选结果，不是生成的测试 fixture，也不用于 Proxy 时序或性能分析。

`run-status.txt` 和 `analysis.txt` 保留原失败；`offline-recheck.txt` 单独记录修复后复核通过。整个仓库检查脚本已包含这组真实证据的离线回放。

## Crater 配置

前提：两侧能读写同一个持久化用户空间目录；两侧镜像均有已验证 CUDA devel 环境以及可用 verbs runtime/provider。优先沿用 CUDA 12.5.82 镜像，但 Day15 曾出现 `libibverbs` 缺失，开启页面 RDMA 不能替代容器内库和 provider。探针失败时先处理对应环境缺口，不改成 Socket 验收。有限 memlock 不一定足够，真正的 MR/QP 注册还要由 NCCL 实际运行检查。[官方容器与 IB 排障说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/troubleshooting.html#infiniband)

| 项目 | Master | Worker |
|---|---|---|
| Replica | 1 | 1 |
| 进程/rank | 一个原生进程，rank0 | 一个原生进程，rank1 |
| GPU | 1 | 1 |
| CPU / 内存 | 建议 2 / 8 GiB | 建议 2 / 8 GiB |
| RDMA | 开启 | 开启 |
| 镜像 | 两侧使用同一兼容环境 | 同左 |
| 挂载 | 同一用户空间，保持已有挂载点 | 同左 |
| 工作目录 | 上传包所在的共享目录 | 同一共享目录 |
| Node | 选择节点 A | 选择另一个节点 B |

选择能分别配置 Role 启动命令的双 Role 作业。若使用此前的 PyTorch DDP Job 页面，只用它调度 Pod，程序不依赖 PyTorch、torchrun 或平台注入的 RANK。镜像要求 `/crater-start.sh` 时沿用已成功的包装方法；不要换成启动多个 rank 的 torchrun。

Target Node Control 的允许列表若包含多个节点，并不保证两侧分散。可以时两侧各限制到一个不同节点；提交后在作业详情核对两个实际 Node 字段。Pod 名或 hostname 不同不能证明物理节点不同。

本地打包（仓库根目录）：

```bash
python3 cluster/crater/scripts/package_m1.py
```

只上传 `.build/m1-upload/m1-rdma.tar.gz`。Windows 文件管理器打开 `\\wsl.localhost\Ubuntu-22.04`，进入本仓库 `.build/m1-upload`。本地接收目录 `.build/m1-download` 已准备；结果只下载一个归档，复制到该目录后由 Codex 解包核对。

已有 NCCL 产物必须包含 `include/nccl.h`、`lib/libnccl.so.2.21.5`、所需库链接和 `build-manifest.txt`；且是同一固定源码的 `TRACE=1` 构建。默认独立期望 hash 为已验证 Day15 TRACE 构建的：

```text
e4437dd0b48e3ab426b17b394043a0d162f1c9040b32bcf34b5b866e6286132a
```

两侧都替换下列 `<NCCL_TRACE_RESULTS>` 为该产物目录。重建库时必须从独立构建验证取得新 hash，再在两侧加相同的 `--expected-sha256 <已验证hash>`；不能为了通过核对而直接信任待检查的运行日志。

Master 启动命令：

```bash
bash -lc 'set -e
task_tmp=$(mktemp -d /tmp/m1-source.XXXXXX)
tar -xzf m1-rdma.tar.gz -C "$task_tmp"
python3 "$task_tmp/m1-rdma/cluster/crater/scripts/run_m1.py" \
  --rank 0 --run-dir "$PWD/m1-rdma-r1" --nccl-root "<NCCL_TRACE_RESULTS>"'
```

Worker 启动命令：

```bash
bash -lc 'set -e
task_tmp=$(mktemp -d /tmp/m1-source.XXXXXX)
tar -xzf m1-rdma.tar.gz -C "$task_tmp"
python3 "$task_tmp/m1-rdma/cluster/crater/scripts/run_m1.py" \
  --rank 1 --run-dir "$PWD/m1-rdma-r1" --nccl-root "<NCCL_TRACE_RESULTS>"'
```

两侧的 `run-dir` 必须指向同一个新目录。不要在两侧分别运行时间戳命令生成名字；重试时可以把两条命令都改为 `m1-rdma-r2`。也可以在旧作业两侧都结束、旧结果已保存后，清理精确的旧运行目录和同名 `.tar.gz` 再复用名称。runner 与打包器都拒绝覆盖已有目标，避免旧 ID/日志混入或覆盖结果。只启动一侧会超时，不会通过。

## 结果与解释

rank0 完成后，在共享工作目录生成 `m1-rdma-r1.tar.gz`。rank1 只写自己的证据，不要求另下载一个包：

```text
m1-results/
├── run-status.txt
├── run-config.json
├── analysis.txt
├── bundle-manifest.json
├── rank0/{status.json,probe.json,environment.json,build-manifest.txt,build-tail.log,run.log}
└── rank1/{status.json,probe.json,environment.json,build-manifest.txt,build-tail.log,run.log}
```

正常应有两个 rank 各九次结果正确、两侧版本 22105/加载路径/hash 正确、同一 communicator、两张不同 GPU、`nNodes 2 localRanks 1` 和 NET/IB 收发连接。rank0 的九条计划须为实际 RING/SIMPLE。应用标记仍是 CPU 对本地 stream 的观测，不是 M2 completion/state 采集。

`analysis.txt` 自动核对通过时打印 `m1_technical_checks=PASS`；M1 最终仍需用户核对 GUI 的两个实际物理 Node、观察证据并验收。若 DMI UUID 在容器内不可读取，自动核对可能因缺少物理节点证据而失败；核对 GUI 后可以在本地补做以下显式检查，不需再运行 GPU：

```bash
python3 workloads/minimal_allreduce/verify_m1.py \
  .build/m1-download/m1-results --confirm-distinct-nodes
```

未确认 Node 时不要使用此选项；相同的 DMI/boot 身份、`nNodes 1`、同一 GPU 或网络回退仍会被拒绝。原始失败状态保留，后续核对结果用于补证，不修改旧日志。

先看失败层次：`probe.json` 是资源可见性；`build-tail.log` 是 workload 编译；`run.log` 是 ID/bootstrap、初始化、API/stream 与结果；`analysis.txt` 是证据核对。资源可见不等于 RDMA 连通，Job Completed 不等于 M1 完成。普通探针、编译、通信和核对失败都会打包已有证据；Pod 被 SIGKILL、驱逐或存储完全不可写时，进程无法保证最终归档，保留持久化目录排障。

## 本地检查

```bash
python3 -m unittest discover -s workloads/minimal_allreduce/tests -p 'test_*.py' -v
python3 -m unittest discover -s tests/nccl -p 'test_m1*.py' -v
python3 -m unittest discover -s instrumentation/nccl-2.21.5/day16/tests -p 'test_*.py' -v
```

本地测试只验证 CPU 设备选择约束、证据解析、失败保存和打包；构造的测试日志不能作为 M1 运行证据。真实 CUDA 编译、双节点通信和节点分布仍须在 Crater 验证。
