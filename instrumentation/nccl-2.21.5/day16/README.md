# M2 最小采集：保留的 Day16 框架

当前状态：进程内有界记录器、NCCL adapter/补丁、独立源码准备与构建入口、双 rank 启动、结果核对及单文件打包已实现。Crater 插桩库构建与加载检查已通过；r3 双物理节点 NET/IB 采集及本地独立技术核对通过，18 项数值结果正确，每侧九条 completion、周期中间进度齐全且零丢失。完整 M2 验收待用户复核，当前无需重跑。导出包含 Event v2、原始发送/接收状态及身份/plan 元数据。

执行依据为 [v0.5 核心复现计划](../../../docs/plans/Mycroft_26日开发路线图.md)。M1 双节点 NET/IB 路径已验证，用户已授权进入 M2 普通 RING/SIMPLE 最小采集。Graph、grouped collectives、完整 P2P GPU 标记/managed memory 方案均已移出当前范围。

## 本轮实验设计与构建过程

研究问题：在 M1 已验证的双节点 NET/IB 路径上，能否把真实操作身份、中间通信进度和本地整体完成可靠关联？本轮仅加入插桩，继续串行执行 float32 Sum AllReduce；RING/SIMPLE、两个物理节点、每节点一个 rank/GPU 与 M1 一致，不注入故障。16 B、16 KiB、4 MiB 各运行三次，第 k 次两端输入分别全为 `1+k`、`2+k`，预期输出全为 `3+2k`。

M2 通过要求：两侧九次结果均正确，每 rank/op 恰有一条 completion，实际 channel/peer/消息大小一致；至少一个操作有周期中间窗口；丢失为零，错误、不支持形式或缺证使核对失败。采样周期初值 10 μs，Proxy 每次进度回调检查是否到期，最终状态强制保存。进度未变仍保存到期快照；这不是独立定时线程，调度延迟会使间隔大于设定周期。后续软件延迟必须让进度回调继续运行。若正常运行太短而没有中间窗口，如实拒绝该次 M2 验收，再根据实际数据调整周期/消息量。

构建链：

```text
固定 NCCL 2.21.5 原源码包
  → prepare_m2.py：校验文件，解出新副本，应用 patch，复制两个 host 源文件
  → build_m2.py：make 调用 g++ / nvcc，链接成 libnccl.so.2.21.5
  → build.sh：g++ 编译原生 AllReduce 程序，链接指定库
  → run_m2.py：两端加载同一插桩库，实际通信并逐元素检查
  → Proxy 线程退出后导出 → verify_capture.py 核对 → rank0 打一个结果包
```

`nccl-2.21.5-m2.patch` 修改上游调度、Proxy、NET 和 Makefile 中的调用位置；`mycroft_m2.cc` 读取真实字段、记录实际 launch stream 上的 CUDA event，并检查 abort/async error；`operation_trace.cpp` 负责预分配内存、计数与停止后导出。所有新增实现由本项目独立编写，没有复制 Mycroft 实现。包内 NVIDIA 源码及 `LICENSE.txt` 来自固定 commit `ab2b89c4c339bd7f816fbc114a4b05d386b66290`，按其许可证保留；archive 仅包含构建所需源码、makefiles、扩展头及许可证，不含上游发行包目录。

只有两个新增 CPU 源文件使用 C++17；共享 adapter 头保持 C++11 兼容，GPU 编译参数沿用上游。新增成员在 host Proxy 对象中，不修改 GPU work 格式。原始 `third_party/nccl` 保持干净；M2 的 manifest 明确 `source_clean=no`，不能套用 M1 的 clean-source 检查或旧库 hash。

依赖：构建需要 `python3`、`git`、`make`、`g++`、`readelf` 和 CUDA 12.5 devel 的 `nvcc`、头文件、静态 CUDA runtime；不需要 GPU，不依赖 CMake、PyTorch 或 pip 包。`RDMA_CORE=0` 使用 NCCL 自带的动态 verbs 包装，编译不新增 verbs 开发包。真实运行使用 M1 已验证的镜像，仍需要 `libibverbs1`、provider、GPU 驱动与实际 GPU/RDMA 资源。

来源核对分三层：上传 manifest 保存原源码包/patch/新增文件 hash；构建记录保存实际命令、准备后源码 hash 和新库 hash；运行再检查实际加载路径/版本/hash。通用记录器自身不具备核验 NCCL 的能力，所以 `capture-manifest.json` 和原始 Event 的 source 始终标为未验证；adapter manifest 是生产端观测声明，只有离线核对器结合构建、连接、数值和 raw/Event 对账后才给出技术检查结果，不原地改写证据。completion 是本地整体操作的 CPU 观测时刻，不是精确 GPU 时间，也不是跨 rank 的全局完成。

### 上传与运行

首轮 `m2-capture-r1` 的退出码 1 来自物理节点核对：两侧各九次数值检查正确，adapter 无错误且记录零丢失，独立核对采集身份、进度及 completion 通过；两侧 DMI 和内核启动标识均相同，不能算双物理节点实验。NCCL 2.21.5 的 `getHostHash()` 包含 hostname，因此两个容器的不同 hostname 也可能使 NCCL 日志显示 `nNodes 2`，该数字不能替代物理节点证据。保留 r1，下一轮须实际落在不同物理 Node，两侧同时使用新采集编号；沿用已有上传包和 `m2-build-r1/nccl/`，不重复构建。详细证据仍记录在当前计划。

2026-10-03 返回的 `m2-build-r1.tar.gz` 已通过来源离线复核：插桩库编译、指定路径加载、版本 `22105` 和三个采集接口导出检查均成功；构建作业本身为 `gpu_execution=NOT_RUN`。r2 普通用户运行在 CQ 创建时报 ENOMEM，随后 SIGSEGV；两侧改用 root、复用原包和库的 r3 已成功，双节点采集和本地技术核对通过，详细证据与 hash 记录在[当前计划的 M2 状态](../../../docs/plans/Mycroft_26日开发路线图.md#m2最小-completionstate-log)。r2/r3 对应的 kernel boot 标识相同，memlock 均为 64 KiB；不能解释成限制被提高，也未直接测得有效 capabilities。已有上传包和远端 `m2-build-r1/nccl/` 保留，无需重复构建或重跑。r2 的两个 null DMI 标识表示不可读取；r3 已读取到不同的 DMI 和 boot 标识。

本地只生成一个上传包：

```bash
python3 cluster/crater/scripts/package_m2.py
```

将 `.build/m2-upload/m2-experiment.tar.gz` 上传到共享目录中的 `M2/`。启动命令会在容器临时目录解压，不需要手工解压或批量上传。下面的 `<共享挂载点>` 替换为 Crater 用户目录，避免在公开文档写个人绝对路径。

以下 CPU 构建已完成，当前重试无需执行。首次从零准备时创建 CPU 作业（Custom 类型、2 CPU、8 GiB、0 GPU、不申请 RDMA），复用 CUDA 12.5 开发镜像，工作目录设为 `<共享挂载点>/M2`，命令：

```bash
bash -lc 'set -e
m2_tmp=$(mktemp -d /tmp/m2-upload.XXXXXX)
tar -xzf m2-experiment.tar.gz -C "$m2_tmp"
python3 "$m2_tmp/m2-experiment/cluster/crater/scripts/build_m2.py" --work-dir "$PWD/m2-build-r1" --jobs 2'
```

成功后插桩库位于 `m2-build-r1/nccl/`。源码副本与对象文件在容器临时目录生成，结束后清理；共享目录保留最终共享库、头文件和证据，不保存不需要的静态库。失败时只下载 `m2-build-r1.tar.gz`，由 Codex 检查编译证据，不逐个下载中间文件。新工作目录及结果包不得已存在；保留旧结果并选新编号。

用户已确认当前 Crater 的 PyTorch 双 Role 页面只有整个作业共用的节点白名单，它不能强制两侧分开；Custom 单机批处理页面同时支持 RDMA 和节点白名单。因此下一轮用两个 Custom 作业共同启动一轮 M2，分别固定到不同物理节点。只调整调度方式；已有 runner 按显式 rank 参数启动，通过共享目录交换 NCCL ID，不依赖 PyTorch 服务。跨作业实际通信与双节点证据仍由本轮核对，不能因页面允许配置就预先宣称通过。

| 配置 | rank0 Custom 作业 | rank1 Custom 作业 |
|---|---|---|
| 副本数 | 1 | 1 |
| 资源 | 2 CPU、8 GiB、1 V100、1 RDMA | 相同 |
| 节点白名单 | 只选节点 A | 只选不同的节点 B |
| 镜像 | M1 已验证的 CUDA 12.5 RDMA 运行环境 | 相同 |
| RDMA 拓扑 | 两节点均支持的同一 IB 拓扑 | 相同 |
| 用户空间挂载点 | `<共享挂载点>` | 相同 |
| 工作目录 | `<共享挂载点>/M2` | 相同 |

用户确认 r2 两侧 Custom Shell 均为普通用户，且页面支持 root。已完成的 r3 对照两侧均选择 Bash 的 root 身份，资源、固定节点及库保持原配置；CQ 错误消失、初始化正常返回、采集正常启动。root 不等于有效 `IPC_LOCK`，具体权限原因未直接测量。以下保留已执行 r3 的单行启动示例；当前无需重跑，未来再次运行须两侧同时换新编号。启动框不再外包跨行的 `bash -lc` 单引号字符串；已有报错 `zsh:5: unmatched` 表明收到的启动文本有单引号未闭合，不能靠换节点解决。单行中的 `&&` 保证前一步失败时不启动后一步。这里只核对本地 Bash/POSIX shell 语法，本地未安装 zsh；原始远端接收到的完整命令尚未取得，因此不把复制遗漏或页面拆行的具体原因写成已确认。

rank0 命令：

```bash
m2_tmp=$(mktemp -d /tmp/m2-upload.XXXXXX) && tar -xzf m2-experiment.tar.gz -C "$m2_tmp" && python3 "$m2_tmp/m2-experiment/cluster/crater/scripts/run_m2.py" --rank 0 --run-dir "$PWD/m2-capture-r3" --nccl-root "$PWD/m2-build-r1/nccl" --peer-timeout 900
```

rank1 命令：

```bash
m2_tmp=$(mktemp -d /tmp/m2-upload.XXXXXX) && tar -xzf m2-experiment.tar.gz -C "$m2_tmp" && python3 "$m2_tmp/m2-experiment/cluster/crater/scripts/run_m2.py" --rank 1 --run-dir "$PWD/m2-capture-r3" --nccl-root "$PWD/m2-build-r1/nccl" --peer-timeout 900
```

尽快连续提交两个作业，不等待一侧完成再提交另一侧。两侧等待就绪的上限设为 900 秒，以容纳分别提交与调度的间隔；它不能解决节点资源不足造成的长期排队，原生通信子进程的执行上限仍为 240 秒。使用相同的新 run-dir；rank0 发布配置和 NCCL unique ID，两侧交换就绪状态再通信。rank0 完成或收集失败结果后打出 `m2-capture-r3.tar.gz`，只下载此文件。本地接收目录为 `.build/m2-download/`，Windows 从 `\\wsl.localhost\Ubuntu-22.04` 进入仓库同名目录。镜像若要求 `/crater-start.sh`，沿用已成功的包装方法，不改为 torchrun。

`m2-build-r1/nccl/` 继续读取复用；两侧临时解包和 workload 构建分别使用独立临时目录，日志/采集分别写入 `rank0/`、`rank1/`。保留首轮 `m2-capture-r1` 与结果包。已完成的两作业共同使用 r3；再次重跑时两侧一起换新采集编号，不在已有目录重试或覆盖。平台提供失败自动重启选项时应关闭。rank0 原子创建运行目录及配置，结果包也拒绝覆盖。现有配置没有平台作业身份绑定：若误用只留下配置的旧目录，rank1 可能读取旧配置再等待超时，因此目录拒绝覆盖不能代替每轮使用新目录。

本地已对现有上传包模拟检查两侧启动先后、双 rank0/双 rank1 竞争，以及旧目录、旧结果包和完成后重复启动的处理；也确认了上述旧配置限制。模拟替代了 GPU 与编译子进程，只验证启动协调和文件保护，不作为 GPU 采集证据。本次命令和说明更新不改变插桩代码，不需要重新上传已有包或重编译库。

真实包解开后，本地复核命令：

```bash
python3 instrumentation/nccl-2.21.5/day16/verify_capture.py <解包目录>/m2-results
```

预期打印两侧 completion/sample/window/loss 信息及 `m2_technical_checks=PASS`；这表示技术证据通过，用户观察关键结果后才更新 M2 验收。当前 r3 的 CUDA 构建来源、双节点 GPU 通信、采集及字段对账已通过；完整 M2 验收待用户复核。

## 框架职责与实现边界

目标：将目标 NET 路径的操作身份、中间进度和本地整体完成组织成可核对的记录，使后续分析能区分“GPU 尚未准备”“请求尚未提交”“网络尚未完成”。记录器及接入代码已实现，真实 CUDA 构建与加载、双节点 GPU 运行及技术核对均已通过，完整 M2 验收待用户复核。

输入：当前记录器接收结构化 C++ 记录和调用者提供的观测 ns；真实适配使用固定 NCCL 2.21.5、原生 workload 与 M1 已验证 NET/IB 路径。M2 最终输出为身份/channel/peer 关联、基于同进程单调时钟的 completion、周期 state、必要接收证据、构建来源及运行状态。CPU fixture 的字段是手写数据，只验证记录器。

最小知识：一次 collective 可拆到多个 channel；API 返回和 Proxy 子任务完成与整个本地 collective 完成有不同语义；单调时间在各进程内使用；operation 逻辑字节数与单个 slice 的传输大小不同。

框架检查命令见下方“本地检查”；预期为 CPU 记录器、契约、来源核对负例和真实源码打包/patch 准备检查通过。M2 最终验收必须来自目标路径的项目修改版 NCCL：两个 rank、普通连续多个操作、实际多 channel 身份、时间/大小正确、数值结果正确，完成条件有可追溯的真实依据，并包含周期 state log 与必要对端状态。具体五步及完成标准只在路线图定义。

## 文件和职责

```text
instrumentation/nccl-2.21.5/day16/
├── include/operation_trace.h  # 身份、连接、发送/接收、plan、采集生命周期接口
├── src/operation_trace.cpp    # 有界并发写入、周期辅助、关联检查和停止后导出
├── verify_capture.py         # 构建/运行来源及 raw/Event/应用操作对账
├── nccl_adapter/             # NCCL host hook 与可复查 patch
├── check_framework.sh        # CPU 记录器、schema、契约和打包检查
└── tests/                    # contract/runtime/bundle 测试及明确标记的 CPU fixture

cluster/crater/scripts/package_day16_results.py
```

`OperationKey` 包含 `comm_hash`、`op_seq_candidate` 和 rank。`op_seq_candidate` 尚未确认，M2 须动态验证普通 NET/IB 操作的跨 rank 对应；Graph replay 不受支持。`record_channel_membership()` 将同一个 key 与各 channel 关联；operation 级 completion 的 Event v2 `context.channel` 为 null。

Event 的 collective 名称沿用仓库现有 `all_reduce`，进度与 completion 使用 `nccl_monotonic_ns`；来自 E02 的 tick 即使格式合法，也不能作为 M2 实测日志。

`record_operation_begin()` 保存操作逻辑大小与开始时间；`record_operation_completion()` 必须由经过确认的本地整个操作完成观测点调用。begin 与 end 使用同一进程单调时钟；`record_plan_binding()` 记录实际单操作 plan 和执行 stream，completion 引用相同编号。调用 completion 前必须检查该 stream 上 CUDA event 成功、abort 未请求和异步错误状态成功。编号只在本进程内有效，不能导出指针冒充跨 rank 身份。

`record_peer_connection()` 将 operation/channel 与 communicator peer、send/recv 方向和实际 NET/IB 连接绑定；不能未经转换使用 transport 的 `tpRank`。两端使用同一 capture ID 与 `commHash` 形成 communicator ID，按操作、channel 和 peer/direction 对账；不直接比较两端的本地 connection ID。

| 记录 | 字段/含义 | 后续用途 |
|---|---|---|
| `SendProgress` | `gpu_ready_steps`、`transmitted_steps`、`done_steps`、`nsteps` | readiness 首次成立、isend 非空 request、网络 test 完成的累计进度 |
| `ReceiveProgress` | `posted_steps`、`received_steps`、`transmitted_steps`、`done_steps` | irecv 非空 request、接收完成、flush 就绪后向 GPU 发布、GPU 消费确认 |
| 两类状态共有 | 实际 channel/connection、观测 ns、step base、slice steps、注册缓冲区标记 | 保留来源与单位；step base 不加进 operation-relative 进度 |
| `CaptureConfig` | capacity、sample period ns | 初始化预分配；周期到期即采样，包括进度未变的窗口 |
| `CaptureStats` | phase、recorded、dropped、invalid、unsupported | 停止后的一致统计；无效调用不改写输出 |

`normalize_send_steps()` 已实现普通发送侧 step→slice 格式检查。例如 `(nsteps, ready, transmitted, done)=(16,8,6,4)`、`slice_steps=2` 对应 Event `(total_chunks,gpu_ready,rdma_transmitted,rdma_done)=(8,4,3,2)`。所有量必须是非负整数且整除同一个 slice steps，满足 `done≤transmitted≤ready≤nsteps`，不静默舍入。这里的 `chunks` 是归一化 Proxy slice 单位，不能直接当 Ring 算法数据块或逻辑字节数。

上述例子是格式说明，不是实测。NCCL 注册缓冲区路径可在运行中增长 `nsteps`，且 GPU/网络计数可能不一一对应；接口保留该标记，当前记录器返回 `Unsupported` 并计数，转换器也拒绝其未经验证的映射。本轮限定普通未注册缓冲区，不增加注册路径实验。接收状态独立导出，现有发送侧 Event payload 不承载其不同语义；固定源码的 sub 结构有 `flushed` 字段，但目标 NET 路径未独立更新它，因此不把它作为有效采集计数。

生命周期已实现：`initialize_capture(config)` → 各 record/sample 入口 → 所有生产者退出 → `stop_capture()` → 读取统计 → `export_capture(capture_id, output_directory)`。初始化时分配所有记录槽；多个生产者用无锁原子预约独立槽，不等待 reader、不覆盖旧记录。记录追加到运行末尾，容量耗尽后返回 `Dropped` 并计数；重新初始化会清空上一轮内存，因此需要先导出。此实现要求 Linux 64 位无锁原子。

`sample_send_progress()` / `sample_receive_progress()` 使用生产者独占的 `SamplingState`：首次观察即记录，周期未到返回 `Skipped`，周期到期即使累计值不变也记录；时钟倒退返回 `InvalidArgument`。每个 operation/connection/direction 用独立 state。它们不创建线程、不读取时钟；adapter 在持续调用的 Proxy 回调中取得 CLOCK_MONOTONIC，同进程应用阶段也使用该时钟。调用者必须检查返回状态。

停止和导出由控制线程串行调用，且所有生产者已经退出/静止；不允许一边写一边 stop 或重新初始化。`read_capture_stats()` 只在停止后提供一致结果。JSON 格式化、关联索引和文件写入全部在停止后执行，不进入热路径。进程崩溃且尚未导出时可能缺证。

导出要求新目录且父目录已经存在，一个进程只承载一个 rank/communicator；run_m2.py 收集两个 rank 的独立输出并打包。`channel-map.jsonl` 保留 begin、channel、peer，`plan-map.jsonl` 保留 plan 与原始 completion，`send-/recv-rankN.jsonl` 保留 step 单位原始计数，`rankN.jsonl` 为归一化发送进度和 completion Event。缺失/重复 begin、错误 plan/stream、缺少 peer/direction 或累计计数倒退时，原始记录保留；不能可靠关联的 Event 不生成，manifest 的 `association_errors` 增加。导出遇到 I/O 错误返回 `IoError` 并可能保留部分文件，完整 manifest 最后写；已有目录拒绝覆盖。

manifest 的 `local_contract_valid` 只描述本记录器的格式/关联检查；丢失、无效、不支持或关联错误都会使其为 false。无论 CPU 检查是否通过，当前固定写入 `nccl_adapter_verified=false`、`completion_observation_verified=false`、`diagnostic_eligible=false` 和 `time_source=caller_supplied_unverified`，不把手写观测伪装成真实 NCCL 证据。

当前单 RING 路径要求每个 operation/channel 恰有一个 send 和一个 recv peer 映射。重复或多个同方向连接会报告缺证，不把多个连接的进度混进同一个 channel Event；adapter 检查实际 NET transport 与 communicator peer，真实运行仍须验证。

M2 剩余验证：操作/channel/peer 身份、实际周期窗口、计数单位与实际 stream 完成观测。接入、来源记录、结果合并及核对入口已有代码，CPU 检查不能代替这些实测。目录名 `day16/` 和既有工具名保留以兼容调用，不表示继续旧逐日路线。

## 已知路径限制：P2P 不必有 NET Proxy 进度

Day15 TRACE 的实际选择是 RING/LL（16 B、16 KiB）及 RING/SIMPLE（4 MiB），连接为 P2P/CUMEM。固定源码中：

1. `src/enqueue.cc:176` 的 `addProxyOpIfNeeded()` 调用 `ncclProxySaveOp(..., &needed)`；只有 needed 为 true 才将条目放进计划的 Proxy 队列。
2. `src/proxy.cc:525` 的 `SaveProxy()` 在 `connector->proxyConn.proxyProgress == NULL` 时直接返回；`ncclProxySaveOp()` 先把 needed 设为 false。
3. `src/transport/p2p.cc:756` 的默认 P2P transport 没有 `proxyProgress`；显式开启 P2P CUDA memcpy 路径时，发送侧可以另行安装 progress 函数。这不等同于 NET 的进度回调。

这些是源码结论；实际每个 channel 是否进入 Proxy 队列仍要在插桩运行中观察。因此仅在 T2/T3 记录 Proxy op，不能无条件覆盖 Day15 P2P workload，也不能据此宣称已经得到所有操作的完成记录。

已沿 `addCBDCollToPlan()` → `appendWorkElemColl()` → kernel plan → `ncclLaunchKernel()` → GPU work 读取源码，下节保留结论。这说明 Day15 workload 不能直接验证 NET 的三进度量，所以 M1 改用真实双节点 NET/RDMA；本版不实现 P2P 专用采集器。`ncclLaunchFinish()` 在 host 侧释放/连接 stream 等依赖，其函数名不证明 GPU 已执行完。M2 完成观测限定普通单操作计划，仍须确认实际 stream、abort/error 和覆盖范围。

## 已完成的源码追踪：身份怎样经过多个 channel

已读取固定版本源码；下表是源码结论，不是新增 GPU 运行证据。

| 位置 | 状态或数据变化 | 对插桩的意义 |
|---|---|---|
| `enqueue.cc:438`，`addCBDCollToPlan()` | 在 channel 循环前取 `plan->collOpCount++`，左移一位作为 Proxy collective 标签值 | 同一 collective 的各 channel 使用同一 plan-local ordinal；下一 collective 才递增 |
| `enqueue.cc:461`，`setCollWorkElem()` / `appendWorkElemColl()` | 先设置该 channel 的 count/offset，再拷贝 work 到 channel 队列 | 可以关联 operation 与实际参加的 channel；逻辑总字节数仍取 `collInfo->nBytes` |
| `include/device.h:227`，`ncclWorkElem` | 有 count、workCount、workOffset 等执行字段，没有 `commHash` 或完整 operation 序号 | 不能假设 GPU work 已自动携带我们的完整操作身份，需要另建关联或可复查扩展 |
| `enqueue.cc:565`，`finishPlan()` | 从非空 work 队列构造 channelMask/channelCount；Proxy 队列是否非空另算 hasProxyOps | work 与 Proxy 是两类队列，不能用 Proxy 队列为空判断 GPU 没有工作 |
| `enqueue.cc:1351`，`ncclLaunchKernel()` | grid.x 等于计划 channelCount，参数包含 channelMask 与 workHead | launch 对象是整个 plan，不无条件等于单个 collective |
| `device/common.h:125`，`ncclKernelMain()` | GPU block 根据 mask 中第几个置位映射到 channel，读取该 channel 的 work | blockIdx.x 是压缩后的 block 编号，不总等于 channelId |

例如某个 mask 只包含 channel 0、2、5，GPU blocks 0、1、2 分别处理 channel 0、2、5。必须记录实际 channelId，不能直接记录 blockIdx.x。这是解释映射的示例，不是 Day15 实测的 channel 集合。

`appendWorkElemColl()` 可以把多个 collective work element 合进同一个 `ncclWork`，一个 plan 也可以容纳多个 collective。于是 operation、work element、work FIFO slot、kernel launch 都有不同粒度。观察整个 kernel 完成可以覆盖该 plan 中的工作，但不能据此声称其中各操作有不同的精确 GPU 结束时间。

### 普通执行与 Graph 执行的序号边界

`uploadProxyOps()` 开头先保存旧的 `comm->sharedRes->collOpCount` 为 base，再加上 `plan->collOpCount`，随后才遍历 Proxy 队列。有 Proxy collective 条目时：

```text
plan_local_ordinal = oldId >> 1
historical_candidate = base + plan_local_ordinal
```

因此满足 `!persistent && persistentRefs == 0 && !ncclCudaLaunchBlocking` 的普通分支，会在 `ncclLaunchKernelAfter_NoCuda()` 调用 `hostStreamPlanTask()`，继而执行这次 base 推进；即使 Proxy 队列为空，推进仍发生，只是没有 q 条目供 T2 循环采集。不能把“无法从 Proxy 条目取到 key”说成“历史计数一定没推进”。多 plan 调度时，base 应绑定到该 plan 的实际推进时刻，不能在提前构建所有计划时反复读取同一个旧 base。

Graph capture 使 plan.persistent 为 true。`ncclLaunchPrepare()` 在这条分支仅对 hasProxyOps 的 plan 安排 `hostStreamPlanCallback()`；`ncclLaunchKernelAfter_NoCuda()` 的普通调用分支也被绕过。所以无 Proxy 的 persistent plan 不能直接依赖这条既有回调为每次 replay 分配新序号。未来扩展 Graph 支持时需要另行验证执行实例身份；不能把 capture 时的 plan-local 编号永久当作每次 replay 的 operation 编号。本版只保留这一限制，不安排 replay 实验。

此外，Day15 INFO 的 `AllReduce: opCount` 来自 `comm->opCount`。它在 `ncclProxyStart()` 中按调用推进，和这里的 collective 历史计数不是同一个字段。Day15 串行单 collective 场景中显示 0—8，不构成其在 grouped collectives 或 Graph replay 中一一对应 operation 的证明。

### 完成信号：workFifoDone 为什么不够

`device/common.h:181` 的顺序是：

```text
读取 work 描述到 GPU shared memory
    → workFifoDone = doneAcks（最后一个 FIFO work 的读取确认）
    → SpecializedRunWork().run(...) 或 ncclDevFuncTable[...]()
    → block 内同步，继续下一个 work 或退出
```

`workFifoDone` 用于 host 端 `waitWorkFifoAvailable()` 回收描述符槽位；它在当前 work 的通信计算前更新，故不能作为该 channel 的通信完成，更不能作为整个 CollOp completion。头部的 isLast 只是该 channel 在本次 kernel 的最后一份 work 描述。

后续完成观测应覆盖整个目标操作的 GPU 工作及全部参与 channel，并和身份关联；必须处理 abort/error，避免把提前退出当成成功完成。CPU 观测时间可用于本进程时差，但要明确是观测时刻，不能伪称精确 GPU 结束时间。`ncclLaunchFinish()` 和 Proxy request completion 同样不能直接替代这个条件。

adapter 已实现普通单操作 plan 的实际 launch stream event 观测；仅以 patch 修改独立副本，不改原始 checkout。上述 Graph/P2P 结论保留为支持范围限制。M1 实际 NET/IB 证据已取得，M2 仍须动态确认身份与 completion/state 语义。

## 本地检查

从仓库根目录执行，不需要 CUDA/GPU：

```bash
bash instrumentation/nccl-2.21.5/day16/check_framework.sh
```

框架检查运行契约/记录器/打包测试并打印 `day16_framework=PASS scope=CPU_recorder_schema_and_contract_tests_only`、`day16_capture=CPU_RECORDER_ONLY`、`day16_nccl_adapter=PATCH_PRESENT_CUDA_BUILD_NOT_RUN`、`day16_gpu_acceptance=NOT_RUN`。其中 CPU runtime 编译并运行实际 C++ 记录器，覆盖多生产者、容量耗尽、进度不变的周期快照、非法状态/计数和导出缺证；手写记录不产生 NCCL 验收证据。

上述 `CUDA_BUILD_NOT_RUN` 仅描述本地框架检查没有执行 CUDA 构建；它不读取远端构建结果。已取得的 Crater 构建证据见本文“上传与运行”及当前计划，GPU 采集仍未验证。

需要自行观察保留的两个相同快照时，可运行明确标记的 CPU fixture：

```bash
mkdir -p .build/m2-cpu
g++ -std=c++17 -pthread -Wall -Wextra -Wpedantic -Werror \
  -I instrumentation/nccl-2.21.5/day16/include \
  instrumentation/nccl-2.21.5/day16/src/operation_trace.cpp \
  instrumentation/nccl-2.21.5/day16/tests/capture_fixture.cpp \
  -o .build/m2-cpu/capture-fixture
.build/m2-cpu/capture-fixture normal .build/m2-cpu/normal-1
cat .build/m2-cpu/normal-1/trace/rank0.jsonl
cat .build/m2-cpu/normal-1/capture-manifest.json
```

首两条 progress 时间分别为 120、130，归一化 `(ready,transmitted,done)=(4,3,2)` 不变，两条均保留；随后进度推进，最后有一条手写 completion。目录已存在时请选择新的输出名。这说明停滞窗口没有被“只记录变化”丢掉，不证明真实 GPU/NIC 停滞。

真实核对入口要求 M2 构建和实际运行证据；给旧 M1 日志或 CPU fixture 会拒绝，不用它们制造通过：

```bash
PYTHONPATH=src python3 instrumentation/nccl-2.21.5/day16/verify_capture.py \
  results/samples/e06/m1
```

该负例应返回 1 并说明需要 M2 provenance。真实核对同时检查 patch/build hash、观测声明、丢失计数、workload 结果与逐操作身份，不把合法 Event JSON 当成真实完成证据。

## 单个结果包与接收目录

本地 `.build/day16-upload/` 已提前创建。Windows 文件管理器先打开 `\\wsl.localhost\Ubuntu-22.04`，进入本仓库的 `.build/day16-upload`；用户以后仅下载并复制一个 `day16-results.tar.gz`，由 Codex 解包核对。

已实现的打包工具接受以下结果布局，缺失项会记录在包内 `bundle-manifest.json` 中；`run-status.txt` 必须存在：

```text
<RESULT_DIR>/
├── run-status.txt
├── environment.txt
├── capture-manifest.json
├── analysis.txt
├── workload/{run.log,rank0.log,rank1.log}
├── trace/{rank0.jsonl,rank1.jsonl,send-rank0.jsonl,send-rank1.jsonl}
├── trace/{recv-rank0.jsonl,recv-rank1.jsonl}
├── trace/{channel-map.jsonl,plan-map.jsonl}
└── nccl-build/{build-manifest.txt,run-status.txt,verification.log,saved-verification.log}
```

将其打成新文件：

```bash
python3 cluster/crater/scripts/package_day16_results.py "<RESULT_DIR>" \
  --output "<RESULT_DIR>.tar.gz"
```

包内统一以 `day16-results/` 为根，保留目录结构。仅包含明确列出的必要文本证据；如果有编译日志，仅保存末尾 128 KiB。不会附带库、头文件或中间产物，也不另写必须下载的 SHA 文件；archive SHA256 打印在终端。已有包拒绝覆盖，先写临时文件再原子发布。

失败作业同样能打包已产生的日志，包内保留原始非零退出状态；打包成功不代表实验通过。未来真实采集运行脚本必须在退出时先保存作业状态，再调用此工具。目前还没有该运行脚本或可运行的插桩包，不应为这份骨架创建 GPU 作业。
