# Day15：原生同机双 rank AllReduce

当前已在 Crater 的 CUDA 12.5 环境使用 g++ 完成真实编译，并在两张 Tesla V100-SXM2-32GB 上通过双 rank 功能验证。两个 rank 各完成九次 AllReduce，全部元素校验通过，退出码为 0，首轮加载的库版本、路径和 hash 与 Day14 产物一致。随后独立 TRACE 构建与相同 workload 运行也已通过，补齐 rank0 逐操作算法/协议选择；连接记录为 P2P/CUMEM。真实技术证据已核对齐全，当前准备 Day15 提交与 Day16 任务交接。本地 WSL 没有 CUDA Toolkit，离线日志核对在 WSL 完成。

目标：在同一个 Pod 的两张 GPU 上，用两个原生 C++ 进程运行 Day14 构建的 NCCL 2.21.5。输入是 Day14 产物目录和迭代次数；输出是每个 rank 的库来源、应用阶段标记和数值检查结果。

## 程序结构

- `CMakeLists.txt`：只从显式 `NCCL_ROOT` 查找 NCCL，链接 CUDA runtime；不需要编译自己的 CUDA kernel。
- `build.sh`：直接使用 C++17 编译器构建，检查 CUDA 头文件及 runtime 开发库，并设置 NCCL/CUDA 加载路径。Crater 无需安装 CMake；CMake 入口仍可手动使用。
- `include/bootstrap.h`、`src/bootstrap.cpp`：rank 0 原子发布 ID 的原始字节，rank 1 有限等待并读取；仅在同一个 Pod 的私有临时目录使用，拒绝覆盖旧 ID。
- `src/main.cpp`：版本及加载路径检查、GPU 绑定、communicator 初始化、输入准备、AllReduce、stream 同步和阶段标记。
- `include/result_check.h`：CPU 检查全部输出元素，遇到首个错误时报告索引、实际值和期望值。
- `run.sh`：在新建的 `/tmp` 目录调用 `build.sh` 编译，并发启动两个 rank，保存独立日志；每个进程由 120 秒 timeout 限制。失败不会打印整体通过。
- `tests/test_bootstrap.py`、`tests/test_result_check.py`：真实文件 IPC、错误路径和实际数值校验函数测试，不调用 CUDA 或 NCCL。
- `cluster/crater/scripts/package_day15.py`：仅打包 workload 源码和运行工具，供 Windows 上传；复用 Crater 上的 Day14 产物。
- `cluster/crater/scripts/run_day15.sh`：保存环境、库 hash、构建与运行日志、两个 rank 的独立日志及退出码到新的持久化结果目录。
- `verify_results.py`：在本地离线核对真实结果，输出明确的失败原因，并可导出逐操作阶段耗时 CSV。默认核对功能基线，`--require-selection` 额外检查 rank0 的逐操作提交计划；用户还需理解这些观测的语义。
- `cluster/crater/scripts/package_day15_trace.py`、`run_day15_trace.sh`：打包同一固定 NCCL 源码，在 `/tmp` 编译独立的 `TRACE=1` 库，保存构建证据并运行相同 workload，最后核对逐操作选择日志。

本次固定 `device=rank`，要求两个进程均能看到同样的两张 GPU。rank 0 生成唯一 ID，rank 1 读取该 ID；两个进程并发调用 `ncclCommInitRank`。

## 程序做什么

程序为后续 Mycroft 插桩提供可重复的真实通信任务：

1. 先检查实际加载的 NCCL 版本和路径，确认使用指定的 Day14 产物。
2. 两个进程各绑定一张 GPU，通过共同 ID 初始化通信组。
3. 在 GPU 上运行多种消息大小的多次 Sum AllReduce。
4. 等待本地 stream 完成，把结果拷回 CPU，检查所有元素。
5. 每次打印调用前、API 返回、stream 同步返回和结果校验信息。

`submit_allreduce()` 的核心调用是 `ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream)`；`wait_for_collective()` 使用 `cudaStreamSynchronize(stream)`。NCCL 的 GPU kernel 内部执行实际的通信和归约。程序本身不实现 Ring 算法。

数值校验在 `result_check.h` 中逐个比较输出与期望值。当前输入及其两 rank 的和都是 float 可精确表示的小整数，因此使用精确比较。NaN、无穷大、漏写和上一操作的旧结果会判错；纯 CPU 测试验证这些错误路径。

消息大小固定为 4、4096、1048576 个 float，每个大小至少三次操作。应用 `operation` 从 0 开始跨消息大小连续增加：

| operation | rank 0 每个输入元素 | rank 1 每个输入元素 | 每个期望输出元素 |
|---|---|---|---|
| 0 | 1 | 2 | 3 |
| 1 | 2 | 3 | 5 |
| 2 | 3 | 4 | 7 |
| k | 1+k | 2+k | 3+2k |

这样可以发现读到上一操作结果的问题。默认三次迭代会让每个 rank 完成九次 operation。输入拷贝、输出初始化和 AllReduce 使用同一个 stream 排序。

## 一次集中完成的推理练习

下面是练习情境，不是已采集的运行轨迹。可直接按编号回答，不必逐题等待确认。

1. 将 `wait_for_collective()` 留空，但保留后面的同步 `cudaMemcpy(..., cudaMemcpyDeviceToHost)`。一次碰巧正确的结果能否证明 `stream_sync_return` 标记可信？又是否保证 D2H 拷贝正确等待了当前 nonblocking stream 上的通信？
2. rank 0 最后打印 `operation=2 stage=api_return`，rank 1 最后打印 `operation=2 stage=api_begin`。能否据此判 rank 1 是根因？分别列出至少一种尚不能排除的情况。
3. 同一 operation 有两个 channel：channel 0 的最后片段已处理完，channel 1 仍在 `waitPeer()`。此时局部进度能否发布为该 rank 的整个 operation completion？聚合条件应是什么？
4. 发送方读取 `head` 一直不变，接收方读取 `tail` 一直不变。分别是在等空间还是等数据？找到等待者后，还要查哪一类发布进度的环节？
5. 两个进程使用同一 ID，但一个先执行 count=4、另一个先执行 count=4096。为什么“建组成功”不能保证后续 collective 正确？

第一题尤其用于区分“数值正确”和“事件语义正确”，这是后续 Mycroft 插桩验收的重点。

## 验证与运行

本地先验证 ID 传递，不需要 CUDA/GPU：

```bash
python3 -m unittest discover -s workloads/minimal_allreduce/tests -p 'test_*.py' -v
bash -n workloads/minimal_allreduce/run.sh
```

真实编译和运行需要 Crater 已通过 Day14 探针的 CUDA 12.5.82 镜像。Custom Job 主 Role 1 副本、Worker 0，每个主 Pod 申请 2 GPU；建议 2 CPU、8 GiB 内存。两个 rank 是该 Pod 内的两个进程，使用同一物理节点的两张 GPU。

镜像必须有 g++、Python3、timeout、nvidia-smi、realpath、sha256sum 和 CUDA 开发头文件/runtime 库。CMake 是可选工具。程序没有自定义 CUDA kernel，因此可以由 g++ 编译，GPU kernel 来自指定的 NCCL 共享库。CUDA_HOME 优先使用显式值，否则从 nvcc 推导，最后检查 `/usr/local/cuda`；优先链接共享 cudart，只有静态开发库时链接静态 cudart。

从包含本 workload 的仓库或解压目录执行：

```bash
NCCL_ROOT="<Day14产物目录，包含include和lib>" \
  bash workloads/minimal_allreduce/run.sh 3
```

`NCCL_ROOT` 使用当前 Crater 中实际路径。脚本不会重新编译 NCCL，也不会下载依赖。直接调用 `run.sh` 时日志留在脚本打印的 `task_dir`；Crater 作业推荐使用下方的持久化运行入口。

### Windows 上传与 Crater 启动

在 WSL 仓库根目录生成新上传包：

```bash
python3 cluster/crater/scripts/package_day15.py
```

早期上传包会因缺少 CMake 而退出。修复后的新包需指定新文件名，避免覆盖已有包：

```bash
python3 cluster/crater/scripts/package_day15.py \
  --output .build/day15-upload/day15-allreduce-r2.tar.gz
```

使用 r2 包时，把下方解压命令的文件名改成 `day15-allreduce-r2.tar.gz`，结果目录改成新名字，例如 `day15-results-r2`。

上传 `.build/day15-upload/day15-allreduce.tar.gz` 到持久化目录，例如放在已有 Day14 上传目录。包不包含 NCCL 库、CUDA Toolkit 或主仓库 Git 元数据。需要确认 Day14 结果目录仍包含 `include/nccl.h`、`lib/libnccl.so.2.21.5` 和 lib 的符号链接；仅有 Day14 日志不够。

主 Role 1、Worker 0、2 GPU、2 CPU、8 GiB，工作目录设为上传包所在目录。在启动命令中使用实际 Day14 结果路径替换 `<DAY14_RESULTS>`：

```bash
bash -lc 'set -e
task_tmp=$(mktemp -d /tmp/day15-source.XXXXXX)
tar -xzf day15-allreduce.tar.gz -C "$task_tmp"
NCCL_ROOT="<DAY14_RESULTS>" RESULT_DIR="$PWD/day15-results" \
  bash "$task_tmp/day15-allreduce/cluster/crater/scripts/run_day15.sh"'
```

若沿用的镜像需要 `/crater-start.sh` 包装，保持此前成功的启动方式。运行目录中的 `day15-results` 必须不存在；重试时换成新名字。编译、临时 ID 和中间产物均在 `/tmp`，结果保存到持久化目录。

作业结束下载整个结果目录：

```text
day15-results/
├── environment.txt
├── nccl-build-manifest.txt   # Day14 目录有 manifest 时保存
├── run.log
├── rank0.log                # 编译成功并启动 rank 后产生
├── rank1.log
└── run-status.txt
```

真实功能验证需同时满足：两个不同 GPU 的运行记录、两个 rank 均使用指定 NCCL 2.21.5、每个 rank 的 operation 0—8 都完成并检查通过、两个 rank 和作业退出码均为 0。作业 Completed 或 `nvidia-smi` 成功本身不足以通过。

预期每个 rank 的每次 operation 依次出现 `api_begin`、`api_return`、`stream_sync_return` 和 `result=PASS`，最终打印 `operations=9 status=PASS`。两个 rank 都必须退出 0，加载的库版本为 22105，实际路径与指定 Day14 产物一致。

阶段标记是 CPU 观测时间；应用 operation 序号尚未映射到 NCCL CollOp 身份。`api_return` 不等于 GPU 完成；正确同步后的 `stream_sync_return` 说明此前本地 stream 工作完成，也不是 GPU 的精确结束时间。其耗时还包含输入准备等此前 stream 工作。

### 首轮 INFO 日志的证据边界

Day14 库以 `TRACE=0` 构建，设置 `NCCL_DEBUG=TRACE` 也不能恢复编译时被移除的 TRACE 调用。默认运行脚本保留 INFO 初始化、图和 transport 日志；不能把初始化 channel 数、算法图或环境变量请求当作逐操作实际算法/协议的完整证据。下方独立 TRACE 实验已补足这一观测缺口，其选择属于新库的这次运行，不能追溯成旧 Day14 库首轮运行的选择。

### 补齐算法和协议：独立 TRACE 实验

输入为固定且干净的 NCCL 2.21.5 源码和相同 workload；输出为新库的构建/加载证据、双 rank 日志、逐操作选择和阶段数据。源码不增加 Mycroft 插桩，仅开启上游已有 TRACE 日志。此实验已在 Crater 完成，本地复核通过。新库 SHA256 为 `e4437dd0b48e3ab426b17b394043a0d162f1c9040b32bcf34b5b866e6286132a`；构建及持久化加载检查均退出 0，版本 22105、SONAME 和加载来源正确，双 rank 共十八条数值结果均通过。

rank0 的实际选择如下，每种大小连续三次，九条选择日志均位于对应 API 调用区间：

| 每个 rank 的逻辑消息大小 | 算法 | 协议 | 计划使用 channel 数 |
|---|---|---|---|
| 16 B | RING | LL | 1 |
| 16 KiB | RING | LL | 2 |
| 4 MiB | RING | SIMPLE | 4 |

均为 CBDColl 调度路径，channel 连接记录为 P2P/CUMEM。该表不是不同协议的性能比较。

选中的算法决定数据通信和归约的组织方式，如 RING/TREE；协议决定该路径使用 SIMPLE/LL/LL128 哪种传输和同步实现。对于后续 Mycroft，必须先知道操作走了哪条执行路径，才能选择相应的 GPU 等待点，并解释 head/tail 等进度。不能只因此前读过 Ring 源码就认定所有操作都用 Ring。

观测依据是固定源码 `src/enqueue.cc` 中 `tunedColl enqueue coll`、`CBDColl enqueue coll` 等日志：字段取自真正写入执行计划的 `collInfo->algorithm`、`collInfo->protocol`。上游仅在 `comm->rank == 0` 时打印这条详细日志，因此脚本只证明 rank0 的提交计划，不伪造 rank1 的字段。此程序每次只调用一个 collective，调用后立即同步；选择日志必须位于该操作的 `api_begin` 与 `api_return` 之间，且元素数、字节数正确，才能对应到应用序号。这不是通用的 NCCL CollOp 身份映射，也不是 GPU 完成事件。

从仓库根目录生成包：

```bash
python3 cluster/crater/scripts/package_day15_trace.py
```

在 Windows 文件管理器打开以下 WSL 发行版，再进入本仓库的 `.build/day15-trace-upload`，上传其中的 `day15-trace.tar.gz`；`.sha256` 是可选的传输校验文件：

```text
\\wsl.localhost\Ubuntu-22.04
```

Crater 沿用已成功运行的 CUDA 12.5.82 镜像。工作目录设为本次包所在持久化目录，保留已有用户空间挂载。Master/主 Role 为 1 副本，Worker 为 0；主 Pod 申请 2 GPU、2 CPU、16 GiB 内存，RDMA 无需开启。一次作业内先构建再运行；构建阶段暂不使用 GPU，但可避免分成两个作业传递新库。若需要平台启动包装，沿用此前成功的方式。

启动命令：

```bash
bash -lc 'set -e
task_tmp=$(mktemp -d /tmp/day15-trace-source.XXXXXX)
tar -xzf day15-trace.tar.gz -C "$task_tmp"
CUDA_HOME=/usr/local/cuda JOBS=2 RESULT_DIR="$PWD/day15-trace-results" \
  bash "$task_tmp/day15-trace/cluster/crater/scripts/run_day15_trace.sh"'
```

本实验拒绝非空的 `NCCL_ALGO`、`NCCL_PROTO`，保留自动选择。结果目录必须不存在，重试要换成新名称；原 Day14 库和之前结果不会覆盖。详细 NCCL 编译仍在 `/tmp`，只保存库、头文件及必要证据到持久化目录。若构建失败则停止，不启动 workload，并保存已有构建日志和非零状态。

成功后应有 `experiment_exit_code=0`，`analysis.txt` 中 `functional_baseline=PASS`、`selection_evidence=PASS scope=rank0_submitted_plan`，以及九行选择表。算法或协议可以随消息大小变化，实验不预设必须 RING/SIMPLE。CSV 的 rank1 算法/协议列留空，因为这条观测点未提供其直接证据。TRACE 日志有额外开销，本次时差用于阶段观察，不与旧 INFO 结果进行性能比较。

本次实验的最小必要下载清单如下，保留子目录结构；分析摘要和 CSV 可选，完整编译日志在失败排查时再取。以后新实验遵循仓库规则，将必要证据自动打成单个结果包供下载。

```text
day15-trace-results/
├── experiment-status.txt
├── analysis.txt             # 可选：可在本地重算
├── nccl-trace/
│   ├── run-status.txt
│   └── verification.log
└── allreduce/
    ├── run-status.txt
    ├── environment.txt
    ├── nccl-build-manifest.txt
    ├── run.log
    ├── rank0.log
    ├── rank1.log
    └── operations.csv       # 可选：可在本地重算
```

将下载文件放在 `.build/day15-trace-upload/day15-trace-results/`。本次用户已将构建侧 `run-status.txt` 和 `verification.log` 放在结果根目录，读取时使用该实际位置，无需再移动；实验状态文件带下载后缀 `(1)`，内容已识别。先确认构建侧 `build_exit_code=0`、`saved_verification_exit_code=0`，从 `verification.log` 的 SHA256 行取得新库 hash；`saved-verification.log` 不包含 hash。使用独立构建证据给出的 hash 复查：

```bash
python3 workloads/minimal_allreduce/verify_results.py \
  .build/day15-trace-upload/day15-trace-results/allreduce \
  --expected-sha256 e4437dd0b48e3ab426b17b394043a0d162f1c9040b32bcf34b5b866e6286132a \
  --require-selection
```

该命令已通过。重建后必须使用该次独立构建证据提供的新 hash，不能使用旧 Day14 hash，也不能为通过检查而直接信任待核对的 `environment.txt`。选择检查仍依赖同次构建的成功及来源证据。

### 公开实验证据与离线复查

[`results/samples/e06/day15`](../../results/samples/e06/day15) 保存两轮 Crater 真实运行的脱敏摘录，共约 32 KiB：`baseline/` 来自原 Day14 库的 INFO 功能验证，`trace/` 来自独立 TRACE 构建。根目录 `verification.log`、`build-status.txt` 保存 TRACE 构建验证及持久化加载检查；原始完整日志仍在忽略的 `.build` 目录。

摘录保留真实操作顺序、逻辑大小、阶段时间、结果、算法/协议、channel 数和连接方式；主机名、进程号、GPU UUID、内存指针、communicator hash 和个人路径改成占位符。占位符保留身份的对应关系；本机 PCI 信息用于核对实际 GPU 绑定。网络地址及无关初始化日志未进入公开样例。样例是已发生实验的摘录，不是生成的模拟轨迹；脱敏后复查通过也不代表重新执行了 GPU 实验。`analysis.txt` 保留采集时的输出，其 `PENDING_USER_REVIEW` 是当时的检查状态。

从任何新 checkout 的仓库根目录复查：

```bash
python3 workloads/minimal_allreduce/verify_results.py results/samples/e06/day15/baseline
python3 workloads/minimal_allreduce/verify_results.py results/samples/e06/day15/trace \
  --expected-sha256 e4437dd0b48e3ab426b17b394043a0d162f1c9040b32bcf34b5b866e6286132a \
  --require-selection
```

两条命令均应退出 0。第二条还应输出九行实际选择和 `selection_evidence=PASS`；预期 hash 可与样例根目录 `verification.log` 核对。`scripts/check.sh` 包含这两条离线回归。

## 不依赖大模型处理结果

先把实验要求写成规则，再用程序检查，而不是读完日志后临时决定是否通过。本实验的规则是：库来源正确、同一节点的两张不同 GPU、相同 communicator、两个 rank 和作业退出 0、每次操作的数量/顺序/元素数/期望值正确，并具备本地同步返回记录。

当前下载文件直接放在 `.build/day15-upload/`。从仓库根目录执行：

```bash
python3 workloads/minimal_allreduce/verify_results.py .build/day15-upload
```

脚本默认核对本次 Day14 已验证库的 SHA256。若之后重建 NCCL，需用 `--expected-sha256` 显式给出该次已验证构建的 hash；不能为了通过检查随便接受输入日志中的 hash。默认每种大小三次，其他运行次数通过 `--iterations` 指定。

核对成功返回 0，输出 `functional_baseline=PASS`、两个 rank、每个 rank 九次操作、十八条结果记录以及连接 transport。失败返回非零，指出不满足的条件或缺失的证据。当前真实日志已通过；对临时副本的七项负向检查也通过，覆盖作业失败、缺少同步、错误期望值、库路径/hash 错误、时间倒置和重复结果。

需要逐操作数据时，选择新的 CSV 文件名：

```bash
python3 workloads/minimal_allreduce/verify_results.py .build/day15-upload \
  --csv .build/day15-upload/stage-times-new.csv
```

CSV 每行表示一个 rank 的一次操作，保留元素数、字节数、期望值和本地阶段时差：

- `api_us`：`api_return - api_begin`。
- `sync_wait_us`：`stream_sync_return - api_return`。
- `observed_total_us`：`stream_sync_return - api_begin`。

这些是 CPU 观测的阶段间隔，包含日志输出等开销；同步等待还受此前 stream 工作影响。它们不等于 GPU kernel 时长，不跨 rank 相减或据此直接判根因。每个大小只有三次操作，可用于阅读阶段差异，不能作为稳定性能结论。

本次 INFO 日志中，网络插件加载失败后使用内部插件，libibverbs 缺失后选择 Socket 网络模块；同时 channel 明确连接为 P2P/CUMEM。应结合日志层次和后续成功行为解释这些提示，不能仅凭 `Failed` 字样判任务失败，也不能把 `Using network Socket` 解释成每次 AllReduce 数据经过 Socket。

出现失败时先看最后成功阶段：bootstrap 未返回就检查 ID 发布/读取；初始化未返回就检查 rank 数、设备绑定和连接建立；API 已返回但同步未返回，则需要 GPU/peer/transport 的进度证据。最后一个等待者不自动等于根因。结论应分别列出已证明、未证明，以及下一项能区分假设的观测。
