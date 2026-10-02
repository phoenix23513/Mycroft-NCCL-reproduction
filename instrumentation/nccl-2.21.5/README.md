# NCCL 2.21.5 Instrumentation

Day 13 使用官方 NCCL submodule 做源码映射；新路线从 Day 14 构建该版本，Day 16 起添加并验证可复查的插桩 patch。

固定基线：

```text
repository: https://github.com/NVIDIA/nccl.git
tag:        v2.21.5-1
commit:     ab2b89c4c339bd7f816fbc114a4b05d386b66290
```

克隆主仓库后初始化源码：

```bash
git submodule update --init --recursive third_party/nccl
git -C third_party/nccl describe --tags --exact-match
git -C third_party/nccl rev-parse HEAD
```

预期分别输出 `v2.21.5-1` 和上述完整 commit。NCCL 保留其上游许可证，源码不采用本项目 Apache-2.0 重新许可。

文档入口：

- [`tracepoints.md`](tracepoints.md)：候选插桩点、能观察的状态和 Day 16—20 动态验证清单；
- [`../../docs/architecture/event-field-sources.md`](../../docs/architecture/event-field-sources.md)：论文字段、版本化 Event 与 NCCL 成员之间的映射及限制。

Day 13 不包含 NCCL 修改或编译结果；候选点只有经过 Day 16—20 的真实 NCCL 运行验证后才能升级为已确认插桩语义。

## Day 14：未插桩构建与加载基线

当前状态：**真实构建与加载基线已完成，用户已进入 Day 15**。Crater 已使用 CUDA 12.5.82 从固定源码构建并加载未插桩 NCCL 2.21.5，运行版本、实际加载路径和退出码均符合要求。当前 WSL 是 Ubuntu 22.04.5、G++ 11.4.0；有 Make、Python 和 CMake，但未找到 `nvcc`、CUDA Toolkit 或 Docker，`nvidia-smi` 报告 GPU 访问被操作系统阻止。本次真实验证在 Crater 完成，未在 WSL 编译。GPU 不是本日版本检查的前提；Day 15 才需要 GPU 执行 collective。

用户返回的候选环境探针结果（基础镜像公开名称尚未提供）：

| 候选 | OS / G++ / Python / CUDA | 实际探针结果 | 选择 |
|---|---|---|---|
| A | Ubuntu 24.04.3 / G++ 13.3.0 / Python 3.12.3 / CUDA 13.0.88 | `compute_70` 不受支持，`CUDA_COMPILE_FAILED` | 不用于当前 V100 编译目标 |
| B | Ubuntu 22.04.4 / G++ 11.4.0 / Python 3.11.11 / CUDA 12.5.82 | 最小 CUDA 编译/链接通过；随后真实 NCCL 构建及两次加载检查均通过 | 本次实际构建环境 |

两次探针 Make 均为 4.3，target 均为 `sm_70` + `compute_70` PTX。B 的完整构建 manifest 已取得；基础镜像的公开名称/标签和 GUI 最终状态尚未单独记录。日志中的两步退出码均为 0。版本检查不验证 GPU 运算或 collective 正确性。

### 构建原理与接口

- [`build_nccl.sh`](../../cluster/crater/scripts/build_nccl.sh)：先核对固定 commit、tag 和源码干净状态，再在新目录构建真实 `libnccl.so` 并自动验证。源码不满足约束返回 1；前置环境缺失或参数错误返回 2；编译或加载失败返回非零。
- [`Dockerfile`](../../cluster/crater/images/Dockerfile)：提供 CUDA 12.4.1 devel、Ubuntu 22.04、GCC/G++ 11 和构建工具。镜像只提供环境，运行时挂载项目 checkout，NCCL 始终来自项目 submodule。
- [`nccl_version.cpp`](../../cluster/crater/probes/nccl_version.cpp)：通过 `dlopen` 打开指定库，`dlsym` 查找并调用 `ncclGetVersion`，`dladdr` 确认这个函数实际来自指定文件。程序不初始化 communicator、不创建 CUDA context，也不执行 GPU 运算。

CUDA devel 与 runtime 的区别是前者提供 `nvcc`、头文件和开发库。`nvidia-smi` 的 CUDA 数字表示驱动能力，PyTorch 的 CUDA 数字表示框架构建版本；两者都不能证明当前有 `nvcc`。[NCCL 2.21.5 官方兼容说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/release-notes/rel_2-21-5.html)列出 CUDA 12.4；基础镜像标签见 [NVIDIA CUDA 镜像](https://hub.docker.com/r/nvidia/cuda/tags?name=12.4.1-devel-ubuntu22.04&page=1)。现有 Crater 探针记录的 PyTorch/CUDA 12.6 镜像尚未验证具备开发工具，也尚未用它编译本项目 NCCL。

### 有 CUDA devel 时的本地命令

筛选候选镜像时，先使用独立的 [`cuda_devel_probe.sh`](../../cluster/crater/probes/cuda_devel_probe.sh)。它不依赖仓库或 submodule，可单独复制到 Crater 持久化目录。为每个候选镜像创建一次 Custom Job：1 个主 Role 副本、Worker 0、2 CPU、2 GiB、GPU 0；启动命令为 `bash <实际挂载目录>/cuda_devel_probe.sh`。该资源建议仅用于小探针，不是完整 NCCL 构建。自建镜像不假定有 `/crater-start.sh`，平台已有镜像如需其启动包装，可把相同命令放在包装内执行。

若当前环境能访问项目 checkout，也可直接执行：

```bash
bash cluster/crater/probes/cuda_devel_probe.sh
```

探针输出 OS、编译器、Make、Python、CUDA 版本和缺失项；实际编译 CUDA kernel 并链接静态 CUDA runtime，但不执行生成的程序。临时文件位于 `/tmp`（或已有 `TMPDIR`），退出时只清理探针自己创建的文件。

| 最后一行 | 含义与下一步 |
|---|---|
| `status=BLOCKED`，退出码 2 | 缺少列出的工具、nvcc、头文件或静态开发库；先换用完整开发环境 |
| `status=CUDA_COMPILE_FAILED`，退出码 1 | 工具存在，但最小 CUDA 编译/链接失败；查看此前编译错误，包括 host compiler 或 GPU target 支持问题 |
| `status=READY_FOR_NCCL_BUILD`，退出码 0 | 最小 CUDA 编译/链接成功，可进入固定源码构建；不是 NCCL 构建成功或 GPU 可运行的证据 |

默认 target 与本日构建脚本一致，覆盖 V100。无需在探针中使用 PyTorch；已有 PyTorch 镜像中的 NCCL 不能替代项目产物。用户返回候选镜像的公开版本与探针输出后，再选取构建环境；原始日志不要直接提交。

从仓库根目录执行；先确认 submodule 已初始化，再做只读预检：

```bash
bash cluster/crater/scripts/build_nccl.sh --check
```

预检成功只表示构建前置条件满足，不代表 NCCL 已构建或运行。开始构建：

```bash
bash cluster/crater/scripts/build_nccl.sh
```

可配置项：

| 环境变量 | 默认值与用途 |
|---|---|
| `CUDA_HOME` | 优先使用显式值，否则从 PATH 中的 `nvcc` 推导，最后检查 `/usr/local/cuda` |
| `CXX` | `g++`；镜像中为 `g++-11`，必须是单个可执行文件名或路径 |
| `JOBS` | `2`，控制并行编译数量；内存不足时可设 `1` |
| `BUILDDIR` | 仓库根目录下 `.build/nccl`；必须不存在且位于 NCCL 源码树外；相对路径相对于调用者工作目录 |
| `NVCC_GENCODE` | `-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70`，包含 V100 原生代码和 PTX；其他 GPU 的原生代码可通过此变量显式指定 |
| `NCCL_TRACE` | `0`；只接受 `0` 或 `1`，Day15 的单独观测构建使用 `1`，必须使用新输出目录 |

例如新建另一份输出，避免复用不同参数的旧对象：

```bash
JOBS=1 BUILDDIR="$PWD/.build/nccl-run2" \
  bash cluster/crater/scripts/build_nccl.sh
```

脚本固定 `DEBUG=0`、`NVTX=1`、`PROFAPI=1`、`RDMA_CORE=0`、`CUDARTLIB=cudart_static`，`TRACE` 默认 `0`，可用 `NCCL_TRACE=1` 在新目录构建单独的观测库。构建完整 collective 集合，仅生成所需共享库。它清除环境中额外的 make/compiler flags，并保留完整编译命令日志及 `trace` 值。路径不能含空白，这是上游 Makefile 的限制。脚本不会自动安装工具、修改 NCCL 源码或删除旧构建目录。已经验收的 Day14 产物仍是 `TRACE=0`；不能把新库的 hash 或运行选择当成旧库的运行证据。

### 使用镜像构建

以下命令需在有 Docker 的机器执行；当前 WSL 尚未执行过镜像构建。构建上下文仅包含镜像目录，无需把整个仓库发送给 Docker：

```bash
docker build -t mycroft-nccl-devel:cuda12.4.1 cluster/crater/images
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace/mycroft" -w /workspace/mycroft \
  mycroft-nccl-devel:cuda12.4.1 \
  bash cluster/crater/scripts/build_nccl.sh
```

不需要 `--gpus`。挂载把宿主机 checkout 暴露给容器，输出仍保留在宿主机 `.build/nccl`；使用当前 UID/GID 可避免产生 root 所有的文件和 Git 所有者检查问题。脚本、工具链版本和参数使构建过程可复查，但镜像标签与 apt 包版本不构成二进制逐字节一致的承诺；实际工具版本以 manifest 为准。

### Crater GUI 执行

优先在已有平台镜像中运行 `nvcc --version` 和脚本 `--check`。若已有镜像缺少开发工具，再使用上述 CUDA devel 环境；镜像构建和进入内部镜像仓库需在具备该能力的环境完成，当前没有已发布或已验证的自建镜像。

使用 **Custom Job**，无需 DDP 模板：

| GUI 字段 | 本日配置 |
|---|---|
| 镜像 | 包含 CUDA devel 与脚本依赖的镜像 |
| Master / 主 Role 副本数 | `1` |
| Worker 副本数 | `0`；页面没有 Worker 时无需添加 |
| CPU / Memory | 建议 2 CPU、16 GiB；本次日志确认编译并行度为 2，未单独记录实际资源配额 |
| GPU | `0` |
| 挂载 | 项目 checkout 所在持久化目录，包含主仓库 `.git` 和 submodule 元数据 |
| 工作目录 | 当前环境确认的仓库根目录 |
| 启动命令 | `bash cluster/crater/scripts/build_nccl.sh` |

这会启动一个 Pod 中的一个构建脚本进程；make 可按 `JOBS` 启动编译子进程。Worker 不参与本日构建。自建 Dockerfile 没有平台镜像中的 `/crater-start.sh`，直接运行脚本；若平台要求额外启动包装，应先确认平台支持的镜像与入口。不要在公开文件写入内部镜像地址或个人挂载路径。

共享目录只保存最终产物和必要日志；若平台提供足够的 Pod 本地临时存储，优先用它作为 `BUILDDIR`，完成后把 `lib/`、版本检查程序与证据文件保存到用户确认的持久化目录。输出不会自动清理，避免在共享目录反复生成大量对象文件。

### Windows 上传离线构建包

本次构建前 Crater 只有单独上传的探针，没有完整 checkout，因此采用离线包。使用 [`package_day14.py`](../../cluster/crater/scripts/package_day14.py) 在本地生成：

```bash
python3 cluster/crater/scripts/package_day14.py
```

输出 `.build/day14-upload/day14-nccl-build.tar.gz` 与同目录 SHA256 文件；不提交这些产物。包包含本日构建/加载程序、上游固定源码及最小 Git 数据，通过本地浅克隆生成；不包含主仓库 Git 元数据、凭据、reflog、用户笔记或旧构建产物。Crater 构建时不需要访问 GitHub，也不需要 Docker。包内源码仍会被构建脚本核对 commit、tag 和干净状态。

从 Windows 上传压缩包（可同时上传 `.sha256` 文件）到持久化目录。使用已通过预检的候选 B 镜像，Custom Job 主 Role 1 副本、Worker 0、2 CPU、16 GiB、GPU 0，工作目录设为实际上传目录。在启动命令中粘贴：

```bash
bash -lc 'set -e; task_tmp=$(mktemp -d /tmp/day14-nccl.XXXXXX); tar -xzf day14-nccl-build.tar.gz -C "$task_tmp"; RESULT_DIR="$PWD/day14-results" bash "$task_tmp/day14-nccl-build/cluster/crater/scripts/run_day14_build.sh"'
```

平台镜像如需 `/crater-start.sh`，沿用此前探针成功的启动包装，将上述命令作为它执行的命令。这里 `task_tmp` 是本次 Pod 的临时目录，源码与对象文件都在其中；`RESULT_DIR` 指向上传目录下的新持久化结果目录。`/tmp` 需有足够存储，实际资源需求尚待本次完整构建确认。运行结束后不需要手动递归删除共享目录文件。

[`run_day14_build.sh`](../../cluster/crater/scripts/run_day14_build.sh) 会保存 `include/`、`lib/`、版本检查程序、manifest、构建/验证日志和 `run-status.txt`；构建失败也会保存已产生的日志并返回原退出码。成功后再次加载持久化目录中的库，结果写入 `saved-verification.log`。已有结果目录会被拒绝；重试时显式换一个新的 `RESULT_DIR`，如 `day14-results-run2`。

预期结束日志包含 `nccl_version=2.21.5`、`status=PASS` 与 `Persistent library loading verified`。本次用户返回的文件已证明构建及持久化库加载检查成功，结果见下方真实运行证据；重现失败时应保留首次编译错误及其上下文。

### 验收与证据

成功构建后默认产生：

```text
.build/nccl/
├── include/
├── lib/libnccl.so -> libnccl.so.2 -> libnccl.so.2.21.5
├── obj/
├── nccl_version
├── build-manifest.txt
├── build.log
└── verification.log
```

复查加载行为：

```bash
./.build/nccl/nccl_version ./.build/nccl/lib/libnccl.so.2
readelf -d ./.build/nccl/lib/libnccl.so.2.21.5
sha256sum ./.build/nccl/lib/libnccl.so.2.21.5
```

版本输出格式如下，`<BUILD_DIR>` 表示实际输出目录；本次真实运行样例见下一节：

```text
requested_library=<BUILD_DIR>/lib/libnccl.so.2.21.5
loaded_library=<BUILD_DIR>/lib/libnccl.so.2.21.5
nccl_version_code=22105
nccl_version=2.21.5
status=PASS
```

两条路径必须相同，并与本次项目产物的 canonical path 对应。检查程序使用 `dlopen`，所以 `ldd nccl_version` 不会列出 NCCL；这里采用 `dladdr` 提供实际已加载库的等价证据。仅看文件名、header 版本或 ELF SONAME 不能通过验收。manifest 记录源码版本、干净状态、OS、内核、CUDA、编译器、Make、Python 和实际 make 参数；`verification.log` 记录 ELF metadata、库 SHA256、运行版本和实际路径。

真实验收必须同时具备：固定且未修改的源码、脚本完成构建、运行版本正确、加载路径正确，以及完整工具链记录。二进制、对象、manifest 和原始日志仅作本地证据，不提交；它们可能包含当前机器的绝对路径，公开样例需先脱敏。

### 本次 Crater 真实运行证据

以下四份小型样例来自用户下载的实际运行文件；只将个人挂载路径替换为 `<USER_MOUNT>`、随机 Pod 临时目录替换为 `<POD_TMP>`，并去掉行末空白。版本、参数、退出码和 hash 均保留原值，不含二进制或完整构建日志：

| 验收项 | 实际结果与证据 |
|---|---|
| 固定、未插桩源码 | tag `v2.21.5-1`、commit `ab2b89c4c339bd7f816fbc114a4b05d386b66290`、`source_clean=yes`，见 [build-manifest.txt](../../results/samples/e06/day14/build-manifest.txt) |
| 可复现构建 | manifest 记录实际 make 命令；并行度 2，`DEBUG=0`、`TRACE=0`、`NVTX=1`、`PROFAPI=1`、`CUDARTLIB=cudart_static`，target 为 `sm_70` + PTX |
| 实际运行版本 | 临时与持久化产物均返回 `nccl_version_code=22105`、`nccl_version=2.21.5`、`status=PASS` |
| 实际加载项目库 | [verification.log](../../results/samples/e06/day14/verification.log) 与 [saved-verification.log](../../results/samples/e06/day14/saved-verification.log) 中 requested/loaded canonical path 各自相同，均指向本次构建的 `libnccl.so.2.21.5`；SONAME 为 `libnccl.so.2` |
| 工具链记录 | Ubuntu 22.04.4、kernel `5.15.0-187-generic`、G++ 11.4.0、CUDA 12.5.82、Make 4.3、Python 3.11.11，见 manifest |
| 两步退出码 | [run-status.txt](../../results/samples/e06/day14/run-status.txt) 中 `build_exit_code=0`、`saved_verification_exit_code=0` |

实际日志记录的库 SHA256：

```text
f3fe9df1bb787e0d8f82d60a185c69299ef3806010dcee35565f6b8f6bad7dc4
```

本次构建与加载证据已核对，用户已明确进入 Day 15。此结果不包含 GPU collective 验证，Dockerfile 也尚未实际构建；Day 15 使用同一份项目库执行真实双 rank AllReduce，当前进度见 [`workload README`](../../workloads/minimal_allreduce/README.md)。

本地自动回归命令（不要求 CUDA/GPU，不生成伪 NCCL 库）：

```bash
python3 -m unittest discover -s tests/nccl -p 'test_*.py' -v
./scripts/check.sh
git diff --check
```

本地已验证范围：检查程序可用 G++ 11.4 严格警告编译；17 个 guard 测试通过，覆盖参数错误、缺失或非 ELF 库、系统库缺少 NCCL 接口、源码有已跟踪/未跟踪改动、固定 tag 缺失，以及无 CUDA 时准确阻塞且不创建构建目录；包含候选镜像探针检查、离线包解压后的固定 commit/tag/干净状态与 Git 对象完整性、包 checksum、元数据排除、旧归档/结果保留，以及构建失败退出码保存。源码测试只修改临时本地 clone。原有 Python、Event、E01/E02 回归也通过。本地单元回归与上述 Crater 真实构建/加载证据分别保留。
