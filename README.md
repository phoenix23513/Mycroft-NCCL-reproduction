# Mycroft NCCL Reproduction

一个从可控通信模拟逐步走向真实 NCCL 插桩的 Mycroft 复现项目。

本项目关注一个实际问题：当分布式训练中的某个 rank、GPU、Proxy 或网络环节变慢时，许多其他 rank 也会因为依赖关系表现为停滞。我们希望通过记录底层通信进度和恢复事件因果关系，区分最早的异常来源与随后被阻塞的受影响 rank。

实现不会直接跳到修改 NCCL，而是依次建立 Ring 通信模型、进度状态机、统一事件、分析器和共享内存通道，最后再接入 NCCL 2.21.5，并在 Crater 集群上验证。

## 当前进度

- **E01 已完成**：4-rank、8-chunk Ring ReduceScatter/AllGather、JSONL 事件、单点延迟注入和因果传播。
- **E02 已完成**：发送侧归一化分析 fixture、三类合成停滞和论文四类快照状态分类均已通过验收；它不作为真实 NCCL 执行或插桩正确性的证据。
- **E03 已完成并完成契约纠错**：Event v1 保持可读，Event v2 补充真实单调时间与 operation completion；乱序恢复、NCCL 2.21.5 固定版本和候选插桩点均已验证，真实映射的动态确认保留到 E06。
- Crater 上已经完成 CPU/Gloo 双进程 AllReduce 与单 GPU PyTorch/CUDA 环境验证。

详细任务与每日进度见 [`docs/plans/Mycroft_26日开发路线图.md`](docs/plans/Mycroft_26日开发路线图.md)。

## 先看一个可运行的结果

项目当前最完整的实验是 E01。它在 CPU 上模拟 Ring AllReduce，不需要 GPU、NCCL 或集群。

环境要求：

- Python 3.10+；
- 支持 C++17 的编译器；
- CMake 3.16+。

构建并运行测试：

```bash
cmake -S . -B .build
cmake --build .build
ctest --test-dir .build --output-on-failure
```

运行 Ring 模拟器：

```bash
./.build/experiments/e01_ring_dependency/e01_ring_sim
```

观察正常执行产生的 96 条事件：

```bash
./.build/experiments/e01_ring_dependency/e01_ring_sim --jsonl
```

在 ReduceScatter 的一个 send 事件上注入 100 个逻辑时间单位的延迟：

```bash
./.build/experiments/e01_ring_dependency/e01_ring_sim \
  --jsonl \
  --inject-delay reduce_scatter 0 1 0 send 100
```

延迟只在根事件上主动增加一次，之后沿消息依赖和同一 rank 的执行顺序传播；通信数值和 contributor mask 不会改变。示例输出位于 [`results/samples/e01/`](results/samples/e01/)。

还可以直接用浏览器打开 [`experiments/e01_ring_dependency/visualizer.html`](experiments/e01_ring_dependency/visualizer.html)，逐步观察 chunk 在 ReduceScatter 和 AllGather 中的移动与聚合。

## 复现路线

| 阶段 | 要解决的问题 | 主要产物 |
|---|---|---|
| E01 | Ring 通信中的数据流和因果依赖如何形成 | Ring 模拟器、事件轨迹、延迟传播 |
| E02 | 如何用统一累计量构造确定性的分析输入 | 发送侧归一化进度 fixture |
| E03 | 如何跨 communicator、operation 和 channel 唯一识别事件 | 兼容 Event v1 的 Event v2 与 NCCL 源码字段映射 |
| E04 | 如何从轨迹中发现停滞并定位根因 | Trigger、MinOp/MinData 和 RCA |
| E05 | 如何低开销地把运行时事件交给独立分析进程 | 共享内存循环缓冲区与 reader |
| E06 | 模型能否接入真实 NCCL 并复现确定性异常 | NCCL 2.21.5 插桩与双节点验证 |

前四个阶段先在确定性的离线模型上验证语义，E05/E06 再进入运行时和真实集群。这样可以把算法错误、插桩错误和环境问题分开定位。

## 仓库中有什么

- [`experiments/`](experiments/)：E01、E02 等独立可运行实验；
- [`docs/plans/`](docs/plans/)：当前 26 日开发路线及其变更记录；
- [`notes/nccl/`](notes/nccl/)：与实现相关的 NCCL 源码学习笔记；
- [`cluster/crater/`](cluster/crater/)：Crater 探针、作业配置和平台实验资料；
- [`src/mycroft/`](src/mycroft/)：后续统一事件和分析代码；
- [`runtime/`](runtime/)：后续共享内存事件通道；
- [`instrumentation/`](instrumentation/)：后续 NCCL 2.21.5 插桩记录；
- [`results/samples/`](results/samples/)：可公开、可复查的小型结果样例。

## 复现范围

这是对 Mycroft 核心思路的独立、小规模复现，不是原系统的源码移植，也不是生产级监控平台。项目不会复刻 Kafka、云数据库、Web 后端或论文中的完整大规模硬件实验；实验结果只用于验证实现和分析思路。

## English summary

This repository independently reproduces the core path of Mycroft-style NCCL stall diagnosis: communication modeling, progress-state tracing, causal analysis, a shared-memory event channel, and finally NCCL 2.21.5 instrumentation.

E01 provides a deterministic Ring AllReduce simulator, JSONL traces, and causal delay propagation. E02 provides a deterministic normalized send-side fixture for analysis tests, not an execution model of NCCL. E03 now keeps Event v1 readable while Event v2 adds process-local monotonic time and operation-completion records required by later analysis. Deterministic timeline recovery and source-grounded NCCL 2.21.5 candidate mappings remain intact; runtime confirmation is an E06 task. See the [26-day development roadmap](docs/plans/Mycroft_26日开发路线图.md) for the current plan.

## License

Unless stated otherwise for third-party material, this project is licensed under the [Apache License 2.0](LICENSE). Third-party components retain their original licenses and notices.
