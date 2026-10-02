# Mycroft NCCL Reproduction

一个以真实 NCCL 2.21.5 运行与插桩为后续实验主线的 Mycroft 复现项目。

本项目关注一个实际问题：当分布式训练中的某个 rank、GPU、Proxy 或网络环节变慢时，许多其他 rank 也会因为依赖关系表现为停滞。我们希望通过记录底层通信进度和恢复事件因果关系，区分最早的异常来源与随后被阻塞的受影响 rank。

项目已用 E01/E02 建立确定性回归夹具，并用 E03 建立版本化事件契约。当前路线只执行核心方法复现必需步骤：确认真实 NET/RDMA 路径，采集可信 completion/state log，注入可重复软件延迟，再用真实轨迹验证 Trigger 与 RCA。分析先离线执行，共享内存和独立 reader 移出当前范围。

## 当前进度

- **E01 已完成**：4-rank、8-chunk Ring ReduceScatter/AllGather、JSONL 事件、单点延迟注入和因果传播。
- **E02 已完成**：发送侧归一化分析 fixture、三类合成停滞和论文四类快照状态分类均已通过验收；它不作为真实 NCCL 执行或插桩正确性的证据。
- **E03 已完成并完成契约纠错**：Event v1 保持可读，Event v2 补充真实单调时间与 operation completion；NCCL 2.21.5 字段仍待目标路径动态验证。
- **真实基线已完成**：项目 NCCL 的构建/加载及原生双 GPU AllReduce 已验证；当前实际路径是同机 P2P/CUMEM。执行入口与证据见 [`instrumentation/nccl-2.21.5/README.md`](instrumentation/nccl-2.21.5/README.md)。
- Crater 上已经完成 CPU/Gloo 双进程 AllReduce 与单 GPU PyTorch/CUDA 环境验证。

详细任务与里程碑进度见 [`docs/plans/Mycroft_26日开发路线图.md`](docs/plans/Mycroft_26日开发路线图.md)。

## 已完成的历史回归演示

E01 是当前已完成、可直接运行的历史演示。它在 CPU 上模拟 Ring AllReduce，不需要 GPU、NCCL 或集群；它只用于理解和回归，不代表新路线的真实 NCCL 验收。

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
| E04 | 如何分析真实 NCCL 轨迹中的停滞和根因候选 | Trigger、MinOp/MinData 和 RCA |
| E05（后续扩展） | 如何持续低开销传递真实 NCCL 运行时事件 | 共享内存循环缓冲区与 reader；本轮不实现 |
| E06 | 真实 NCCL 能否产生可分析事件并复现确定性异常 | NCCL 2.21.5 构建、插桩与双节点验证 |

阶段编号表示能力边界。剩余工作是 M1 目标 NET/RDMA 路径 → M2 最小真实日志 → M3 软件延迟对照 → M4 Trigger/RCA → M5 双节点 RDMA 端到端验收。E01/E02 只保留为单元回归。Graph、多协议、完整 P2P GPU 采集与 E05 不作为本轮门槛；最终只标记核心方法复现已验证，不声称完整实时系统已完成。

## 仓库中有什么

- [`experiments/`](experiments/)：E01、E02 确定性回归夹具；
- [`docs/plans/`](docs/plans/)：五个必做里程碑、历史任务及变更记录；
- [`docs/legacy-nccl-source-study-notes/`](docs/legacy-nccl-source-study-notes/)：保留的历史源码学习资料，先核对版本，不作为当前执行计划；
- [`cluster/crater/`](cluster/crater/)：Crater 探针、作业配置和平台实验资料；
- [`src/mycroft/`](src/mycroft/)：已实现的事件契约与轨迹恢复代码，分析器待实现；
- [`instrumentation/`](instrumentation/)：固定 NCCL 构建、候选插桩点和未实现的采集接口；
- [`results/samples/`](results/samples/)：可公开、可复查的小型结果样例。

## 复现范围

这是对 Mycroft 核心思路的独立、小规模复现，不是原系统的源码移植，也不是生产级监控平台。项目不会复刻 Kafka、云数据库、Web 后端或论文中的完整大规模硬件实验；实验结果只用于验证实现和分析思路。

## English summary

This repository independently reproduces the core method of Mycroft-style NCCL stall diagnosis. Five remaining milestones cover the actual NET/RDMA path, validated completion and state traces, controlled software delay, Trigger/RCA, and end-to-end validation. Analysis initially runs offline; shared-memory transport and an independent reader are outside the current scope.

E01 provides a deterministic Ring AllReduce simulator and E02 supplies normalized send-side fixtures for analysis tests. E03 keeps Event v1 readable while Event v2 adds process-local monotonic time and operation-completion records. Experimental acceptance requires actual execution of the project-built NCCL 2.21.5; E01/E02 and handwritten events cannot replace that evidence. See the [core reproduction plan](docs/plans/Mycroft_26日开发路线图.md) for scope and completion criteria.

## License

Unless stated otherwise for third-party material, this project is licensed under the [Apache License 2.0](LICENSE). Third-party components retain their original licenses and notices.
