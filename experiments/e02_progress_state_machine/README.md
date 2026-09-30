# E02——通信进度状态机

Day 08 建立确定性的三阶段抽象流水线。Day 09 在同一模型上加入可配置的 GPU、Proxy 和 Network 合成停滞；整个 E02 不调用真实 GPU、RDMA 或 NCCL。

## 模型定位和边界

E02 是供 Event、Trigger 和 RCA 测试使用的**发送侧归一化分析 fixture**，不是 NCCL 执行模拟器。字段名沿用 Mycroft 论文中的 `GPU_ready`、`RDMA_transmitted` 和 `RDMA_done`，但不代表 NCCL 2.21.5 中存在三个同名变量。

模型只保留三项分析语义：累计进度单调不减、上游进度约束下游进度、快照只描述积压位置而不直接证明故障组件。它有意不模拟协议 step/slice、一次调用推进多个 step、接收侧反压、LL/LL128、多个 channel/QP 或远端 GPU 消费。

Day 13 的源码映射进一步确认：发送侧 `sub->transmitted` 和 `sub->done` 是后两个累计量的候选来源；`gpu_ready` 需要从 GPU 写 FIFO/tail 后的 readiness 条件派生；`total_chunks` 也必须由 `nsteps/sliceSteps` 等真实单位归一化。以上候选关系要由 Day 17 的真实 NCCL state log 动态验证，E02 自身不能证明它们成立。

## 三个逻辑执行者

一次发送被抽象成三个累计进度量：

1. GPU producer 准备数据，推进 gpu_ready；
2. CPU Proxy 只能发送已经由 GPU 准备好的 chunk，推进 rdma_transmitted；
3. Network completer 只能完成已经提交发送的 chunk，推进 rdma_done。

任何时刻必须满足：

~~~
0 <= rdma_done <= rdma_transmitted <= gpu_ready <= total_chunks
~~~

这里的 Proxy actor 只代表受 NCCL CPU Proxy 启发的逻辑阶段，不是对真实 `sendProxyProgress()` 的逐字段复刻，也不是另一台机器或 GPU 线程。

## Tick 模型

`tick` 是合成轨迹的离散逻辑步，不对应物理时间，也不对应一次真实 Proxy 回调。每个 tick 中，三个 actor 都读取同一份 `previous` 快照，再分别计算下一时刻的累计值；“每拍最多推进一个 chunk”只用于产生确定且易验证的测试轨迹。

四个 chunk 的正常预期轨迹是：

~~~
tick  gpu_ready  rdma_transmitted  rdma_done
0     0          0                 0
1     1          0                 0
2     2          1                 0
3     3          2                 1
4     4          3                 2
5     4          4                 3
6     4          4                 4
~~~

## Day 08 正常路径

`run_normal_progress()` 保留无故障入口。三个 actor 每个 tick 最多推进一个 chunk，并始终读取同一份 `previous` 快照。

运行正常演示：

~~~bash
python3 experiments/e02_progress_state_machine/cli.py --chunks 4
~~~

预期输出七行 JSONL，从全零快照开始，到三个累计量均为 4 结束。

## Day 09 合成停滞框架

`FaultSpec` 用三个字段描述一次临时停滞：

- `target`：`gpu_producer`、`proxy_transmitter` 或 `network_completer`；
- `start_tick`：合成停滞开始影响的 `current.tick`；
- `duration_ticks`：连续停滞的 tick 数。

停滞窗口采用半开区间：

~~~text
start_tick <= tick < start_tick + duration_ticks
~~~

核心实现必须接受任意合法参数，不能针对固定轨迹写特殊分支。`4/3/2` 只作为可复查示例，不是唯一测试输入。

三个 actor 的合成停滞分支已经实现：窗口内保持各自的 `previous` 累计值，窗口结束后恢复正常推进。`FaultSpec.target` 只是生成已知测试轨迹的开关，不表示已经向真实 GPU、NCCL Proxy 或网络注入故障。参数化测试覆盖两组规模和窗口配置下的全部三个 target。

唯一人工验收命令为：

~~~bash
python3 experiments/e02_progress_state_machine/cli.py \
  --demo-faults \
  --chunks 6 \
  --fault-start-tick 2 \
  --fault-duration-ticks 3
~~~

该命令使用同一组可配置参数，以人工可读的分组表格依次展示三种 fault target。相同输出保存在 `results/samples/e02/faults.txt`。自动测试和全量检查由开发过程执行，不作为多项人工验收步骤。

## Day 10 状态分类框架

Day 10 不再制造新的故障，而是按照论文中的三个累计量关系解释单个进度快照。`classify_snapshot(snapshot, total_chunks)` 输出的是“数据目前处于什么状态”，不是“哪个 actor 已经发生故障”。

| 状态 | 快照条件 | 本地可能原因 | 远端可能原因 |
|---|---|---|---|
| `not_started` | `gpu_ready == rdma_transmitted == rdma_done == 0` | 未初始化 | 被依赖阻塞 |
| `not_transmitted` | `gpu_ready > rdma_transmitted` | RDMA 发送问题 | 接收端未就绪 |
| `not_delivered` | `rdma_transmitted > rdma_done` | RDMA 完成问题 | 接收端故障 |
| `gpu_not_ready` | `0 < gpu_ready == rdma_transmitted == rdma_done < total_chunks` | GPU 或软件层未继续准备 | 无直接远端结论 |

多个条件可以同时成立。例如 `(6, 4, 1)` 同时具有 `not_transmitted` 和 `not_delivered` 状态，但这不能证明 Proxy 和 Network 同时故障。正常流水线中的 `(3, 2, 1)` 也具有这两个状态，因为分阶段流水线本来就会存在尚未发送和尚未完成的数据。

Day 09 的 `FaultSpec.target` 是模拟器生成轨迹时使用的已知实验真值，不是论文状态分类器的输入。判断某个 actor 是否在相邻 tick 中停止推进属于测试模型或后续时间分析；不能改写上述四种快照状态的含义。

分类结果只列本地可能原因、远端可能原因和仍缺失的证据。E02 没有消息方向和对端事件身份，不能用另一个普通发送快照冒充接收端证据，也不能输出唯一根因；事件分组能力由 E03 提供；Day 21—23 的 E04 只接受真实 NCCL 轨迹作为实验验收输入。

`classify_snapshot()` 已实现上述四类状态、输入验证和证据边界。定向运行：

~~~bash
python3 -m unittest discover \
  -s experiments/e02_progress_state_machine/tests \
  -p 'test_state_classification.py' -v
~~~

预期论文状态表、重叠条件、完成态、正常流水线状态和非法输入测试全部通过；Day 08/09 的既有测试也应继续通过。
