# E02——通信进度状态机

Day 08 只建立正常的三阶段流水线，不注入故障，也不调用真实 GPU、RDMA 或 NCCL。

## 三个逻辑执行者

一次发送被抽象成三个累计进度量：

1. GPU producer 准备数据，推进 gpu_ready；
2. CPU Proxy 只能发送已经由 GPU 准备好的 chunk，推进 rdma_transmitted；
3. Network completer 只能完成已经提交发送的 chunk，推进 rdma_done。

任何时刻必须满足：

~~~
0 <= rdma_done <= rdma_transmitted <= gpu_ready <= total_chunks
~~~

这里的 Proxy 是 NCCL 所在进程内的 CPU 线程，不是另一台机器，也不是 GPU 线程。Day 08 暂时只模拟职责和先后关系；QP、请求提交和 CQE 的具体机制在需要时再补充。

## Tick 模型

每个 tick 中，三个 actor 都读取同一份 previous 快照，再分别计算下一时刻的累计值。因此一个新 chunk 不能在同一个 tick 内连续穿过三个阶段。

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

## 当前实现任务

只需理解并完成 progress_sim.py 中三个 advance 方法：

- GPUProducer.advance：未准备完时最多增加一个；
- ProxyTransmitter.advance：只有 previous.rdma_transmitted 小于 previous.gpu_ready 时才能增加一个；
- NetworkCompleter.advance：只有 previous.rdma_done 小于 previous.rdma_transmitted 时才能增加一个。

调度器、快照记录、不变量验证、CLI 和测试已经搭好。三个方法都必须返回累计数量，不能返回本 tick 的增量。

## 观察预期失败

~~~bash
python3 -m unittest discover \
  -s experiments/e02_progress_state_machine/tests \
  -p 'test_*.py' -v
~~~

骨架阶段测试应因 TODO(OWNER) 抛出 NotImplementedError。三个状态转换完成后，测试应全部通过。

运行 CLI：

~~~bash
python3 experiments/e02_progress_state_machine/cli.py --chunks 4
~~~

预期输出七行 JSONL，从全零快照开始，到三个累计量均为 4 结束。
