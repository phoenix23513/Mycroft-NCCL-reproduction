# E01 — Ring 依赖模拟器

Day 05 实现 ReduceScatter；Day 06 在同一个 4-rank、8-chunk、单 channel 模型上补全 AllGather 和 JSONL 事件。这里不调用 GPU、NCCL、线程或网络。

交互式过程可直接用浏览器打开 `visualizer.html`。它用移动的 chunk 方块完整展示 ReduceScatter 与 AllGather，仍是固定模拟的离线教学视图，不代表真实 NCCL/GPU 时间线。

## 固定输入与结果

```text
rank 0: [ 1,  2,  3,  4,  5,  6,  7,  8]
rank 1: [11, 12, 13, 14, 15, 16, 17, 18]
rank 2: [21, 22, 23, 24, 25, 26, 27, 28]
rank 3: [31, 32, 33, 34, 35, 36, 37, 38]
```

完成三个 ReduceScatter step 后：

```text
rank 0 owns chunk 0 = 64, chunk 4 = 80
rank 1 owns chunk 1 = 68, chunk 5 = 84
rank 2 owns chunk 2 = 72, chunk 6 = 88
rank 3 owns chunk 3 = 76, chunk 7 = 92
```

8 个 chunk 分成两条 lane：lane 0 包含 chunk 0—3，lane 1 包含 chunk 4—7。每个 rank 最终负责每条 lane 中与自己编号对应的一个 chunk。每个最终 chunk 的 `contributor_mask` 必须为 `0x0f`，表示四个 rank 的贡献均被包含。

## 调度规则

Ring 方向固定为：

```text
rank 0 -> rank 1 -> rank 2 -> rank 3 -> rank 0
```

对 `rank`、`step` 和 `lane`：

```text
next_rank   = (rank + 1) mod RING_RANKS
prev_rank   = (rank - 1 + RING_RANKS) mod RING_RANKS
chunk_base  = lane * RING_RANKS
send_chunk  = chunk_base + (rank - step - 1 + RING_RANKS) mod RING_RANKS
recv_chunk  = chunk_base + (rank - step - 2 + RING_RANKS) mod RING_RANKS
```

一个 step 中每个 rank 为每条 lane 各发送一条消息，因此一共生成 8 条消息。必须先根据旧状态生成全部消息，再统一应用消息。不能让前一个 rank 在本 step 的更新被后一个 rank 立即读取。

```text
旧状态
  -> 为所有 rank 和 lane 生成 message[rank][lane]
  -> 为所有目标 rank 和 lane 应用 message[rank][lane]
  -> completed_steps 增加 1
```

## 用户实现范围

在 `src/ring_sim.c` 中依次完成：

1. `ring_next_rank` 和 `ring_prev_rank`；
2. `ring_send_chunk` 和 `ring_recv_chunk`；
3. `ring_sim_init`；
4. `ring_reduce_scatter_step`；
5. `ring_run_reduce_scatter`。

每处理一条接收消息，将规约后的状态保存到 `history[step][dst_rank][lane]`。Day 05 不输出 JSONL；序列化留到 Day 06。

当前模型要求 `RING_CHUNKS` 能被 `RING_RANKS` 整除。非整除分配不在 Day 05 范围内。

## 构建与测试

```bash
cmake -S . -B .build
cmake --build .build --target e01_ring_sim test_e01_reduce_scatter
ctest --test-dir .build -R e01_reduce_scatter --output-on-failure
```

骨架刚建立时，测试应因 `RING_ERROR_NOT_IMPLEMENTED` 失败。实现完成后的程序位于：

```bash
./.build/experiments/e01_ring_dependency/e01_ring_sim
```

AddressSanitizer 验证：

```bash
cmake -S . -B .build-asan -DMYCROFT_ENABLE_ASAN=ON
cmake --build .build-asan --target test_e01_reduce_scatter
ctest --test-dir .build-asan -R e01_reduce_scatter --output-on-failure
```

## Day 05 验证结果

- 4 ranks、8 chunks、2 lanes 的三个 ReduceScatter step 全部通过；
- 每个 rank 的两个 owner chunk 得到完整规约值和 `0x0f` 来源掩码；
- 重复或乱序 step 被拒绝；
- 普通测试、全仓库回归和 AddressSanitizer 均通过；
- `visualizer.html` 用于逐步检查消息、部分结果和最终 owner 状态。

## Day 06 实现与验证

在 `src/ring_sim.c` 中完成以下四个 TODO：

1. `ring_all_gather_send_chunk`；
2. `ring_all_gather_recv_chunk`；
3. `ring_all_gather_step`；
4. `ring_run_all_gather`。

AllGather 的每个 step 仍先快照全部发送，再应用全部接收；接收方复制完整的 `ChunkState`，不能再做加法。`record_event` 中的 ReduceScatter 调用是事件记录示例，AllGather 应按相同顺序记录：先记录全部 `send`，再记录全部 `recv`。

每条 JSONL 事件包含：

```text
op_seq, rank, channel, phase, step, chunk, action, peer,
timestamp, value, contributor_mask
```

`timestamp` 是从 0 开始递增的逻辑时间，不是系统时钟。

构建后先观察预期失败：

```bash
cmake -S . -B .build
cmake --build .build
ctest --test-dir .build -R 'e01_all_gather|e01_trace_jsonl' --output-on-failure
```

完成 TODO 后运行：

```bash
./.build/experiments/e01_ring_dependency/e01_ring_sim
./.build/experiments/e01_ring_dependency/e01_ring_sim --jsonl
./scripts/check.sh
```

最终应有 96 条 JSONL 事件，且每个 rank 都持有相同的 8 个完整规约 chunk。

验证结果：

- ReduceScatter 与 AllGather 各三个 step 通过；
- 四个 rank 最终拥有相同的8个完整结果，mask 均为 `0x0f`；
- 96 条 JSONL 事件可独立解析，缺失或交换必要事件会被拒绝；
- 普通测试、全仓库回归、AddressSanitizer 和浏览器动画验收通过。

## Day 07 任务：单点延迟与因果传播

Day 07 不改变 Ring 数值计算，只在已经生成的 96 条事件上建立因果关系。框架文件是 include/delay.h、src/delay.c 和 tests/test_delay_propagation.c。

你需要完成三个接口：

1. delay_find_message_predecessor：为 recv 找到同一次传输的 send；
2. delay_find_rank_predecessor：找到同一 rank 上离当前事件最近的前一个事件；
3. ring_apply_causal_delay：只给根事件主动增加延迟，再让延迟沿上述依赖传播。

固定根事件是 ReduceScatter 的 step=0, rank=1, chunk=0, send，主动延迟 100 个逻辑时间单位。只有它应标记为 injected_root；后继事件只能标记为 affected，不能再次主动加 100。

先观察预期失败：

~~~bash
cmake -S . -B .build
cmake --build .build --target test_e01_delay_propagation e01_ring_sim
ctest --test-dir .build -R e01_delay_propagation --output-on-failure
~~~

实现后观察 JSONL：

~~~bash
./.build/experiments/e01_ring_dependency/e01_ring_sim \
  --jsonl \
  --inject-delay reduce_scatter 0 1 0 send 100
~~~

验收时必须同时满足：

- 恰好一个 injected_root；
- 至少一个下游 rank 出现 affected；
- 未受影响事件保持基线时间；
- AllReduce 数值与 contributor mask 完全不变。
