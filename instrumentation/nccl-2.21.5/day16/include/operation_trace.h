#pragma once

#include <cstddef>
#include <cstdint>

namespace mycroft_trace::day16 {

// M2 进程内采集接口草案；范围与验收见当前计划及本目录 README。
// op_seq_candidate 须在普通串行 RING/SIMPLE NET/IB 中动态对账。
// 已实现进程内记录与停止后导出；没有接入 NCCL，不支持 Graph/完整 P2P。
struct OperationKey {
  std::uint64_t comm_hash = 0;
  std::uint64_t op_seq_candidate = 0;
  std::int32_t rank = -1;
};

struct OperationBegin {
  OperationKey key;
  std::uint64_t started_at_ns = 0;
  std::uint64_t message_bytes = 0;  // 整个 collective 的逻辑大小，不是 Proxy slice 大小。
};

struct ChannelMembership {
  OperationKey key;
  std::int32_t channel = -1;
};

enum class Direction { Unknown, Send, Receive };
enum class Transport { Unknown, NetIb };

// connection_id 是本进程采集器分配的紧凑编号，不是指针或跨 rank 公共 ID。
// 对端关联使用 operation/channel/peer/direction，不能比较两端 connection_id。
struct ConnectionRef {
  OperationKey key;
  std::int32_t channel = -1;
  std::uint64_t connection_id = 0;
};

struct PeerConnection {
  ConnectionRef ref;
  std::int32_t peer_rank = -1;  // communicator rank，不直接填 tpRank。
  Direction direction = Direction::Unknown;
  Transport transport = Transport::Unknown;  // 由实际 transport 证据确认。
};

// 均为 operation-relative Proxy step 计数，尚未转换为 Event 的 slice 数。
// step_base 仅供溯源，不能混入累计进度。registered_buffer 保留路径差异：
// 此路径可能动态增加 nsteps，GPU readiness 与网络 step 不一定一一对应。
struct SendProgress {
  ConnectionRef ref;
  std::uint64_t observed_at_ns = 0;
  std::uint64_t step_base = 0;
  std::uint64_t nsteps = 0;
  std::uint64_t slice_steps = 0;
  bool registered_buffer = false;
  std::uint64_t gpu_ready_steps = 0;  // readiness 首次成立时维护；不能用 posted 替代。
  std::uint64_t transmitted_steps = 0;  // isend 返回非空 request 后推进。
  std::uint64_t done_steps = 0;  // 网络 test 完成后推进。
};

// 接收侧含义不同，保留独立记录；不得套入发送侧 ProgressSnapshotPayload。
struct ReceiveProgress {
  ConnectionRef ref;
  std::uint64_t observed_at_ns = 0;
  std::uint64_t step_base = 0;
  std::uint64_t nsteps = 0;
  std::uint64_t slice_steps = 0;
  bool registered_buffer = false;
  std::uint64_t posted_steps = 0;       // irecv 返回非空 request。
  std::uint64_t received_steps = 0;     // 接收 request 的 test 完成。
  std::uint64_t transmitted_steps = 0;  // flush 就绪并向 GPU 发布，不是发送 isend。
  std::uint64_t done_steps = 0;         // GPU 消费确认，不是整个 collective 完成。
};

// 记录真实 launch plan 与执行 stream 的本进程编号；仅单操作计划可关联 completion。
struct PlanBinding {
  OperationKey key;
  std::uint64_t plan_id = 0;
  std::uint64_t stream_id = 0;
  std::uint32_t collective_count = 0;
};

struct OperationCompletion {
  OperationKey key;
  std::uint64_t observed_at_ns = 0;  // 和 started_at_ns 使用同一进程单调时钟。
  std::uint64_t plan_id = 0;
  std::uint64_t stream_id = 0;
};

struct CaptureConfig {
  std::size_t capacity = 0;           // 初始化时分配，热路径不能扩容。
  std::uint64_t sample_period_ns = 0; // 周期到期须采样，即使进度未变。
};

enum class CapturePhase { Uninitialized, Running, Stopped };

struct CaptureStats {
  CapturePhase phase = CapturePhase::Uninitialized;
  std::uint64_t recorded = 0;
  std::uint64_t dropped = 0;
  std::uint64_t invalid = 0;
  std::uint64_t unsupported = 0;
};

// 每个 operation/connection/direction 的生产者独占，不共享或跨操作复用。
struct SamplingState {
  bool initialized = false;
  std::uint64_t last_observed_at_ns = 0;
  std::uint64_t last_sampled_at_ns = 0;
};

enum class CaptureStatus {
  Recorded,
  Dropped,          // M2 有界缓冲区满时显式记录丢失，不阻塞通信。
  Skipped,          // 周期尚未到；不是丢失，进度不变不影响到期采样。
  InvalidArgument,
  InvalidState,     // 未初始化或停止后记录；运行中导出。
  Unsupported,     // 路径/单位/plan 不在已确认范围内。
  AllocationFailed,
  IoError,         // 停止后导出失败；可能保留部分文件，不能视为完整结果。
  NotImplemented,
};

// 初始化/停止/导出由控制线程串行调用，与 record 不得并发。
// record 可由多个生产者并发调用，不分配内存、不做 I/O；需要 Linux 64 位无锁原子。
CaptureStatus initialize_capture(const CaptureConfig& config) noexcept;
CaptureStatus record_operation_begin(const OperationBegin& record) noexcept;
CaptureStatus record_channel_membership(const ChannelMembership& record) noexcept;
CaptureStatus record_peer_connection(const PeerConnection& record) noexcept;
CaptureStatus record_plan_binding(const PlanBinding& record) noexcept;
CaptureStatus record_send_progress(const SendProgress& record) noexcept;
CaptureStatus record_receive_progress(const ReceiveProgress& record) noexcept;
CaptureStatus sample_send_progress(const SendProgress& record, SamplingState& state) noexcept;
CaptureStatus sample_receive_progress(const ReceiveProgress& record, SamplingState& state) noexcept;

// 只允许整个本地 CollOp 的完成观测调用此入口。
// API 返回、单个 channel 或 Proxy request 完成均不能直接调用。
// 调用者还须检查实际 stream 上 CUDA event 成功、abort 未请求且异步错误为成功。
CaptureStatus record_operation_completion(const OperationCompletion& record) noexcept;

// 先让所有生产者退出，再 stop；不得并发销毁正在写入的缓冲区。
CaptureStatus stop_capture() noexcept;
// 仅在 Stopped 读取一致统计；无效调用不改写 stats。
CaptureStatus read_capture_stats(CaptureStats& stats) noexcept;

// 导出到新的、父目录已存在的本进程目录；两端的合并交给未来 M2 runner。
// 热路径不格式化 JSON、不执行文件 I/O。
// capture_id 和 comm_hash 共同形成 Event v2 communicator_id；仅 completion channel=null。
// 导出 Event、独立接收状态、channel/peer/plan 映射及含 drops 的 manifest，拒绝覆盖。
// 当前 manifest 明确标注 adapter/completion 未验证；成功导出不等于真实采集验收。
CaptureStatus export_capture(const char* capture_id, const char* output_directory) noexcept;

}  // namespace mycroft_trace::day16
