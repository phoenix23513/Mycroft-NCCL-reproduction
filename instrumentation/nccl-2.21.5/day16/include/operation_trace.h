#pragma once

#include <cstddef>
#include <cstdint>

namespace mycroft_trace::day16 {

// M2 进程内采集接口草案；范围与验收见当前计划及本目录 README。
// op_seq_candidate 须在普通串行 RING/SIMPLE NET/IB 中动态对账。
// 当前无周期 state/peer 采集接口；M2 实现时需补齐，不支持 Graph/完整 P2P。
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

struct OperationCompletion {
  OperationKey key;
  std::uint64_t observed_at_ns = 0;  // 和 started_at_ns 使用同一进程单调时钟。
};

enum class CaptureStatus {
  Recorded,
  Dropped,          // M2 有界缓冲区满时显式记录丢失，不阻塞通信。
  InvalidArgument,
  NotImplemented,
};

// 以下函数目前统一返回 NotImplemented，不能接入 NCCL 当成可用采集器。
CaptureStatus initialize_capture(std::size_t capacity) noexcept;
CaptureStatus record_operation_begin(const OperationBegin& record) noexcept;
CaptureStatus record_channel_membership(const ChannelMembership& record) noexcept;

// 只允许整个本地 CollOp 的完成观测调用此入口。
// API 返回、单个 channel 或 Proxy request 完成均不能直接调用。
CaptureStatus record_operation_completion(const OperationCompletion& record) noexcept;

// 此导出接口用于受控停止后的导出；后台批量导出方案须在 M2 确定。
// 热路径不格式化 JSON、不执行文件 I/O。
// capture_id 和 comm_hash 共同形成 Event v2 communicator_id，channel=null。
CaptureStatus export_event_v2(const char* capture_id, const char* output_path) noexcept;

}  // namespace mycroft_trace::day16
