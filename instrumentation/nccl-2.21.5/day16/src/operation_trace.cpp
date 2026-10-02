#include "operation_trace.h"

namespace mycroft_trace::day16 {

// TODO(M2)：依据真实 Proxy 回调补齐周期 state 与必要接收端采集，归一化 step/slice 单位。
// 只写预分配内存，保留中间窗口；后台或受控停止后导出，进程崩溃时可能缺证。

CaptureStatus initialize_capture(std::size_t) noexcept {
  // TODO(M2)：在目标路径初始化时预分配容量，维护记录数、dropped 与截止状态。
  return CaptureStatus::NotImplemented;
}

CaptureStatus record_operation_begin(const OperationBegin&) noexcept {
  // TODO(M2)：动态对账普通 NET/IB operation 身份，保存逻辑大小及同进程开始时间。
  return CaptureStatus::NotImplemented;
}

CaptureStatus record_channel_membership(const ChannelMembership&) noexcept {
  // TODO(M2)：关联目标 RING/SIMPLE NET/IB 操作的实际 channel、peer 和连接方向。
  return CaptureStatus::NotImplemented;
}

CaptureStatus record_operation_completion(const OperationCompletion&) noexcept {
  // TODO(M2)：核对单操作计划、实际 stream 及 abort/error 后记录整体完成；sub.done 不足。
  return CaptureStatus::NotImplemented;
}

CaptureStatus export_event_v2(const char*, const char*) noexcept {
  // TODO(M2)：受控停止后导出 Event v2 与元数据，保留身份、同一时钟时差、逻辑字节及丢失。
  return CaptureStatus::NotImplemented;
}

}  // namespace mycroft_trace::day16
