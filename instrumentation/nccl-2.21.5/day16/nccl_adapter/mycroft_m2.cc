// Independently implemented host instrumentation. NCCL types come from NVIDIA's pinned source.
#include "comm.h"
#include "info.h"
#include "transport.h"
#include "cudawrap.h"
#include "mycroft_m2.h"
#include "operation_trace.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <time.h>

namespace {
namespace trace = mycroft_trace::day16;
std::atomic<bool> active{false};
std::atomic<bool> producer_joined{false};
std::atomic<int> error{0};
std::atomic<uint64_t> send_samples{0}, recv_samples{0};
ncclComm* target = nullptr;
ncclProxyState* target_proxy = nullptr;
cudaStream_t target_stream = nullptr;
cudaEvent_t completion_event = nullptr;
ncclKernelPlan* pending_plan = nullptr;
trace::OperationKey operation;
uint64_t started_ns = 0, logical_bytes = 0, plan_id = 0, completed_operations = 0;
bool identified = false, event_recorded = false, completed = true;

void fail(int code) {
  int zero = 0;
  error.compare_exchange_strong(zero, code, std::memory_order_relaxed);
}
uint64_t now_ns() {
  timespec t{};
  if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) { fail(1); return 0; }
  return uint64_t(t.tv_sec)*1000000000ull + uint64_t(t.tv_nsec);
}
void accept(trace::CaptureStatus status) {
  // Skipped is pacing; Dropped stays in the recorder's explicit loss counters.
  if (status != trace::CaptureStatus::Recorded && status != trace::CaptureStatus::Skipped &&
      status != trace::CaptureStatus::Dropped) fail(2);
}
trace::ConnectionRef reference(ncclProxySubArgs* sub) {
  const auto& tag = sub->mycroft_m2_tag;
  return {{tag.comm_hash, tag.op_seq, tag.rank}, sub->channelId, tag.connection_id};
}
trace::SamplingState sampling(const MycroftM2Sampling& value) {
  return {value.initialized, value.last_observed, value.last_sampled};
}
void store_sampling(MycroftM2Sampling& dest, const trace::SamplingState& src) {
  dest = {src.initialized, src.last_observed_at_ns, src.last_sampled_at_ns};
}
}

extern "C" int mycroftM2Start(void* comm_value, void* stream_value, uint64_t capacity, uint64_t period) {
  if (active.load() || completion_event) return 1;
  target = static_cast<ncclComm*>(comm_value);
  target_stream = static_cast<cudaStream_t>(stream_value);
  if (!target || !target_stream || target->nRanks != 2 || target->nNodes != 2 || target->localRanks != 1 ||
      target->intraRanks != 1 || target->config.blocking != 1)
    return 2;
  for (int r = 0; r < target->nRanks; ++r) if (target->topParentRanks[r] != r) return 2;
  error.store(0); send_samples.store(0); recv_samples.store(0);
  target_proxy = target->proxyState; producer_joined.store(false);
  plan_id = completed_operations = 0; completed = true; pending_plan = nullptr;
  if (trace::initialize_capture({static_cast<std::size_t>(capacity), period}) != trace::CaptureStatus::Recorded)
    return 3;
  if (cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming) != cudaSuccess) {
    trace::stop_capture(); return 4;
  }
  active.store(true, std::memory_order_release);
  return 0;
}
void mycroftM2Collective(ncclComm* comm, ncclKernelPlan* plan, ncclInfo* info) {
  if (!active.load(std::memory_order_acquire)) return;
  if (comm != target || !completed || plan->collOpCount != 1 || plan->persistent ||
      comm->persistentRefs != 0 || ncclCudaLaunchBlocking || info->coll != ncclFuncAllReduce ||
      info->datatype != ncclFloat || info->algorithm != NCCL_ALGO_RING ||
      info->protocol != NCCL_PROTO_SIMPLE || info->regBufType != NCCL_REGULAR_BUFFER ||
      info->stream != target_stream) { fail(5); return; }
  pending_plan = plan;
  ++plan_id;
  started_ns = now_ns(); logical_bytes = info->nBytes;
  completed = false; identified = false; event_recorded = false;
}
void mycroftM2Launched(ncclComm* comm, ncclKernelPlan* plan, cudaStream_t stream) {
  if (!active.load(std::memory_order_acquire)) return;
  if (comm != target || pending_plan != plan || plan->collOpCount != 1 || plan->persistent ||
      stream != target_stream || comm->tasks.streams == nullptr || comm->tasks.streams->next != nullptr ||
      error.load()) { fail(6); return; }
  if (cudaEventRecord(completion_event, stream) != cudaSuccess) { fail(7); return; }
  event_recorded = true;
}
void mycroftM2Identity(ncclComm* comm, ncclKernelPlan* plan, uint64_t historical_base) {
  if (!active.load(std::memory_order_acquire)) return;
  if (comm != target || pending_plan != plan || !event_recorded || identified ||
      plan->collOpCount != 1 || !plan->hasProxyOps || error.load()) { fail(8); return; }
  operation = {comm->commHash, historical_base, comm->rank};
  accept(trace::record_operation_begin({operation, started_ns, logical_bytes}));
  accept(trace::record_plan_binding({operation, plan_id, 1, static_cast<uint32_t>(plan->collOpCount)}));
  for (int c = 0; c < plan->channelUbound; ++c) {
    if ((plan->channelMask >> c) & 1) accept(trace::record_channel_membership({operation, c}));
  }
  identified = true;
}
void mycroftM2Attach(ncclComm* comm, ncclChannel* channel, int direction, int peer,
                     ncclConnector* connector, ncclProxyOp* op) {
  op->mycroft_m2_tag.enabled = 0;
  if (!active.load(std::memory_order_acquire)) return;
  const auto* expected_transport = direction == 1 ? &netTransport.send : &netTransport.recv;
  if (comm != target || !identified || error.load() || op->opCount != (operation.op_seq_candidate << 1) ||
      connector->transportComm != expected_transport || !comm->ncclNet ||
      std::strcmp(comm->ncclNet->name, "IB") != 0 || op->coll != ncclFuncAllReduce ||
      op->protocol != NCCL_PROTO_SIMPLE || op->reg || peer < 0 || peer == comm->rank) { fail(9); return; }
  const uint64_t connection_id = uint64_t(channel->id)*2 + uint64_t(direction);
  op->mycroft_m2_tag = {1, operation.comm_hash, operation.op_seq_candidate, comm->rank, peer, connection_id};
  accept(trace::record_peer_connection({{operation, channel->id, connection_id}, peer,
      direction == 1 ? trace::Direction::Send : trace::Direction::Receive, trace::Transport::NetIb}));
}
void mycroftM2Ready(ncclProxyArgs* args, ncclProxySubArgs* sub) {
  if (!active.load(std::memory_order_acquire) || !sub->mycroft_m2_tag.enabled) return;
  const uint64_t ready = sub->transmitted + args->sliceSteps;
  if (ready > static_cast<uint64_t>(sub->nsteps)) { fail(10); return; }
  // Observed readiness frontier; repeated isend attempts for the same slice must not double-count.
  sub->mycroft_m2_ready = std::max(sub->mycroft_m2_ready, ready);
}
void mycroftM2SendSample(ncclProxyArgs* args, ncclProxySubArgs* sub, bool force) {
  if (!active.load(std::memory_order_acquire) || !sub->mycroft_m2_tag.enabled) return;
  const trace::SendProgress progress{reference(sub), now_ns(), sub->base,
      static_cast<uint64_t>(sub->nsteps), static_cast<uint64_t>(args->sliceSteps), sub->reg != 0,
      sub->mycroft_m2_ready, sub->transmitted, sub->done};
  auto state = sampling(sub->mycroft_m2_sampling);
  const auto status = force ? trace::record_send_progress(progress) : trace::sample_send_progress(progress, state);
  store_sampling(sub->mycroft_m2_sampling, state); accept(status);
  if (status == trace::CaptureStatus::Recorded) send_samples.fetch_add(1, std::memory_order_relaxed);
}
void mycroftM2RecvSample(ncclProxyArgs* args, ncclProxySubArgs* sub, bool force) {
  if (!active.load(std::memory_order_acquire) || !sub->mycroft_m2_tag.enabled) return;
  const trace::ReceiveProgress progress{reference(sub), now_ns(), sub->base,
      static_cast<uint64_t>(sub->nsteps), static_cast<uint64_t>(args->sliceSteps), sub->reg != 0,
      sub->posted, sub->received, sub->transmitted, sub->done};
  auto state = sampling(sub->mycroft_m2_sampling);
  const auto status = force ? trace::record_receive_progress(progress) : trace::sample_receive_progress(progress, state);
  store_sampling(sub->mycroft_m2_sampling, state); accept(status);
  if (status == trace::CaptureStatus::Recorded) recv_samples.fetch_add(1, std::memory_order_relaxed);
}
extern "C" int mycroftM2Complete(void* comm_value) {
  if (!active.load() || comm_value != target || !identified || !event_recorded || completed || error.load()) return 1;
  ncclResult_t async_status;
  if (__atomic_load_n(target->abortFlag, __ATOMIC_ACQUIRE) != 0 ||
      ncclCommGetAsyncError(target, &async_status) != ncclSuccess || async_status != ncclSuccess) { fail(11); return 2; }
  if (cudaEventQuery(completion_event) != cudaSuccess) { fail(12); return 3; }
  accept(trace::record_operation_completion({operation, now_ns(), plan_id, 1}));
  if (error.load()) return 4;
  completed = true; ++completed_operations;
  return 0;
}
extern "C" int mycroftM2Finish(const char* capture_id, const char* directory) {
  // Teardown APIs alone do not prove quiescence; ncclCommAbort masks some cleanup errors.
  if (!producer_joined.load(std::memory_order_acquire)) return 4;
  if (!active.exchange(false)) return 1;
  accept(trace::stop_capture());
  const auto result = trace::export_capture(capture_id, directory);
  const auto destroy_status = cudaEventDestroy(completion_event);
  completion_event = nullptr;
  if (result != trace::CaptureStatus::Recorded || destroy_status != cudaSuccess) return 2;
  std::ofstream manifest(std::filesystem::path(directory) / "adapter-manifest.json");
  manifest << "{\"adapter_version\":1,\"base_commit\":\"ab2b89c4c339bd7f816fbc114a4b05d386b66290\","
      << "\"clock\":\"CLOCK_MONOTONIC\",\"scope\":\"two_node_serial_float32_ring_simple_net_ib\","
      << "\"completion_observation\":\"actual_launch_stream_event_query_after_application_stream_sync\","
      << "\"single_plan_collectives\":1,\"stream_matches_user_stream\":true,\"producer_join_verified\":true,"
      << "\"readiness\":\"observed_eligible_slice_frontier_before_isend\","
      << "\"exact_gpu_timestamp\":false,\"hardware_fault_proof\":false,"
      << "\"completed_operations\":" << completed_operations << ",\"plan_count\":" << plan_id
      << ",\"send_samples\":" << send_samples.load() << ",\"recv_samples\":" << recv_samples.load()
      << ",\"adapter_error\":" << error.load() << ",\"pending_operation\":" << (completed ? "false" : "true") << "}\n";
  manifest.close();
  return manifest && error.load() == 0 && completed ? 0 : 3;
}
void mycroftM2ProxyStopped(ncclProxyState* state) {
  if (active.load(std::memory_order_acquire) && state == target_proxy)
    producer_joined.store(true, std::memory_order_release);
}
