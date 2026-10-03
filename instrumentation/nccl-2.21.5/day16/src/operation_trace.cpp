#include "operation_trace.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace mycroft_trace::day16 {
namespace {
using Record = std::variant<OperationBegin, ChannelMembership, PeerConnection,
                            PlanBinding, SendProgress, ReceiveProgress, OperationCompletion>;
static_assert(std::is_trivially_copyable_v<Record>);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(std::atomic<CapturePhase>::is_always_lock_free);
std::unique_ptr<Record[]> records;
CaptureConfig config;
std::atomic<CapturePhase> phase{CapturePhase::Uninitialized};
std::atomic<std::uint64_t> reserved{0}, recorded{0}, dropped{0}, invalid{0}, unsupported{0};

bool valid_key(const OperationKey& key) noexcept { return key.rank >= 0; }
bool valid_ref(const ConnectionRef& ref) noexcept {
  return valid_key(ref.key) && ref.channel >= 0;
}
CaptureStatus check(const OperationBegin& r) noexcept {
  return valid_key(r.key) ? CaptureStatus::Recorded : CaptureStatus::InvalidArgument;
}
CaptureStatus check(const ChannelMembership& r) noexcept {
  return valid_key(r.key) && r.channel >= 0 ? CaptureStatus::Recorded : CaptureStatus::InvalidArgument;
}
CaptureStatus check(const PeerConnection& r) noexcept {
  if (!valid_ref(r.ref) || r.peer_rank < 0 || r.peer_rank == r.ref.key.rank ||
      (r.direction != Direction::Send && r.direction != Direction::Receive))
    return CaptureStatus::InvalidArgument;
  return r.transport == Transport::NetIb ? CaptureStatus::Recorded : CaptureStatus::Unsupported;
}
CaptureStatus check(const PlanBinding& r) noexcept {
  if (!valid_key(r.key) || r.collective_count == 0) return CaptureStatus::InvalidArgument;
  return r.collective_count == 1 ? CaptureStatus::Recorded : CaptureStatus::Unsupported;
}
CaptureStatus check(const SendProgress& r) noexcept {
  if (!valid_ref(r.ref) || r.nsteps == 0 || r.slice_steps == 0) return CaptureStatus::InvalidArgument;
  // Supported scope is ordinary unregistered buffers; reject the distinct registered semantics.
  if (r.registered_buffer) return CaptureStatus::Unsupported;
  if (r.nsteps % r.slice_steps || r.gpu_ready_steps % r.slice_steps ||
      r.transmitted_steps % r.slice_steps || r.done_steps % r.slice_steps ||
      r.done_steps > r.transmitted_steps || r.transmitted_steps > r.gpu_ready_steps ||
      r.gpu_ready_steps > r.nsteps) return CaptureStatus::InvalidArgument;
  return CaptureStatus::Recorded;
}
CaptureStatus check(const ReceiveProgress& r) noexcept {
  if (!valid_ref(r.ref) || r.nsteps == 0 || r.slice_steps == 0) return CaptureStatus::InvalidArgument;
  if (r.registered_buffer) return CaptureStatus::Unsupported;
  if (r.nsteps % r.slice_steps || r.posted_steps % r.slice_steps ||
      r.received_steps % r.slice_steps || r.transmitted_steps % r.slice_steps ||
      r.done_steps % r.slice_steps || r.done_steps > r.transmitted_steps ||
      r.transmitted_steps > r.received_steps || r.received_steps > r.posted_steps ||
      r.posted_steps > r.nsteps) return CaptureStatus::InvalidArgument;
  return CaptureStatus::Recorded;
}
CaptureStatus check(const OperationCompletion& r) noexcept {
  return valid_key(r.key) ? CaptureStatus::Recorded : CaptureStatus::InvalidArgument;
}
CaptureStatus count_failure(CaptureStatus status) noexcept {
  if (status == CaptureStatus::InvalidArgument) invalid.fetch_add(1, std::memory_order_relaxed);
  if (status == CaptureStatus::Unsupported) unsupported.fetch_add(1, std::memory_order_relaxed);
  return status;
}
template <typename T> CaptureStatus append(const T& r) noexcept {
  if (phase.load(std::memory_order_acquire) != CapturePhase::Running) return CaptureStatus::InvalidState;
  const auto status = check(r);
  if (status != CaptureStatus::Recorded) return count_failure(status);
  // 固定一次原子预约，没有锁、重试环、分配或读者等待。极端计数溢出不在支持范围。
  const auto index = reserved.fetch_add(1, std::memory_order_relaxed);
  if (index >= config.capacity) {
    dropped.fetch_add(1, std::memory_order_relaxed);
    return CaptureStatus::Dropped;
  }
  records[index] = r;  // 每个生产者只写自己的槽；停止前所有生产者必须已退出。
  recorded.fetch_add(1, std::memory_order_relaxed);
  return CaptureStatus::Recorded;
}
template <typename T> CaptureStatus sample(const T& r, SamplingState& state) noexcept {
  if (phase.load(std::memory_order_acquire) != CapturePhase::Running) return CaptureStatus::InvalidState;
  const auto status = check(r);
  if (status != CaptureStatus::Recorded) return count_failure(status);
  if (state.initialized && r.observed_at_ns < state.last_observed_at_ns)
    return count_failure(CaptureStatus::InvalidArgument);
  state.last_observed_at_ns = r.observed_at_ns;
  if (state.initialized && r.observed_at_ns - state.last_sampled_at_ns < config.sample_period_ns)
    return CaptureStatus::Skipped;
  state.initialized = true;
  state.last_sampled_at_ns = r.observed_at_ns;
  return append(r);  // 进度未变仍采样；满了也推进周期，避免不断尝试写满缓冲区。
}

using OpId = std::tuple<std::uint64_t, std::uint64_t, std::int32_t>;
using ConnId = std::tuple<OpId, std::int32_t, std::uint64_t>;
OpId id(const OperationKey& k) { return {k.comm_hash, k.op_seq_candidate, k.rank}; }
ConnId id(const ConnectionRef& r) { return {id(r.key), r.channel, r.connection_id}; }
const OperationKey& key(const OperationBegin& r) { return r.key; }
const OperationKey& key(const ChannelMembership& r) { return r.key; }
const OperationKey& key(const PlanBinding& r) { return r.key; }
const OperationKey& key(const OperationCompletion& r) { return r.key; }
const OperationKey& key(const PeerConnection& r) { return r.ref.key; }
const OperationKey& key(const SendProgress& r) { return r.ref.key; }
const OperationKey& key(const ReceiveProgress& r) { return r.ref.key; }
std::string key_json(const OperationKey& k) {
  std::ostringstream out;
  out << "\"comm_hash\":" << k.comm_hash << ",\"op_seq_candidate\":" << k.op_seq_candidate
      << ",\"rank\":" << k.rank;
  return out.str();
}
std::string ref_json(const ConnectionRef& r) {
  return key_json(r.key) + ",\"channel\":" + std::to_string(r.channel) +
         ",\"connection_id\":" + std::to_string(r.connection_id);
}
template <typename T> std::string progress_json(const T& r) {
  std::ostringstream out;
  out << ref_json(r.ref) << ",\"observed_at_ns\":" << r.observed_at_ns
      << ",\"step_base\":" << r.step_base << ",\"nsteps\":" << r.nsteps
      << ",\"slice_steps\":" << r.slice_steps << ",\"registered_buffer\":"
      << (r.registered_buffer ? "true" : "false");
  return out.str();
}
std::string communicator_id(const std::string& capture, const OperationKey& k) {
  std::ostringstream out;
  out << capture << ':' << std::hex << std::setw(16) << std::setfill('0') << k.comm_hash;
  return out.str();
}
std::string event_json(const std::string& capture, std::size_t index, const OperationKey& k,
                       const std::string& kind, const std::string& channel, std::uint64_t ns,
                       const std::string& payload) {
  std::ostringstream out;
  out << "{\"schema_version\":2,\"event_id\":\"" << capture << ':' << k.rank << ':' << index
      << "\",\"event_kind\":\"" << kind << "\",\"source\":\"m2_process_local_recorder_unverified\","
      << "\"context\":{\"communicator_id\":\"" << communicator_id(capture, k)
      << "\",\"op_seq\":" << k.op_seq_candidate << ",\"collective\":\"all_reduce\",\"rank\":"
      << k.rank << ",\"channel\":" << channel << "},\"time\":{\"value\":" << ns
      << ",\"domain\":\"nccl_monotonic_ns\"},\"dependencies\":[],\"payload\":{" << payload << "}}";
  return out.str();
}
bool safe_capture_id(const char* value) {
  if (value == nullptr || *value == '\0') return false;
  std::size_t length = 0;
  for (; *value; ++value) {
    const char c = *value;
    if (++length > 128 || !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) return false;
  }
  return true;
}

// 仅在停止后分配索引与格式化。关联缺失/重复时保留原始记录，拒绝生成伪造 Event。
struct Associations {
  std::map<OpId, const OperationBegin*> begins;
  std::map<OpId, const PlanBinding*> plans;
  std::map<ConnId, const PeerConnection*> peers;
  std::map<std::tuple<OpId, std::int32_t, Direction>, std::size_t> peer_counts;
  std::set<std::pair<OpId, std::int32_t>> channels;
  std::map<OpId, std::size_t> completion_counts;
  std::vector<bool> progress_valid;
  std::uint64_t errors = 0;
  void build(std::size_t count) {
    progress_valid.assign(count, true);
    std::map<ConnId, std::vector<std::size_t>> sends, receives;
    for (std::size_t i = 0; i < count; ++i) {
      const auto& r = records[i];
      if (auto p = std::get_if<OperationBegin>(&r)) {
        if (!begins.emplace(id(p->key), p).second) { begins[id(p->key)] = nullptr; ++errors; }
      } else if (auto p = std::get_if<PlanBinding>(&r)) {
        if (!plans.emplace(id(p->key), p).second) { plans[id(p->key)] = nullptr; ++errors; }
      } else if (auto p = std::get_if<PeerConnection>(&r)) {
        if (!peers.emplace(id(p->ref), p).second) { peers[id(p->ref)] = nullptr; ++errors; }
        ++peer_counts[{id(p->ref.key), p->ref.channel, p->direction}];
      } else if (auto p = std::get_if<ChannelMembership>(&r)) {
        if (!channels.emplace(id(p->key), p->channel).second) ++errors;
      } else if (auto p = std::get_if<OperationCompletion>(&r)) ++completion_counts[id(p->key)];
      else if (auto p = std::get_if<SendProgress>(&r)) sends[id(p->ref)].push_back(i);
      else if (auto p = std::get_if<ReceiveProgress>(&r)) receives[id(p->ref)].push_back(i);
    }
    validate_series<SendProgress>(sends);
    validate_series<ReceiveProgress>(receives);
    for (const auto& entry : begins) {
      if (completion_counts[entry.first] != 1 || !has_channel(entry.first)) ++errors;
    }
    for (const auto& channel : channels) {
      const auto op = begins.find(channel.first);
      if (op == begins.end() || !op->second || peer_count(channel.first, channel.second, Direction::Send) != 1 ||
          peer_count(channel.first, channel.second, Direction::Receive) != 1) ++errors;
    }
    for (const auto& peer : peers) {
      if (peer.second && !channels.count({id(peer.second->ref.key), peer.second->ref.channel})) ++errors;
    }
  }
  bool has_channel(const OpId& op) const {
    auto channel = channels.lower_bound({op, -1});
    return channel != channels.end() && channel->first == op;
  }
  std::size_t peer_count(const OpId& op, std::int32_t channel, Direction direction) const {
    auto entry = peer_counts.find({op, channel, direction});
    return entry == peer_counts.end() ? 0 : entry->second;
  }
  template <typename T> void validate_series(std::map<ConnId, std::vector<std::size_t>>& groups) {
    for (auto& group : groups) {
      auto& indices = group.second;
      std::sort(indices.begin(), indices.end(), [](std::size_t a, std::size_t b) {
        const auto ta = std::get<T>(records[a]).observed_at_ns;
        const auto tb = std::get<T>(records[b]).observed_at_ns;
        return std::tie(ta, a) < std::tie(tb, b);
      });
      bool valid = true;
      for (std::size_t j = 1; j < indices.size(); ++j) {
        const auto& a = std::get<T>(records[indices[j-1]]);
        const auto& b = std::get<T>(records[indices[j]]);
        valid = valid && a.nsteps == b.nsteps && a.slice_steps == b.slice_steps &&
                a.step_base == b.step_base && a.transmitted_steps <= b.transmitted_steps &&
                a.done_steps <= b.done_steps;
        if constexpr (std::is_same_v<T, SendProgress>) valid = valid && a.gpu_ready_steps <= b.gpu_ready_steps;
        else valid = valid && a.posted_steps <= b.posted_steps && a.received_steps <= b.received_steps;
      }
      if (!valid) for (auto i : indices) progress_valid[i] = false;
    }
  }
  bool connected(const ConnectionRef& ref, Direction direction) const {
    auto peer = peers.find(id(ref));
    return channels.count({id(ref.key), ref.channel}) && peer != peers.end() && peer->second &&
           peer->second->direction == direction && peer_count(id(ref.key), ref.channel, direction) == 1;
  }
  const OperationBegin* begin(const OperationKey& k) const {
    auto p = begins.find(id(k));
    return p == begins.end() ? nullptr : p->second;
  }
};

CaptureStatus export_stopped(const char* capture, const char* output) {
  const auto count = static_cast<std::size_t>(recorded.load());
  std::int32_t rank = -1;
  std::uint64_t comm_hash = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto k = std::visit([](const auto& r) { return key(r); }, records[i]);
    if (rank < 0) { rank = k.rank; comm_hash = k.comm_hash; }
    if (rank != k.rank || comm_hash != k.comm_hash) return CaptureStatus::Unsupported;
  }
  Associations association;
  association.build(count);
  // create_directory 对已存在路径拒绝覆盖；错误后保留部分输出，manifest 最后写。
  const std::filesystem::path directory(output);
  if (!std::filesystem::create_directory(directory)) return CaptureStatus::InvalidArgument;
  std::filesystem::create_directory(directory / "trace");
  std::ofstream events, sends, receives;
  std::ofstream channels(directory / "trace/channel-map.jsonl");
  std::ofstream plans(directory / "trace/plan-map.jsonl");
  if (rank >= 0) {
    const auto suffix = "rank" + std::to_string(rank) + ".jsonl";
    events.open(directory / "trace" / suffix);
    sends.open(directory / "trace" / ("send-" + suffix));
    receives.open(directory / "trace" / ("recv-" + suffix));
    if (!events || !sends || !receives) return CaptureStatus::IoError;
  }
  if (!channels || !plans) return CaptureStatus::IoError;
  std::uint64_t emitted = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& r = records[i];
    if (auto p = std::get_if<OperationBegin>(&r)) {
      channels << "{\"record_kind\":\"operation_begin\"," << key_json(p->key)
               << ",\"started_at_ns\":" << p->started_at_ns << ",\"message_bytes\":" << p->message_bytes << "}\n";
    } else if (auto p = std::get_if<ChannelMembership>(&r)) {
      channels << "{\"record_kind\":\"channel_membership\"," << key_json(p->key)
               << ",\"channel\":" << p->channel << "}\n";
    } else if (auto p = std::get_if<PeerConnection>(&r)) {
      channels << "{\"record_kind\":\"peer_connection\"," << ref_json(p->ref)
               << ",\"peer_rank\":" << p->peer_rank << ",\"direction\":\""
               << (p->direction == Direction::Send ? "send" : "recv") << "\",\"transport\":\"NET/IB\"}\n";
    } else if (auto p = std::get_if<PlanBinding>(&r)) {
      plans << "{\"record_kind\":\"plan_binding\"," << key_json(p->key) << ",\"plan_id\":" << p->plan_id
            << ",\"stream_id\":" << p->stream_id << ",\"collective_count\":" << p->collective_count << "}\n";
    } else if (auto p = std::get_if<SendProgress>(&r)) {
      sends << '{' << progress_json(*p) << ",\"gpu_ready_steps\":" << p->gpu_ready_steps
            << ",\"transmitted_steps\":" << p->transmitted_steps << ",\"done_steps\":" << p->done_steps << "}\n";
      const auto* begin = association.begin(p->ref.key);
      if (!begin || p->observed_at_ns < begin->started_at_ns || !association.progress_valid[i] ||
          !association.connected(p->ref, Direction::Send)) { ++association.errors; continue; }
      const auto unit = p->slice_steps;
      const auto payload = "\"total_chunks\":" + std::to_string(p->nsteps / unit) +
          ",\"gpu_ready\":" + std::to_string(p->gpu_ready_steps / unit) +
          ",\"rdma_transmitted\":" + std::to_string(p->transmitted_steps / unit) +
          ",\"rdma_done\":" + std::to_string(p->done_steps / unit);
      events << event_json(capture, i, p->ref.key, "progress_snapshot", std::to_string(p->ref.channel),
                           p->observed_at_ns, payload) << '\n';
      ++emitted;
    } else if (auto p = std::get_if<ReceiveProgress>(&r)) {
      receives << '{' << progress_json(*p) << ",\"posted_steps\":" << p->posted_steps
               << ",\"received_steps\":" << p->received_steps << ",\"transmitted_steps\":"
               << p->transmitted_steps << ",\"done_steps\":" << p->done_steps << "}\n";
      const auto* begin = association.begin(p->ref.key);
      if (!begin || p->observed_at_ns < begin->started_at_ns || !association.progress_valid[i] ||
          !association.connected(p->ref, Direction::Receive)) ++association.errors;
    } else if (auto p = std::get_if<OperationCompletion>(&r)) {
      plans << "{\"record_kind\":\"operation_completion\"," << key_json(p->key) << ",\"observed_at_ns\":"
            << p->observed_at_ns << ",\"plan_id\":" << p->plan_id << ",\"stream_id\":" << p->stream_id << "}\n";
      const auto* begin = association.begin(p->key);
      const auto plan = association.plans.find(id(p->key));
      if (!begin || p->observed_at_ns < begin->started_at_ns || association.completion_counts[id(p->key)] != 1 ||
          !association.has_channel(id(p->key)) ||
          plan == association.plans.end() || !plan->second || plan->second->plan_id != p->plan_id ||
          plan->second->stream_id != p->stream_id) { ++association.errors; continue; }
      events << event_json(capture, i, p->key, "operation_completion", "null", p->observed_at_ns,
          "\"started_at_ns\":" + std::to_string(begin->started_at_ns) +
          ",\"message_bytes\":" + std::to_string(begin->message_bytes)) << '\n';
      ++emitted;
    }
  }
  // close 的失败也计入 I/O 错误，不能在写盘失败后发布完整 manifest。
  channels.close(); plans.close();
  if (!channels || !plans) return CaptureStatus::IoError;
  if (rank >= 0) {
    events.close(); sends.close(); receives.close();
    if (!events || !sends || !receives) return CaptureStatus::IoError;
  }
  std::ofstream manifest(directory / "capture-manifest.json");
  const bool local_valid = count > 0 && association.errors == 0 && dropped.load() == 0 &&
                           invalid.load() == 0 && unsupported.load() == 0;
  // Recorder alone cannot verify NCCL provenance. The adapter and offline verifier add that evidence separately.
  manifest << "{\"manifest_version\":1,\"capture_id\":\"" << capture << "\",\"rank\":"
           << (rank < 0 ? "null" : std::to_string(rank)) << ",\"phase\":\"stopped\",\"capacity\":"
           << config.capacity << ",\"sample_period_ns\":" << config.sample_period_ns
           << ",\"recorded\":" << count << ",\"dropped\":" << dropped.load()
           << ",\"invalid\":" << invalid.load() << ",\"unsupported\":" << unsupported.load()
           << ",\"event_records\":" << emitted << ",\"association_errors\":" << association.errors
           << ",\"local_contract_valid\":" << (local_valid ? "true" : "false")
           << ",\"nccl_adapter_verified\":false,\"completion_observation_verified\":false,"
           << "\"evidence_scope\":\"process_local_recorder_only\",\"time_source\":\"caller_supplied_unverified\","
           << "\"diagnostic_eligible\":false,\"export_complete\":true}\n";
  manifest.close();
  return manifest ? CaptureStatus::Recorded : CaptureStatus::IoError;
}
}  // namespace

CaptureStatus initialize_capture(const CaptureConfig& value) noexcept {
  if (phase.load() == CapturePhase::Running) return CaptureStatus::InvalidState;
  if (value.capacity == 0 || value.sample_period_ns == 0 ||
      value.capacity > std::numeric_limits<std::size_t>::max() / sizeof(Record)) return CaptureStatus::InvalidArgument;
  try {
    auto allocated = std::make_unique<Record[]>(value.capacity);
    records = std::move(allocated);
    config = value;
    reserved.store(0); recorded.store(0); dropped.store(0); invalid.store(0); unsupported.store(0);
    phase.store(CapturePhase::Running, std::memory_order_release);
    return CaptureStatus::Recorded;
  } catch (...) { return CaptureStatus::AllocationFailed; }
}
CaptureStatus record_operation_begin(const OperationBegin& r) noexcept { return append(r); }
CaptureStatus record_channel_membership(const ChannelMembership& r) noexcept { return append(r); }
CaptureStatus record_peer_connection(const PeerConnection& r) noexcept { return append(r); }
CaptureStatus record_plan_binding(const PlanBinding& r) noexcept { return append(r); }
CaptureStatus record_send_progress(const SendProgress& r) noexcept { return append(r); }
CaptureStatus record_receive_progress(const ReceiveProgress& r) noexcept { return append(r); }
CaptureStatus sample_send_progress(const SendProgress& r, SamplingState& state) noexcept { return sample(r, state); }
CaptureStatus sample_receive_progress(const ReceiveProgress& r, SamplingState& state) noexcept { return sample(r, state); }
// The NCCL adapter gates this call on a single-operation plan and actual stream event/abort/error observations.
CaptureStatus record_operation_completion(const OperationCompletion& r) noexcept { return append(r); }
CaptureStatus stop_capture() noexcept {
  if (phase.load() != CapturePhase::Running) return CaptureStatus::InvalidState;
  phase.store(CapturePhase::Stopped, std::memory_order_release);
  return CaptureStatus::Recorded;
}
CaptureStatus read_capture_stats(CaptureStats& stats) noexcept {
  if (phase.load(std::memory_order_acquire) != CapturePhase::Stopped) return CaptureStatus::InvalidState;
  stats = {CapturePhase::Stopped, recorded.load(), dropped.load(), invalid.load(), unsupported.load()};
  return CaptureStatus::Recorded;
}
CaptureStatus export_capture(const char* capture, const char* output) noexcept {
  if (phase.load(std::memory_order_acquire) != CapturePhase::Stopped) return CaptureStatus::InvalidState;
  if (!safe_capture_id(capture) || output == nullptr || *output == '\0') return CaptureStatus::InvalidArgument;
  try { return export_stopped(capture, output); }
  catch (...) { return CaptureStatus::IoError; }
}
}  // namespace mycroft_trace::day16
