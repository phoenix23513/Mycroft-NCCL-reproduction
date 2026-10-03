// Project-owned M3 host hook; NCCL internal types come from NVIDIA's pinned source.
#include "proxy.h"
#include "mycroft_m3.h"
#include "mycroft_m3_send_delay.h"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <time.h>

namespace {
mycroft_m3::SendDelay gate;
std::atomic<bool> configured{false};
int local_rank = -1;
uint64_t comm_hash = 0, connection_id = 0;
}
extern "C" int mycroftM3Configure(int local, int rank, uint64_t op, int channel, uint64_t ns) {
  if (configured || local < 0 || local > 1 || rank < 0 || rank > 1 || op > 8 ||
      channel < 0 || channel > 1 || ns > 2000000000ull) return 1;
  gate = mycroft_m3::SendDelay({rank, op, channel, ns});
  local_rank = local; configured.store(true, std::memory_order_release);
  return 0;
}
bool mycroftM3AllowSend(ncclProxyArgs* args, ncclProxySubArgs* sub) {
  if (!configured.load(std::memory_order_acquire) || gate.target().delay_ns == 0 || !sub->mycroft_m2_tag.enabled) return true;
  const auto& tag = sub->mycroft_m2_tag;
  if (tag.rank != gate.target().rank || tag.op_seq != gate.target().operation ||
      sub->channelId != gate.target().channel) return true;
  timespec t{};
  if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
    gate.invalidate();
    return true;
  }
  const uint64_t now = uint64_t(t.tv_sec)*1000000000ull + uint64_t(t.tv_nsec);
  const bool started = gate.observation().started, released = gate.observation().released;
  const bool allow = gate.allow(now, tag.rank, tag.op_seq, sub->channelId, sub->transmitted);
  if (!started && gate.observation().started) {
    comm_hash = tag.comm_hash; connection_id = tag.connection_id;
    mycroftM2SendSample(args, sub, true);
  }
  if (!released && gate.observation().released) mycroftM2SendSample(args, sub, true);
  return allow;
}
extern "C" int mycroftM3Finish(const char* directory) {
  if (!configured || !directory) return 1;
  const auto& t = gate.target(); const auto& o = gate.observation();
  std::ofstream out(std::filesystem::path(directory) / "injection.json");
  out << "{\"injection_version\":1,\"clock\":\"CLOCK_MONOTONIC\",\"kind\":\"software_isend_delay\","
      << "\"local_rank\":" << local_rank << ",\"target_rank\":" << t.rank
      << ",\"target_operation\":" << t.operation << ",\"target_channel\":" << t.channel
      << ",\"delay_ns\":" << t.delay_ns << ",\"comm_hash\":" << comm_hash
      << ",\"connection_id\":" << connection_id << ",\"held_step\":" << o.held_step
      << ",\"begin_ns\":" << o.begin_ns << ",\"deadline_ns\":" << o.deadline_ns
      << ",\"release_ns\":" << o.release_ns << ",\"blocked_attempts\":" << o.blocked_attempts
      << ",\"started\":" << (o.started ? "true" : "false")
      << ",\"released\":" << (o.released ? "true" : "false")
      << ",\"error\":" << (o.error ? "true" : "false") << "}\n";
  out.close();
  return out && !o.error ? 0 : 2;
}
