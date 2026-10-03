#pragma once
// Independent, allocation-free M3 deadline gate. Caller supplies monotonic ns.
#include <cstdint>
#include <limits>

namespace mycroft_m3 {
struct Target {
  int rank = 0;
  std::uint64_t operation = 6;
  int channel = 0;
  std::uint64_t delay_ns = 0;
};
struct Observation {
  bool started = false, released = false, error = false;
  std::uint64_t begin_ns = 0, deadline_ns = 0, release_ns = 0;
  std::uint64_t held_step = 0, blocked_attempts = 0, last_ns = 0;
};
class SendDelay {
 public:
  explicit SendDelay(Target target = {}) : target_(target) {}
  bool allow(std::uint64_t now, int rank, std::uint64_t op, int channel,
             std::uint64_t step) noexcept {
    if (target_.delay_ns == 0 || rank != target_.rank || op != target_.operation ||
        channel != target_.channel || observation_.released || observation_.error) return true;
    auto& o = observation_;
    if (!o.started) {
      o.started = true; o.begin_ns = now; o.held_step = step;
      if (now > std::numeric_limits<std::uint64_t>::max() - target_.delay_ns) {
        o.error = true; return true; // Fail evidence, not the entire communication loop.
      }
      o.deadline_ns = now + target_.delay_ns;
    } else if (now < o.last_ns || step != o.held_step) {
      o.error = true; return true;
    }
    o.last_ns = now;
    if (now < o.deadline_ns) { ++o.blocked_attempts; return false; }
    o.released = true; o.release_ns = now;
    return true;
  }
  const Target& target() const noexcept { return target_; }
  const Observation& observation() const noexcept { return observation_; }
  void invalidate() noexcept { observation_.error = true; }
 private:
  Target target_;
  Observation observation_;
};
} // namespace mycroft_m3
