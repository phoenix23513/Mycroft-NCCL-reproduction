// Supplied clock fixtures exercise the gate; this is not a GPU or RDMA experiment.
#include "send_delay.h"
#include <cassert>
#include <cstdint>
#include <limits>

int main() {
  using mycroft_m3::SendDelay;
  SendDelay disabled({0,6,0,0});
  assert(disabled.allow(100,0,6,0,0) && !disabled.observation().started);
  SendDelay gate({0,6,0,100});
  assert(gate.allow(100,1,6,0,0));
  assert(gate.allow(100,0,5,0,0));
  assert(gate.allow(100,0,6,1,0));
  assert(!gate.observation().started);
  assert(!gate.allow(100,0,6,0,0));
  assert(!gate.allow(199,0,6,0,0));
  assert(gate.allow(150,0,6,1,0)); // Another channel continues while the target waits.
  assert(gate.observation().deadline_ns==200 && gate.observation().blocked_attempts==2);
  assert(gate.allow(200,0,6,0,0));
  assert(gate.observation().released && gate.observation().release_ns==200);
  assert(gate.allow(201,0,6,0,2)); // Never delay a later slice again.
  assert(gate.observation().begin_ns==100 && !gate.observation().error);
  SendDelay overflow({0,6,0,100});
  assert(overflow.allow(std::numeric_limits<std::uint64_t>::max()-50,0,6,0,0));
  assert(overflow.observation().error);
  SendDelay regression({0,6,0,100});
  assert(!regression.allow(100,0,6,0,2));
  assert(regression.allow(99,0,6,0,2) && regression.observation().error);
  SendDelay moved({0,6,0,100});
  assert(!moved.allow(100,0,6,0,2));
  assert(moved.allow(110,0,6,0,4) && moved.observation().error);
  SendDelay clock_failure({0,6,0,100});
  clock_failure.invalidate();
  assert(clock_failure.allow(100,0,6,0,0) && clock_failure.observation().error);
}
