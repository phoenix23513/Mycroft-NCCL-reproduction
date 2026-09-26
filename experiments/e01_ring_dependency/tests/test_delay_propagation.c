#include "delay.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define CHECK(condition, ...)                                                  \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "  assertion failed: ");                          \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, " (%s:%d)\n", __FILE__, __LINE__);                \
            return false;                                                      \
        }                                                                      \
    } while (0)

typedef bool (*TestFunction)(void);

static DelayConfig root_fixture(void) {
    const DelayConfig config = {
        .phase = TRACE_PHASE_REDUCE_SCATTER,
        .step = 0,
        .rank = 1,
        .chunk_id = 0,
        .action = TRACE_ACTION_SEND,
        .delay_ticks = UINT64_C(100),
    };
    return config;
}

static int find_event(
    const RingSim *sim,
    TracePhase phase,
    int step,
    int rank,
    int chunk,
    TraceAction action
) {
    for (int index = 0; index < sim->event_count; ++index) {
        const TraceEvent *event = &sim->events[index];
        if (event->phase == phase && event->step == step &&
            event->rank == rank && event->chunk_id == chunk &&
            event->action == action) {
            return index;
        }
    }
    return -1;
}

static bool build_completed_sim(RingSim *sim) {
    CHECK(ring_sim_init(sim) == RING_OK, "initialization must succeed");
    CHECK(ring_run_all_reduce(sim) == RING_OK, "AllReduce must succeed");
    return true;
}

static bool test_causal_predecessors(void) {
    RingSim sim;
    CHECK(build_completed_sim(&sim), "fixture must be created");

    const int send_index = find_event(
        &sim,
        TRACE_PHASE_REDUCE_SCATTER,
        0,
        1,
        0,
        TRACE_ACTION_SEND
    );
    const int recv_index = find_event(
        &sim,
        TRACE_PHASE_REDUCE_SCATTER,
        0,
        2,
        0,
        TRACE_ACTION_RECV
    );
    CHECK(send_index >= 0 && recv_index >= 0, "fixture events must exist");
    CHECK(
        delay_find_message_predecessor(
            sim.events,
            (size_t)sim.event_count,
            (size_t)recv_index
        ) == send_index,
        "rank 2 recv must depend directly on rank 1 send"
    );
    CHECK(
        delay_find_message_predecessor(
            sim.events,
            (size_t)sim.event_count,
            (size_t)send_index
        ) == -1,
        "a send has no message predecessor"
    );

    const int rank_predecessor = delay_find_rank_predecessor(
        sim.events,
        (size_t)sim.event_count,
        (size_t)recv_index
    );
    CHECK(rank_predecessor >= 0, "rank 2 recv must have a local predecessor");
    CHECK(
        rank_predecessor < recv_index &&
            sim.events[rank_predecessor].rank == sim.events[recv_index].rank,
        "local predecessor must be the nearest earlier event on rank 2"
    );
    return true;
}

static bool test_one_root_propagates_without_changing_values(void) {
    RingSim sim;
    CHECK(build_completed_sim(&sim), "fixture must be created");

    int values[RING_RANKS][RING_CHUNKS];
    uint32_t masks[RING_RANKS][RING_CHUNKS];
    for (int rank = 0; rank < RING_RANKS; ++rank) {
        for (int chunk = 0; chunk < RING_CHUNKS; ++chunk) {
            values[rank][chunk] = sim.chunks[rank][chunk].value;
            masks[rank][chunk] = sim.chunks[rank][chunk].contributor_mask;
        }
    }

    const DelayConfig config = root_fixture();
    DelayReport report;
    CHECK(
        ring_apply_causal_delay(&sim, &config, &report) == RING_OK,
        "causal delay propagation must succeed"
    );

    int root_count = 0;
    int affected_count = 0;
    for (int index = 0; index < sim.event_count; ++index) {
        const TraceEvent *event = &sim.events[index];
        if (event->delay_role == TRACE_DELAY_INJECTED_ROOT) {
            root_count += 1;
            CHECK(
                event->timestamp ==
                    event->baseline_timestamp + config.delay_ticks,
                "the root must receive the configured delay"
            );
        } else if (event->delay_role == TRACE_DELAY_AFFECTED) {
            affected_count += 1;
            CHECK(
                event->timestamp > event->baseline_timestamp,
                "affected events must be shifted by dependencies"
            );
        } else {
            CHECK(
                event->timestamp == event->baseline_timestamp,
                "unaffected events must stay on their baseline"
            );
        }
    }

    CHECK(root_count == 1, "exactly one event must be actively injected");
    CHECK(affected_count > 0, "the root delay must reach later events");
    CHECK(
        report.affected_event_count == affected_count,
        "the report must count all affected events"
    );
    CHECK(
        (report.affected_rank_mask & (UINT32_C(1) << 2)) != 0,
        "the delay must propagate from rank 1 to downstream rank 2"
    );

    for (int rank = 0; rank < RING_RANKS; ++rank) {
        for (int chunk = 0; chunk < RING_CHUNKS; ++chunk) {
            CHECK(
                sim.chunks[rank][chunk].value == values[rank][chunk] &&
                    sim.chunks[rank][chunk].contributor_mask ==
                        masks[rank][chunk],
                "delay analysis must not alter rank=%d chunk=%d",
                rank,
                chunk
            );
        }
    }
    return true;
}

static bool test_missing_root_is_rejected(void) {
    RingSim sim;
    CHECK(build_completed_sim(&sim), "fixture must be created");

    DelayConfig config = root_fixture();
    config.chunk_id = RING_CHUNKS;
    DelayReport report;
    CHECK(
        ring_apply_causal_delay(&sim, &config, &report) ==
            RING_ERROR_EVENT_NOT_FOUND,
        "a nonexistent root event must be rejected"
    );
    return true;
}

static void run_test(
    const char *name,
    TestFunction function,
    int *failure_count
) {
    if (function()) {
        printf("[PASS] %s\n", name);
        return;
    }
    printf("[FAIL] %s\n", name);
    *failure_count += 1;
}

int main(void) {
    int failures = 0;
    run_test("causal predecessors", test_causal_predecessors, &failures);
    run_test(
        "one injected root propagates without changing values",
        test_one_root_propagates_without_changing_values,
        &failures
    );
    run_test("missing root is rejected", test_missing_root_is_rejected, &failures);

    if (failures != 0) {
        fprintf(stderr, "%d test group(s) failed\n", failures);
        return 1;
    }
    return 0;
}
