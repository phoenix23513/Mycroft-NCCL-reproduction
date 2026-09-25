#include "ring_sim.h"

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

static int input_value(int rank, int chunk) {
    return rank * 10 + chunk + 1;
}

static int reduced_value(int chunk) {
    int total = 0;
    for (int rank = 0; rank < RING_RANKS; ++rank) {
        total += input_value(rank, chunk);
    }
    return total;
}

static bool test_all_gather_schedule(void) {
    const int expected_send[RING_STEPS][RING_CHUNKS_PER_RANK][RING_RANKS] = {
        {{0, 1, 2, 3}, {4, 5, 6, 7}},
        {{3, 0, 1, 2}, {7, 4, 5, 6}},
        {{2, 3, 0, 1}, {6, 7, 4, 5}},
    };

    for (int step = 0; step < RING_STEPS; ++step) {
        for (int lane = 0; lane < RING_CHUNKS_PER_RANK; ++lane) {
            for (int rank = 0; rank < RING_RANKS; ++rank) {
                const int next = ring_next_rank(rank);
                CHECK(
                    ring_all_gather_send_chunk(rank, step, lane) ==
                        expected_send[step][lane][rank],
                    "AllGather send chunk for rank=%d step=%d lane=%d",
                    rank,
                    step,
                    lane
                );
                CHECK(
                    ring_all_gather_recv_chunk(next, step, lane) ==
                        expected_send[step][lane][rank],
                    "AllGather send/receive mismatch"
                );
            }
        }
    }
    return true;
}

static bool test_all_gather_step_order(void) {
    RingSim sim;
    CHECK(ring_sim_init(&sim) == RING_OK, "initialization must succeed");
    CHECK(
        ring_all_gather_step(&sim, 0) == RING_ERROR_INVALID_ARGUMENT,
        "AllGather must not start before ReduceScatter finishes"
    );
    CHECK(
        ring_run_reduce_scatter(&sim) == RING_OK,
        "ReduceScatter setup must succeed"
    );
    CHECK(
        ring_all_gather_step(&sim, 1) == RING_ERROR_INVALID_ARGUMENT,
        "step one must not run before step zero"
    );
    CHECK(
        ring_all_gather_step(&sim, 0) == RING_OK,
        "step zero must succeed"
    );
    CHECK(
        ring_all_gather_step(&sim, 0) == RING_ERROR_INVALID_ARGUMENT,
        "a completed step must not run twice"
    );
    return true;
}

static bool test_final_all_reduce_result(void) {
    RingSim sim;
    CHECK(ring_sim_init(&sim) == RING_OK, "initialization must succeed");
    CHECK(ring_run_all_reduce(&sim) == RING_OK, "AllReduce must succeed");
    CHECK(
        sim.completed_all_gather_steps == RING_STEPS,
        "all AllGather steps must complete"
    );

    for (int rank = 0; rank < RING_RANKS; ++rank) {
        for (int chunk = 0; chunk < RING_CHUNKS; ++chunk) {
            CHECK(
                sim.chunks[rank][chunk].value == reduced_value(chunk),
                "final value for rank=%d chunk=%d",
                rank,
                chunk
            );
            CHECK(
                sim.chunks[rank][chunk].contributor_mask == UINT32_C(0x0f),
                "final contributor mask for rank=%d chunk=%d",
                rank,
                chunk
            );
        }
    }
    return true;
}

static bool test_complete_event_log(void) {
    RingSim sim;
    CHECK(ring_sim_init(&sim) == RING_OK, "initialization must succeed");
    CHECK(ring_run_all_reduce(&sim) == RING_OK, "AllReduce must succeed");
    CHECK(
        sim.event_count == RING_MAX_TRACE_EVENTS,
        "expected %d events, got %d",
        RING_MAX_TRACE_EVENTS,
        sim.event_count
    );

    for (int index = 0; index < sim.event_count; ++index) {
        const TraceEvent *event = &sim.events[index];
        CHECK(
            event->timestamp == (uint64_t)index,
            "logical timestamps must be contiguous at index=%d",
            index
        );
        CHECK(event->op_seq == 0, "op_seq must be fixed to zero");
        CHECK(event->channel == 0, "channel must be fixed to zero");
        CHECK(
            event->rank >= 0 && event->rank < RING_RANKS,
            "event rank must be valid"
        );
        CHECK(
            event->peer >= 0 && event->peer < RING_RANKS,
            "event peer must be valid"
        );
    }
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
    run_test("AllGather schedule", test_all_gather_schedule, &failures);
    run_test("AllGather step order", test_all_gather_step_order, &failures);
    run_test("final AllReduce result", test_final_all_reduce_result, &failures);
    run_test("complete event log", test_complete_event_log, &failures);

    if (failures != 0) {
        fprintf(stderr, "%d test group(s) failed\n", failures);
        return 1;
    }
    return 0;
}
