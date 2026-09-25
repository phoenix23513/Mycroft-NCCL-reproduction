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

static bool test_rank_and_chunk_counts_are_distinct(void) {
    CHECK(RING_RANKS == 4, "the fixture requires four ranks");
    CHECK(RING_CHUNKS == 8, "the fixture requires eight chunks");
    CHECK(
        RING_RANKS != RING_CHUNKS,
        "rank and chunk counts must exercise different values"
    );
    CHECK(
        RING_CHUNKS_PER_RANK == 2,
        "each rank must own two completed chunks"
    );
    return true;
}

static bool test_rank_neighbors(void) {
    const int expected_next[RING_RANKS] = {1, 2, 3, 0};
    const int expected_prev[RING_RANKS] = {3, 0, 1, 2};

    for (int rank = 0; rank < RING_RANKS; ++rank) {
        CHECK(
            ring_next_rank(rank) == expected_next[rank],
            "next rank for %d",
            rank
        );
        CHECK(
            ring_prev_rank(rank) == expected_prev[rank],
            "previous rank for %d",
            rank
        );
    }
    return true;
}

static bool test_chunk_schedule(void) {
    const int expected_send[RING_STEPS][RING_CHUNKS_PER_RANK][RING_RANKS] = {
        {{3, 0, 1, 2}, {7, 4, 5, 6}},
        {{2, 3, 0, 1}, {6, 7, 4, 5}},
        {{1, 2, 3, 0}, {5, 6, 7, 4}},
    };

    for (int step = 0; step < RING_STEPS; ++step) {
        for (int lane = 0; lane < RING_CHUNKS_PER_RANK; ++lane) {
            for (int rank = 0; rank < RING_RANKS; ++rank) {
                const int next = (rank + 1) % RING_RANKS;
                CHECK(
                    ring_send_chunk(rank, step, lane) ==
                        expected_send[step][lane][rank],
                    "send chunk for rank=%d step=%d lane=%d",
                    rank,
                    step,
                    lane
                );
                CHECK(
                    ring_recv_chunk(next, step, lane) ==
                        expected_send[step][lane][rank],
                    "send/receive mismatch for rank=%d step=%d lane=%d",
                    rank,
                    step,
                    lane
                );
            }
        }
    }
    return true;
}

static bool test_initial_state(void) {
    RingSim sim;
    CHECK(ring_sim_init(&sim) == RING_OK, "ring_sim_init must succeed");
    CHECK(sim.completed_steps == 0, "completed_steps must start at zero");

    for (int rank = 0; rank < RING_RANKS; ++rank) {
        for (int chunk = 0; chunk < RING_CHUNKS; ++chunk) {
            CHECK(
                sim.chunks[rank][chunk].value == input_value(rank, chunk),
                "initial value for rank=%d chunk=%d",
                rank,
                chunk
            );
            CHECK(
                sim.chunks[rank][chunk].contributor_mask == (UINT32_C(1) << rank),
                "initial mask for rank=%d chunk=%d",
                rank,
                chunk
            );
        }
    }
    return true;
}

static bool test_first_step_updates_both_lanes(void) {
    RingSim sim;

    CHECK(ring_sim_init(&sim) == RING_OK, "ring_sim_init must succeed");
    CHECK(
        ring_reduce_scatter_step(&sim, 0) == RING_OK,
        "step zero must succeed"
    );
    CHECK(sim.completed_steps == 1, "one step must be completed");

    for (int dst_rank = 0; dst_rank < RING_RANKS; ++dst_rank) {
        const int src_rank = ring_prev_rank(dst_rank);
        for (int lane = 0; lane < RING_CHUNKS_PER_RANK; ++lane) {
            const int chunk = ring_recv_chunk(dst_rank, 0, lane);
            const int expected_value =
                input_value(dst_rank, chunk) + input_value(src_rank, chunk);
            const uint32_t expected_mask =
                (UINT32_C(1) << dst_rank) | (UINT32_C(1) << src_rank);
            const TraceEvent *event = &sim.history[0][dst_rank][lane];

            CHECK(
                sim.chunks[dst_rank][chunk].value == expected_value,
                "one-hop value for rank=%d chunk=%d",
                dst_rank,
                chunk
            );
            CHECK(
                sim.chunks[dst_rank][chunk].contributor_mask == expected_mask,
                "one-hop mask for rank=%d chunk=%d",
                dst_rank,
                chunk
            );
            CHECK(
                event->step == 0 && event->src_rank == src_rank &&
                    event->dst_rank == dst_rank && event->chunk_id == chunk &&
                    event->value == expected_value &&
                    event->contributor_mask == expected_mask,
                "history record for destination rank=%d lane=%d",
                dst_rank,
                lane
            );
        }
    }
    return true;
}

static bool test_rejects_out_of_order_or_repeated_steps(void) {
    RingSim sim;

    CHECK(ring_sim_init(&sim) == RING_OK, "ring_sim_init must succeed");
    CHECK(
        ring_reduce_scatter_step(&sim, 1) == RING_ERROR_INVALID_ARGUMENT,
        "step one must not run before step zero"
    );
    CHECK(sim.completed_steps == 0, "a rejected step must not advance state");
    CHECK(
        ring_reduce_scatter_step(&sim, 0) == RING_OK,
        "step zero must succeed once"
    );
    CHECK(
        ring_reduce_scatter_step(&sim, 0) == RING_ERROR_INVALID_ARGUMENT,
        "step zero must not run twice"
    );
    CHECK(sim.completed_steps == 1, "a repeated step must not advance state");
    return true;
}

static bool test_final_reduce_scatter_result(void) {
    RingSim sim;

    CHECK(ring_sim_init(&sim) == RING_OK, "ring_sim_init must succeed");
    CHECK(
        ring_run_reduce_scatter(&sim) == RING_OK,
        "ring_run_reduce_scatter must succeed"
    );
    CHECK(
        sim.completed_steps == RING_STEPS,
        "exactly %d steps must complete",
        RING_STEPS
    );

    for (int rank = 0; rank < RING_RANKS; ++rank) {
        for (int lane = 0; lane < RING_CHUNKS_PER_RANK; ++lane) {
            const int chunk = lane * RING_RANKS + rank;
            const ChunkState *owned = &sim.chunks[rank][chunk];
            CHECK(
                owned->value == reduced_value(chunk),
                "final value for rank=%d chunk=%d",
                rank,
                chunk
            );
            CHECK(
                owned->contributor_mask == UINT32_C(0x0f),
                "final contributor mask for rank=%d chunk=%d",
                rank,
                chunk
            );
        }
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

    run_test(
        "rank and chunk counts are distinct",
        test_rank_and_chunk_counts_are_distinct,
        &failures
    );
    run_test("rank neighbors", test_rank_neighbors, &failures);
    run_test("two-lane chunk schedule", test_chunk_schedule, &failures);
    run_test("initial state", test_initial_state, &failures);
    run_test(
        "first step updates both lanes",
        test_first_step_updates_both_lanes,
        &failures
    );
    run_test(
        "rejects out-of-order or repeated steps",
        test_rejects_out_of_order_or_repeated_steps,
        &failures
    );
    run_test(
        "final reduce-scatter result",
        test_final_reduce_scatter_result,
        &failures
    );

    if (failures != 0) {
        fprintf(stderr, "%d test group(s) failed\n", failures);
        return 1;
    }

    return 0;
}
