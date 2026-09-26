#include "delay.h"
#include "ring_sim.h"
#include "trace_jsonl.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s [--jsonl] [--inject-delay PHASE STEP RANK CHUNK ACTION TICKS]\n",
        program
    );
}

static bool parse_int(const char *text, int *value) {
    char *end = NULL;
    errno = 0;
    const long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < INT32_MIN || parsed > INT32_MAX) {
        return false;
    }
    *value = (int)parsed;
    return true;
}

static bool parse_ticks(const char *text, uint64_t *value) {
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) {
        return false;
    }
    *value = (uint64_t)parsed;
    return true;
}

static bool parse_phase(const char *text, TracePhase *phase) {
    if (strcmp(text, "reduce_scatter") == 0) {
        *phase = TRACE_PHASE_REDUCE_SCATTER;
        return true;
    }
    if (strcmp(text, "all_gather") == 0) {
        *phase = TRACE_PHASE_ALL_GATHER;
        return true;
    }
    return false;
}

static bool parse_action(const char *text, TraceAction *action) {
    if (strcmp(text, "send") == 0) {
        *action = TRACE_ACTION_SEND;
        return true;
    }
    if (strcmp(text, "recv") == 0) {
        *action = TRACE_ACTION_RECV;
        return true;
    }
    return false;
}

int main(int argc, char **argv) {
    bool jsonl = false;
    bool inject_delay = false;
    DelayConfig delay = {0};

    for (int index = 1; index < argc;) {
        if (strcmp(argv[index], "--jsonl") == 0) {
            jsonl = true;
            index += 1;
            continue;
        }
        if (strcmp(argv[index], "--inject-delay") == 0 &&
            index + 6 < argc && !inject_delay) {
            if (!parse_phase(argv[index + 1], &delay.phase) ||
                !parse_int(argv[index + 2], &delay.step) ||
                !parse_int(argv[index + 3], &delay.rank) ||
                !parse_int(argv[index + 4], &delay.chunk_id) ||
                !parse_action(argv[index + 5], &delay.action) ||
                !parse_ticks(argv[index + 6], &delay.delay_ticks)) {
                print_usage(argv[0]);
                return 2;
            }
            inject_delay = true;
            index += 7;
            continue;
        }

        print_usage(argv[0]);
        return 2;
    }

    RingSim sim;
    RingStatus status = ring_sim_init(&sim);
    if (status != RING_OK) {
        fprintf(stderr, "ring_sim_init failed: %d\n", status);
        return 1;
    }

    status = ring_run_all_reduce(&sim);
    if (status != RING_OK) {
        fprintf(stderr, "ring_run_all_reduce failed: %d\n", status);
        return 1;
    }

    if (inject_delay) {
        DelayReport report;
        status = ring_apply_causal_delay(&sim, &delay, &report);
        if (status != RING_OK) {
            fprintf(stderr, "ring_apply_causal_delay failed: %d\n", status);
            return 1;
        }
        fprintf(
            stderr,
            "delay root_event=%d affected_events=%d affected_ranks=0x%02" PRIx32
            "\n",
            report.root_event_index,
            report.affected_event_count,
            report.affected_rank_mask
        );
    }

    if (jsonl) {
        if (trace_events_write_jsonl(stdout, sim.events, sim.event_count) != 0) {
            fprintf(stderr, "failed to write JSONL trace\n");
            return 1;
        }
        return 0;
    }

    for (int rank = 0; rank < RING_RANKS; ++rank) {
        for (int chunk = 0; chunk < RING_CHUNKS; ++chunk) {
            const ChunkState *state = &sim.chunks[rank][chunk];
            printf(
                "rank=%d chunk=%d value=%d contributors=0x%02" PRIx32 "\n",
                rank,
                chunk,
                state->value,
                state->contributor_mask
            );
        }
    }

    return 0;
}
