#include "ring_sim.h"
#include "trace_jsonl.h"

#include <stdbool.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const bool jsonl = argc == 2 && strcmp(argv[1], "--jsonl") == 0;
    if (argc > 1 && !jsonl) {
        fprintf(stderr, "usage: %s [--jsonl]\n", argv[0]);
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
