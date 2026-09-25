#include "ring_sim.h"

#include <inttypes.h>
#include <stdio.h>

int main(void) {
    RingSim sim;
    RingStatus status = ring_sim_init(&sim);
    if (status != RING_OK) {
        fprintf(stderr, "ring_sim_init failed: %d\n", status);
        return 1;
    }

    status = ring_run_reduce_scatter(&sim);
    if (status != RING_OK) {
        fprintf(stderr, "ring_run_reduce_scatter failed: %d\n", status);
        return 1;
    }

    for (int rank = 0; rank < RING_RANKS; ++rank) {
        for (int lane = 0; lane < RING_CHUNKS_PER_RANK; ++lane) {
            const int chunk = lane * RING_RANKS + rank;
            const ChunkState *owned = &sim.chunks[rank][chunk];
            printf(
                "rank=%d owner_chunk=%d value=%d contributors=0x%02" PRIx32 "\n",
                rank,
                chunk,
                owned->value,
                owned->contributor_mask
            );
        }
    }

    return 0;
}
