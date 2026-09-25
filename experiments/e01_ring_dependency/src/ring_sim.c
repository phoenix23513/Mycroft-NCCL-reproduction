#include "ring_sim.h"

#include <stddef.h>
#include <string.h>

int ring_next_rank(int rank) {
    int next_rank = (rank + 1) % RING_RANKS;
    return next_rank;
}

int ring_prev_rank(int rank) {
    int prev_rank = (rank - 1 + RING_RANKS) % RING_RANKS;
    return prev_rank;
}

int ring_send_chunk(int rank, int step, int lane) {
    int chunk_base = lane * RING_RANKS;
    int chunk_offset =
        (rank - step - 1 + RING_RANKS) % RING_RANKS;
    int send_chunk = chunk_base + chunk_offset;
    return send_chunk;
}

int ring_recv_chunk(int rank, int step, int lane) {
    int chunk_base = lane * RING_RANKS;
    int chunk_offset =
        (rank - step - 2 + RING_RANKS) % RING_RANKS;
    int recv_chunk = chunk_base + chunk_offset;
    return recv_chunk;
}

RingStatus ring_sim_init(RingSim *sim) {
    if (sim == NULL) {
        return RING_ERROR_INVALID_ARGUMENT;
    }

    memset(sim, 0, sizeof(*sim));

    for (int rank = 0; rank < RING_RANKS; ++rank) {
        for (int chunk = 0; chunk < RING_CHUNKS; ++chunk) {
            sim->chunks[rank][chunk].value = rank * 10 + chunk + 1;
            sim->chunks[rank][chunk].contributor_mask =
                UINT32_C(1) << rank;
        }
    }

    return RING_OK;
}

RingStatus ring_reduce_scatter_step(RingSim *sim, int step) {
    if (sim == NULL || step < 0 || step >= RING_STEPS) {
        return RING_ERROR_INVALID_ARGUMENT;
    }
    if (step != sim->completed_steps) {
        return RING_ERROR_INVALID_ARGUMENT;
    }

    RingMessage messages[RING_RANKS][RING_CHUNKS_PER_RANK];

    for (int src_rank = 0; src_rank < RING_RANKS; ++src_rank) {
        for (int lane = 0; lane < RING_CHUNKS_PER_RANK; ++lane) {
            int chunk = ring_send_chunk(src_rank, step, lane);
            RingMessage *message = &messages[src_rank][lane];

            message->src_rank = src_rank;
            message->dst_rank = ring_next_rank(src_rank);
            message->chunk_id = chunk;
            message->payload = sim->chunks[src_rank][chunk];
        }
    }

    for (int src_rank = 0; src_rank < RING_RANKS; ++src_rank) {
        for (int lane = 0; lane < RING_CHUNKS_PER_RANK; ++lane) {
            const RingMessage *message = &messages[src_rank][lane];
            ChunkState *destination =
                &sim->chunks[message->dst_rank][message->chunk_id];

            destination->value += message->payload.value;
            destination->contributor_mask |=
                message->payload.contributor_mask;

            TraceEvent *event =
                &sim->history[step][message->dst_rank][lane];
            event->step = step;
            event->src_rank = message->src_rank;
            event->dst_rank = message->dst_rank;
            event->chunk_id = message->chunk_id;
            event->value = destination->value;
            event->contributor_mask = destination->contributor_mask;
        }
    }

    sim->completed_steps += 1;
    return RING_OK;
}

RingStatus ring_run_reduce_scatter(RingSim *sim) {
    if (sim == NULL) {
        return RING_ERROR_INVALID_ARGUMENT;
    }

    for (int step = 0; step < RING_STEPS; ++step) {
        RingStatus status = ring_reduce_scatter_step(sim, step);
        if (status != RING_OK) {
            return status;
        }
    }

    return RING_OK;
}
