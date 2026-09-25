#include "trace_jsonl.h"

#include <inttypes.h>

static const char *phase_name(TracePhase phase) {
    switch (phase) {
        case TRACE_PHASE_REDUCE_SCATTER:
            return "reduce_scatter";
        case TRACE_PHASE_ALL_GATHER:
            return "all_gather";
    }
    return NULL;
}

static const char *action_name(TraceAction action) {
    switch (action) {
        case TRACE_ACTION_SEND:
            return "send";
        case TRACE_ACTION_RECV:
            return "recv";
    }
    return NULL;
}

int trace_event_write_jsonl(FILE *stream, const TraceEvent *event) {
    if (stream == NULL || event == NULL) {
        return -1;
    }

    const char *phase = phase_name(event->phase);
    const char *action = action_name(event->action);
    if (phase == NULL || action == NULL) {
        return -1;
    }

    const int written = fprintf(
        stream,
        "{\"op_seq\":%" PRIu64 ",\"rank\":%d,\"channel\":%d,"
        "\"phase\":\"%s\",\"step\":%d,\"chunk\":%d,"
        "\"action\":\"%s\",\"peer\":%d,\"timestamp\":%" PRIu64 ","
        "\"value\":%d,\"contributor_mask\":%" PRIu32 "}\n",
        event->op_seq,
        event->rank,
        event->channel,
        phase,
        event->step,
        event->chunk_id,
        action,
        event->peer,
        event->timestamp,
        event->value,
        event->contributor_mask
    );
    return written < 0 ? -1 : 0;
}

int trace_events_write_jsonl(
    FILE *stream,
    const TraceEvent *events,
    size_t event_count
) {
    if (stream == NULL || (events == NULL && event_count != 0)) {
        return -1;
    }

    for (size_t index = 0; index < event_count; ++index) {
        if (trace_event_write_jsonl(stream, &events[index]) != 0) {
            return -1;
        }
    }
    return 0;
}
