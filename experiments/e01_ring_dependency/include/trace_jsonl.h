#ifndef MYCROFT_E01_TRACE_JSONL_H
#define MYCROFT_E01_TRACE_JSONL_H

#include <stddef.h>
#include <stdio.h>

#include "trace_event.h"

int trace_event_write_jsonl(FILE *stream, const TraceEvent *event);
int trace_events_write_jsonl(
    FILE *stream,
    const TraceEvent *events,
    size_t event_count
);

#endif

