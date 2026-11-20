#pragma once

#include <stdint.h>

/* Keep in sync with graph_fold.py (every field is 8 bytes). */
#ifdef __cplusplus
extern "C" {
#endif

struct GfReport {
    int64_t status;                   /* 0 ok, 1 refused (message says why) */
    int64_t n_nodes;                  /* nodes of the captured graph */
    int64_t n_kernels;
    int64_t n_edges;
    int64_t n_event_record;
    int64_t n_event_wait;
    int64_t n_nondefault_edges;       /* programmatic edges, turned into full edges */
    int64_t n_classes;                /* distinct kernel launch classes */
    int64_t region_start;             /* kernel index where the loop starts */
    int64_t period;                   /* kernels per iteration */
    int64_t iters;                    /* loop trip count */
    int64_t prefix_kernels;
    int64_t suffix_kernels;
    int64_t loop_events;              /* EVENT_RECORD nodes dropped from the loop */
    int64_t in_edges;                 /* prefix nodes the WHILE waits for */
    int64_t out_edges;                /* suffix nodes that wait for the WHILE */
    int64_t cross_edges;              /* iteration i -> i+1 edges, replaced by the barrier */
    int64_t body_edges;
    int64_t n_updates;                /* SetParam spans per iteration */
    int64_t row_bytes;                /* table bytes per iteration */
    int64_t table_bytes;
    int64_t unrolled_nodes;
    int64_t folded_nodes;             /* predicted */
    int64_t folded_nodes_actual;      /* parent graph nodes (WHILE counts as one) */
    int64_t body_nodes;
    int64_t instantiate_driver_delta; /* cuMemGetInfo delta around instantiate */
    int64_t pdl_attr_dropped;         /* kernels whose PDL launch attribute was dropped */
    int64_t ctx_mode;                 /* 0 current, 1 stream ctx pushed, 2 node ctx pushed, 3 none */
    char message[512];
};

struct GfHandle;

/* dump_dir (nullable): write graph_before.json, fold_analysis.json and, unless dry_run,
 * graph_after_parent.json / graph_after_body.json into that directory. The directory must
 * exist. A dump never changes the fold result; dump failures are reported in rep->message
 * of the fold only when the fold itself fails. */
int gf_fold(void* cugraph, void* stream, int min_iters, int min_saved_nodes, int dry_run,
            const char* dump_dir, struct GfReport* rep, struct GfHandle** out);
/* Dump any (non-conditional-recursive) graph as JSON; returns 0 on success. */
int gf_dump_graph(void* cugraph, const char* path, const char* label);
int gf_launch(struct GfHandle* h, void* stream);
int gf_state(struct GfHandle* h, int64_t* out3);
void gf_destroy(struct GfHandle* h);

#ifdef __cplusplus
}
#endif
