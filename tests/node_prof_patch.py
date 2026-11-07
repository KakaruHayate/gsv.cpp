#!/usr/bin/env python3
# Add per-op timing to ggml-cpu.c (local measurement patch for AR latency work).
# Idempotent. Not intended for upstream: guarded by GGML_NODE_PROF env var.
import sys

p = sys.argv[1]
src = open(p).read()
if "GGML_NODE_PROF" in src:
    print("already patched"); sys.exit(0)

HELPERS = r'''
// ---------------------------------------------------------------- GGML_NODE_PROF
// Local measurement patch (ar-opt / task-4): per-op wall time on thread 0, printed at exit.
// Guarded by GGML_NODE_PROF=1. Barrier gaps (thread 0 idle between ops) are attributed
// to "BARRIER" so the table sums to the graph compute time.
#include <time.h>
#include <stdlib.h>
static int    np_prof = -1;
static double np_ms[GGML_OP_COUNT];
static long   np_cnt[GGML_OP_COUNT];
static double np_ms_barrier;
static double np_last_end;
static int    np_have_first;

static double np_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}
static void np_report(void) {
    double tot = np_ms_barrier;
    for (int i = 0; i < GGML_OP_COUNT; i++) tot += np_ms[i];
    fprintf(stderr, "[node_prof] nodes_total=%.1f ms  barrier=%.1f ms (%.1f%%)\n",
            tot, np_ms_barrier, tot > 0 ? 100.0 * np_ms_barrier / tot : 0.0);
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        if (np_cnt[i] == 0) continue;
        fprintf(stderr, "[node_prof] %-22s n=%-6ld total=%8.2f ms  %5.1f%%  %7.1f us/op\n",
                ggml_op_name((enum ggml_op) i), np_cnt[i], np_ms[i],
                tot > 0 ? 100.0 * np_ms[i] / tot : 0.0, 1000.0 * np_ms[i] / (double) np_cnt[i]);
    }
}
static void np_init(void) {
    if (np_prof >= 0) return;
    np_prof = getenv("GGML_NODE_PROF") != NULL ? 1 : 0;
    if (np_prof) atexit(np_report);
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
'''

ANCHOR1 = "static thread_ret_t ggml_graph_compute_thread(void * data) {\n"
assert src.count(ANCHOR1) == 1
src = src.replace(ANCHOR1, HELPERS.lstrip("\n"))

A2 = """        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
"""
assert src.count(A2) == 1
src = src.replace(A2, A2 + """
        np_init();
        const double np_t0 = np_prof ? np_now() : 0.0;
""")

A3 = """        if (state->ith == 0 && cplan->abort_callback &&
"""
assert src.count(A3) == 1
src = src.replace(A3, """        if (np_prof && state->ith == 0) {
            const double np_t1 = np_now();
            np_ms[node->op] += np_t1 - np_t0;
            np_cnt[node->op]++;
            if (np_have_first) np_ms_barrier += np_t0 - np_last_end;
            else np_have_first = 1;
            np_last_end = np_t1;
        }

""" + A3)

open(p, "w").write(src)
print("patched", p)
