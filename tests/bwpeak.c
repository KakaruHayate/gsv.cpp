// Roofline probe for the 4-vCPU box: achievable memory bandwidth + AVX2/FMA peak.
//
//   cc -O3 -mavx2 -mfma bwpeak.c -o bwpeak -lpthread
//   ./bwpeak [threads] [mb]
//
// Reports, per thread count:
//   copy   : dst[i] = src[i]              (1 read + 1 write per 4 B)
//   scale  : dst[i] = 3.0f * src[i]       (same traffic, no reuse)
//   triad  : dst[i] = a[i] + 3.0f*b[i]    (2 reads + 1 write)
//   dot    : sum += a[i] * b[i]           (2 reads, 1 flop/FMA, pure streaming)
// and a register-blocked FMA loop for the compute peak (no memory traffic).
//
// Everything runs on buffers >= 8x L3 so the numbers are DRAM, not cache.
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static float * alloc_f32(size_t n) {
    float * p = NULL;
    if (posix_memalign((void **) &p, 64, n * sizeof(float))) { perror("posix_memalign"); exit(1); }
    for (size_t i = 0; i < n; i++) p[i] = 1.0f + (float) (i & 1023) * 1e-6f;
    return p;
}

int main(int argc, char ** argv) {
    int tmax = argc > 1 ? atoi(argv[1]) : 4;
    int mb   = argc > 2 ? atoi(argv[2]) : 512;   // per buffer

    const size_t n = (size_t) mb * 1024 * 1024 / 4;
    const size_t bytes = (size_t) mb * 1024 * 1024;
    float * a = alloc_f32(n), * b = alloc_f32(n), * c = alloc_f32(n);
    printf("buffers: 3 x %d MB (n=%zu floats each)\n", mb, n);
    printf("%-6s %-8s %10s %10s %10s\n", "thr", "kernel", "GB/s", "GFLOP/s", "ms");

    for (int t = 1; t <= tmax; t++) {
        omp_set_num_threads(t);
        double t0, t1;
        volatile float sink = 0;

        // copy
        t0 = now_s();
        #pragma omp parallel for
        for (long i = 0; i < (long) n; i += 8) {
            __m256 v = _mm256_load_ps(a + i);
            _mm256_store_ps(c + i, v);
        }
        t1 = now_s();
        printf("%-6d %-8s %10.2f %10s %10.2f\n", t, "copy", 2.0 * bytes / (t1 - t0) / 1e9, "-", (t1 - t0) * 1e3);

        // scale
        const __m256 s3 = _mm256_set1_ps(3.0f);
        t0 = now_s();
        #pragma omp parallel for
        for (long i = 0; i < (long) n; i += 8) {
            __m256 v = _mm256_mul_ps(_mm256_load_ps(a + i), s3);
            _mm256_store_ps(c + i, v);
        }
        t1 = now_s();
        printf("%-6d %-8s %10.2f %10s %10.2f\n", t, "scale", 2.0 * bytes / (t1 - t0) / 1e9, "-", (t1 - t0) * 1e3);

        // triad
        t0 = now_s();
        #pragma omp parallel for
        for (long i = 0; i < (long) n; i += 8) {
            __m256 v = _mm256_fmadd_ps(s3, _mm256_load_ps(b + i), _mm256_load_ps(a + i));
            _mm256_store_ps(c + i, v);
        }
        t1 = now_s();
        printf("%-6d %-8s %10.2f %10.2f %10.2f\n", t, "triad", 3.0 * bytes / (t1 - t0) / 1e9,
               2.0 * n / (t1 - t0) / 1e9, (t1 - t0) * 1e3);

        // dot
        float acc = 0;
        t0 = now_s();
        #pragma omp parallel
        {
            __m256 s = _mm256_setzero_ps();
            #pragma omp for
            for (long i = 0; i < (long) n; i += 8)
                s = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), s);
            float tmp[8];
            _mm256_storeu_ps(tmp, s);
            float ls = 0;
            for (int k = 0; k < 8; k++) ls += tmp[k];
            #pragma omp atomic
            acc += ls;
        }
        sink = acc;
        t1 = now_s();
        printf("%-6d %-8s %10.2f %10.2f %10.2f\n", t, "dot", 2.0 * bytes / (t1 - t0) / 1e9,
               2.0 * n / (t1 - t0) / 1e9, (t1 - t0) * 1e3);

        // FMA peak: 8 independent accumulators in registers, 64 MB working set stays in L3/L2
        const long iters = 200000;
        __m256 x0 = _mm256_set1_ps(1.0000001f), x1 = x0, x2 = x0, x3 = x0;
        __m256 x4 = x0, x5 = x0, x6 = x0, x7 = x0;
        const __m256 m = _mm256_set1_ps(0.9999999f);
        t0 = now_s();
        #pragma omp parallel
        {
            __m256 y0 = x0, y1 = x1, y2 = x2, y3 = x3, y4 = x4, y5 = x5, y6 = x6, y7 = x7;
            for (long it = 0; it < iters; it++) {
                y0 = _mm256_fmadd_ps(y0, m, x0);
                y1 = _mm256_fmadd_ps(y1, m, x1);
                y2 = _mm256_fmadd_ps(y2, m, x2);
                y3 = _mm256_fmadd_ps(y3, m, x3);
                y4 = _mm256_fmadd_ps(y4, m, x4);
                y5 = _mm256_fmadd_ps(y5, m, x5);
                y6 = _mm256_fmadd_ps(y6, m, x6);
                y7 = _mm256_fmadd_ps(y7, m, x7);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, _mm256_add_ps(_mm256_add_ps(y0, y1), _mm256_add_ps(y2, y3)));
            float s2[8];
            _mm256_storeu_ps(s2, _mm256_add_ps(_mm256_add_ps(y4, y5), _mm256_add_ps(y6, y7)));
            float ls = 0;
            for (int k = 0; k < 8; k++) ls += tmp[k] + s2[k];
            #pragma omp atomic
            acc += ls;
        }
        sink = acc;
        t1 = now_s();
        const double flops = 2.0 * 8.0 * 8.0 * (double) iters * t;   // 8 lanes * 8 acc * FMA(2) * threads
        printf("%-6d %-8s %10s %10.2f %10.2f\n", t, "fma-peak", "-", flops / (t1 - t0) / 1e9, (t1 - t0) * 1e3);
        if (sink == 12345.678f) printf("unreachable\n");
        printf("\n");
    }
    return 0;
}
