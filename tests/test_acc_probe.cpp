// ggml_acc 语义探针: band 写入 dst 的 [1, L] 位置 — 记录 rel-pos 调试时的关键发现。
//
// 结论 (本仓库 llama.cpp 的 CPU acc kernel):
//   ggml_acc(dst, b, nb1, nb2, nb3, offset) 把 b 的每一行 (行内 ne0 个元素连续)
//   写到 dst 的 [offset + i1*nb1] 处 (i1 = b 的行号)。b 是 [1, L] 时 nr=L, nc=1,
//   即每行只写 1 个元素 —— 位置步进完全由 nb1 决定。
//   想在对角带上写 [b+j, j] (ggml [ki, tq] 布局, ne0=ki): 取 nb1=(T+1)*4, offset=b*4。
//   最初误传 nb1=T*4 → band 被写到连续位置 (sc 差 1.44 的乱结果即源于此;
//   最终实现改用 concat/pad+reshape 复刻 torch _rel_to_abs, 见 gsv_encp.cpp 头注释)。
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>

int main() {
    const int T = 6;
    ggml_init_params ip = { ggml_tensor_overhead() * 256, NULL, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, T);
    const float z[T * T] = { 0 };
    memcpy(a->data, z, sizeof(z));

    // b=+1 的 band: [1, L=T-1], 元素 1000+j
    ggml_tensor * band = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, T - 1);
    const float binit[5] = { 1000, 1001, 1002, 1003, 1004 };
    memcpy(band->data, binit, sizeof(binit));

    // 对角带: nb1=(T+1)*4 (每行步进一列+一元素), offset=1*4 (b=+1)
    ggml_tensor * out = ggml_acc(c, a, band, (size_t)((T + 1) * 4), 0, 0, 1 * 4);
    ggml_cgraph * g = ggml_new_graph(c);
    ggml_build_forward_expand(g, out);
    ggml_graph_compute_with_ctx(c, g, 1);
    printf("acc 对角线 band (期望 out[1+j, j] = 1000+j):\n");
    int bad = 0;
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            const float v = ((float *) out->data)[i + j * T];
            printf("%7.0f", v);
            if (i > 0 && i == j + 1 && i < T) { if (v != 1000 + j) bad++; }
        }
        printf("\n");
    }
    printf(bad ? "ACC PROBE FAIL\n" : "ACC PROBE OK (对角线写入语义符合预期)\n");
    ggml_free(c);
    return bad ? 1 : 0;
}
