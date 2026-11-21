"""Split-K (flash-decoding) Metal kernel: one query token against an 8-bit
affine (group 64) K/V cache, with Inkling's relative-bias tail and sink logit.

Pass 1 (Metal): each threadgroup takes CHUNK keys of one KV head; each of its
4 simdgroups runs an online softmax over a contiguous quarter and writes a
partial (max, sum, acc[128]) for the REP query heads sharing that KV head.
Each lane owns 4 of the 128 dims, so a key row is one coalesced 128-byte read.
Pass 2 (MLX ops): merge the partials and the sink.
"""
import mlx.core as mx

CHUNK = 1024
SG = 4
D = 128

_SRC = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint chunk = threadgroup_position_in_grid.x;
    const uint h = threadgroup_position_in_grid.y;
    const int n = params[0];
    const int maxlen = params[1];
    const int P = params[2];
    const uint g = lane / 16;

    float qv[REP][4];
    float qs[REP];
    float m[REP];
    float l[REP];
    float acc[REP][4];
    for (int r = 0; r < REP; r++) {
        qs[r] = 0.0f;
        for (int k = 0; k < 4; k++) {
            qv[r][k] = q[(h * REP + r) * 128 + lane * 4 + k];
            qs[r] += qv[r][k];
            acc[r][k] = 0.0f;
        }
        m[r] = -1e30f;
        l[r] = 0.0f;
    }
    const int per = CHUNK / SG;
    const int start = (int)chunk * CHUNK + (int)sg * per;
    const int end = min(start + per, n);
    for (int j = start; j < end; j++) {
        const size_t row = (size_t)h * (size_t)maxlen + (size_t)j;
        const uint kw = kq[row * 32 + lane];
        const uint vw = vq[row * 32 + lane];
        const float ksc = float(ks[row * 2 + g]);
        const float kbi = float(kb[row * 2 + g]);
        const float vsc = float(vs[row * 2 + g]);
        const float vbi = float(vb[row * 2 + g]);
        const float c0 = float(kw & 0xffu), c1 = float((kw >> 8) & 0xffu);
        const float c2 = float((kw >> 16) & 0xffu), c3 = float(kw >> 24);
        float v[4];
        v[0] = float(vw & 0xffu) * vsc + vbi;
        v[1] = float((vw >> 8) & 0xffu) * vsc + vbi;
        v[2] = float((vw >> 16) & 0xffu) * vsc + vbi;
        v[3] = float(vw >> 24) * vsc + vbi;
        const int d = n - 1 - j;
        for (int r = 0; r < REP; r++) {
            float part = ksc * (qv[r][0] * c0 + qv[r][1] * c1 + qv[r][2] * c2 + qv[r][3] * c3)
                         + kbi * qs[r];
            float s = simd_sum(part);
            if (d < RELN) {
                s += float(rel[(h * REP + r) * RELN + d]);
            }
            const float mn = max(m[r], s);
            const float corr = metal::exp(m[r] - mn);
            const float p = metal::exp(s - mn);
            l[r] = l[r] * corr + p;
            for (int k = 0; k < 4; k++) {
                acc[r][k] = acc[r][k] * corr + p * v[k];
            }
            m[r] = mn;
        }
    }
    const size_t pidx = (size_t)chunk * SG + sg;
    for (int r = 0; r < REP; r++) {
        const size_t o = ((size_t)h * (size_t)P + pidx) * REP + r;
        if (lane == 0) {
            pm[o] = m[r];
            pl[o] = l[r];
        }
        for (int k = 0; k < 4; k++) {
            pacc[o * 128 + lane * 4 + k] = acc[r][k];
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="inkling_q8_decode_attn",
    input_names=["q", "kq", "ks", "kb", "vq", "vs", "vb", "rel", "params"],
    output_names=["pm", "pl", "pacc"],
    source=_SRC,
)


def q8_flash_decode(q, cache, rel, sink, scale):
    """q: [1, hq, 1, 128] bf16; cache: Q8GlobalKVCache holding n keys (after
    append); rel: [1, 1, hq, R]; sink: [1, 1, hq]. Returns [1, hq, 1, 128]."""
    hq = q.shape[1]
    hkv = cache.kq.shape[1]
    rep = hq // hkv
    n = cache.offset
    reln = rel.shape[-1]
    chunks = (n + CHUNK - 1) // CHUNK
    P = chunks * SG
    qf = (q.astype(mx.float32) * scale).reshape(hkv * rep * D)
    relf = rel[0, 0].astype(mx.float32).reshape(hq * reln)
    params = mx.array([n, cache.max_len, P], dtype=mx.int32)
    pm, pl, pacc = _kernel(
        inputs=[qf, cache.kq, cache.ks, cache.kb, cache.vq, cache.vs, cache.vb, relf, params],
        template=[("REP", rep), ("RELN", reln), ("CHUNK", CHUNK), ("SG", SG)],
        grid=(chunks * 32 * SG, hkv, 1),
        threadgroup=(32 * SG, 1, 1),
        output_shapes=[(hkv, P, rep), (hkv, P, rep), (hkv, P, rep, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    sk = sink[0, 0].astype(mx.float32).reshape(hkv, rep)
    M = mx.maximum(pm.max(axis=1), sk)
    w = mx.exp(pm - M[:, None, :])
    den = (w * pl).sum(axis=1) + mx.exp(sk - M)
    num = (w[..., None] * pacc).sum(axis=1)
    out = num / den[..., None]
    return out.reshape(1, hq, 1, D).astype(q.dtype)
