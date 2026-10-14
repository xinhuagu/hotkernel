"""Candidate for the "decode_attn" target: the file the kernel-speedrun loop edits.

It starts as the baseline itself, so the first verify() is a fair "inconclusive" and every
later verdict measures only what changed. Define decode_attn_candidate(x, k, v) and leave the
run() call alone: inputs, seed, reference, tolerances, rotation and timer are the baseline's,
and verify() reports `invalid` if the contract changes. Triton and torch.compile are fair game.
"""

import torch
import triton
import triton.language as tl

from decode_attn_baseline import baseline_fn, run  # noqa: F401

NSPLIT = 128
BLOCK_N = 64
WARPS = 4
STAGES = 3
COMBINE_BLOCK_D = 32


@triton.jit
def _split_kernel(Q, K, V, O_part, M_part, L_part, sm_scale,
                  stride_kh, stride_vh,
                  G: tl.constexpr, D: tl.constexpr, BLOCK_G: tl.constexpr,
                  BLOCK_N: tl.constexpr, CHUNK: tl.constexpr, NSPLIT: tl.constexpr,
                  STAGES: tl.constexpr):
    kvh = tl.program_id(0)
    s = tl.program_id(1)
    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, D)
    gmask = offs_g < G
    q = tl.load(Q + (kvh * G + offs_g)[:, None] * D + offs_d[None, :], mask=gmask[:, None], other=0.0)
    m_i = tl.full([BLOCK_G], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_G], tl.float32)
    acc = tl.zeros([BLOCK_G, D], tl.float32)
    start = s * CHUNK
    k_base = K + kvh * stride_kh
    v_base = V + kvh * stride_vh
    for n in tl.range(start, start + CHUNK, BLOCK_N, num_stages=STAGES):
        offs_n = n + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + offs_n[:, None] * D + offs_d[None, :])
        v = tl.load(v_base + offs_n[:, None] * D + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    row = (kvh * G + offs_g) * NSPLIT + s
    tl.store(M_part + row, m_i, mask=gmask)
    tl.store(L_part + row, l_i, mask=gmask)
    tl.store(O_part + row[:, None] * D + offs_d[None, :], acc, mask=gmask[:, None])


@triton.jit
def _combine_kernel(O_part, M_part, L_part, Out, D: tl.constexpr, NSPLIT: tl.constexpr,
                    BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    offs_s = tl.arange(0, NSPLIT)
    offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    m = tl.load(M_part + h * NSPLIT + offs_s)
    l = tl.load(L_part + h * NSPLIT + offs_s)
    acc = tl.load(O_part + (h * NSPLIT + offs_s)[:, None] * D + offs_d[None, :])
    w = tl.exp(m - tl.max(m, 0))
    out = tl.sum(acc * w[:, None], 0) / tl.sum(l * w, 0)
    tl.store(Out + h * D + offs_d, out.to(Out.dtype.element_ty))


_scratch = {}


def _buffers(device, hq, d):
    key = (device, hq, d)
    buf = _scratch.get(key)
    if buf is None:
        buf = (torch.empty(hq * NSPLIT, d, device=device, dtype=torch.float32),
               torch.empty(hq * NSPLIT, device=device, dtype=torch.float32),
               torch.empty(hq * NSPLIT, device=device, dtype=torch.float32))
        _scratch[key] = buf
    return buf


_compiled = {}


def decode_attn_candidate(x, k, v):
    B, HQ, _, D = x.shape
    HKV, N = k.shape[1], k.shape[2]
    o_part, m_part, l_part = _buffers(x.device, HQ, D)
    out = torch.empty_like(x)
    key = (HQ, HKV, N, D, k.stride(1), v.stride(1))
    ck = _compiled.get(key)
    if ck is None:
        # First call goes through the JIT; later calls launch the compiled kernels directly,
        # skipping the per-call Python argument binding (host overhead matters at ~80 us/call).
        split = _split_kernel[(HKV, NSPLIT)](x, k, v, o_part, m_part, l_part, D ** -0.5,
                                             k.stride(1), v.stride(1),
                                             G=HQ // HKV, D=D, BLOCK_G=16, BLOCK_N=BLOCK_N,
                                             CHUNK=N // NSPLIT, NSPLIT=NSPLIT, STAGES=STAGES,
                                             num_warps=WARPS)
        comb = _combine_kernel[(HQ, D // COMBINE_BLOCK_D)](o_part, m_part, l_part, out, D=D,
                                                           NSPLIT=NSPLIT, BLOCK_D=COMBINE_BLOCK_D,
                                                           num_warps=4)
        _compiled[key] = (split[(HKV, NSPLIT, 1)], comb[(HQ, D // COMBINE_BLOCK_D, 1)])
        return out
    split_run, comb_run = ck
    split_run(x, k, v, o_part, m_part, l_part, D ** -0.5, k.stride(1), v.stride(1),
              HQ // HKV, D, 16, BLOCK_N, N // NSPLIT, NSPLIT, STAGES)
    comb_run(o_part, m_part, l_part, out, D, NSPLIT, COMBINE_BLOCK_D)
    return out


if __name__ == "__main__":
    run(decode_attn_candidate, impl="agent candidate",
        extra={"config": f"triton split-KV NSPLIT={NSPLIT},BLOCK_N={BLOCK_N},warps={WARPS},stages={STAGES},"
                         f"combine BLOCK_D={COMBINE_BLOCK_D}, cached scratch"})
