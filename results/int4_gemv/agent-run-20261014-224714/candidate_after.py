"""Blank candidate for the int4_gemv target: the file the agent edits in the kernel-speedrun
loop.

It starts as the baseline's own call (torch._weight_int4pack_mm), so the first verify() is a
fair "inconclusive" and every later verdict measures only what the agent changed. Define
int4_gemv_candidate(x, w_q, scales, zeros, w_packed, scales_and_zeros) and leave the run()
call alone: inputs, quantization, seed, reference, tolerances, rotation and timer are the
baseline's, and verify() reports `invalid` if the contract changes.

Either weight layout may be read, never both per call: w_q is (n, k/2) uint8 with the even
column in the high nibble, scales and zeros are (n, k/128) bf16 and dequant(q) = (q - 8) *
scale + zero; w_packed and scales_and_zeros are the same codes in tinygemm's layout. Triton
and torch.compile are fair game.
"""

import sys

import torch
import triton
import triton.language as tl

from int4_gemv_baseline import baseline_fn, run  # noqa: F401

BLOCK_N, BLOCK_KB, NUM_WARPS = 4, 512, 2  # rows per program, code bytes per step (8 groups of 64 B)


@triton.jit
def _int4_gemv_kernel(x32_ptr, wq_ptr, s_ptr, z_ptr, y_ptr, N, KB: tl.constexpr, NG: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr):
    # One program: BLOCK_N rows of W. Per group of 128 values (64 code bytes):
    # sum_j ((q_j - 8) s + z) x_j = s * (sum_j q_j x_j - 8 xs) + z * xs, xs = sum_j x_j.
    # Tiles are loaded as (rows, groups, 64 bytes) so the per-group sum is a reduction along
    # the fastest axis (in-thread + warp shuffles), not a reshape through shared memory.
    # x32 is x viewed as int32: word j holds x[2j] (low half) and x[2j+1] (high half), the
    # two values whose codes share byte j of w_q (high nibble, low nibble).
    pid = tl.program_id(0)
    GPB: tl.constexpr = BLOCK_KB // 64  # groups per step
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    r3 = rows[:, None, None]
    g3 = tl.arange(0, GPB)[None, :, None]
    c3 = tl.arange(0, 64)[None, None, :]
    g2 = tl.arange(0, GPB)[None, :]
    acc = tl.zeros((BLOCK_N, GPB), dtype=tl.float32)
    for kb in range(0, KB, BLOCK_KB):
        b = tl.load(wq_ptr + r3 * KB + kb + g3 * 64 + c3).to(tl.int32)
        x32 = tl.load(x32_ptr + kb + g3 * 64 + c3)
        xe = (x32 << 16).to(tl.float32, bitcast=True)
        xo = (x32 & -65536).to(tl.float32, bitcast=True)
        hi = (b >> 4).to(tl.float32)
        lo = (b & 15).to(tl.float32)
        gq = tl.sum(hi * xe + lo * xo, axis=2)
        xs = tl.sum(xe + xo, axis=2)
        g0 = kb // 64
        s = tl.load(s_ptr + rows[:, None] * NG + g0 + g2).to(tl.float32)
        z = tl.load(z_ptr + rows[:, None] * NG + g0 + g2).to(tl.float32)
        acc += s * (gq - 8.0 * xs) + z * xs
    y = tl.sum(acc, axis=1)
    tl.store(y_ptr + rows, y.to(tl.bfloat16))


if len(sys.argv) > 1:  # sweep hook for bench(): "BLOCK_N,BLOCK_KB,warps[,fast]"
    _a = sys.argv[1].split(",")
    BLOCK_N, BLOCK_KB, NUM_WARPS = int(_a[0]), int(_a[1]), int(_a[2])
    FAST = len(_a) < 4 or _a[3] != "0"
else:
    FAST = True

_compiled = {}


def int4_gemv_candidate(x, w_q, scales, zeros, w_packed, scales_and_zeros):
    n, kb = w_q.shape
    ng = scales.shape[1]
    y = torch.empty(n, device=x.device, dtype=x.dtype)
    x32 = x.view(torch.int32)  # bf16 pairs (x[2j], x[2j+1]) as one word
    grid = (n // BLOCK_N,)
    c = _compiled.get((n, kb, ng)) if FAST else None
    if c is not None:
        # Launch the already-compiled kernel straight through Triton's C launcher: skips the
        # JIT's per-call argument binding, cache lookup and stream query. The timer measures
        # CPU dispatch gaps too, and here they are comparable to the ~30 us kernel.
        ck, stream = c
        ck.run(grid[0], 1, 1, stream, ck.function, ck.packed_metadata, None, None, None,
               x32, w_q, scales, zeros, y, n, kb, ng, BLOCK_N, BLOCK_KB)
        return y
    ck = _int4_gemv_kernel[grid](x32, w_q, scales, zeros, y, n, KB=kb, NG=ng,
                                 BLOCK_N=BLOCK_N, BLOCK_KB=BLOCK_KB, num_warps=NUM_WARPS)
    ck._init_handles()
    _compiled[(n, kb, ng)] = (ck, torch.cuda.current_stream().cuda_stream)
    return y


if __name__ == "__main__":
    run(int4_gemv_candidate, impl="agent candidate",
        extra={"config": f"BLOCK_N={BLOCK_N},BLOCK_KB={BLOCK_KB},warps={NUM_WARPS}"})
