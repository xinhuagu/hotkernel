"""Candidate for the rmsnorm target: a fused single-pass Triton kernel (hand-written during
tool bring-up, 2026-10-14; the autonomous loop later converged on the same design).

This is the kind of file the agent edits: it defines one function with the baseline's
signature and hands it to the baseline's run(), which owns the shape, dtype, reference,
tolerances, rotation and timer. Baseline (plain eager bf16): x.pow(2) -> mean -> rsqrt -> mul
-> mul, five kernels and at least three passes over the 64 MiB tensor. Candidate: one program
per row loads the row once (bf16 -> fp32 in registers), reduces, scales, and writes it once.
"""

import torch
import triton
import triton.language as tl
from rmsnorm_baseline import EPS, run

NUM_WARPS = 8


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, n_cols, row_stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * row_stride + offs, (x * inv * w).to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm_candidate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    rmsnorm_fwd_kernel[(x.shape[0],)](x, w, y, x.shape[1], x.stride(0), EPS,
                                       BLOCK=triton.next_power_of_2(x.shape[1]),
                                       num_warps=NUM_WARPS)
    return y


if __name__ == "__main__":
    run(rmsnorm_candidate, impl="triton fused one-row rmsnorm (hand-written)",
        extra={"triton": triton.__version__, "kernel": "rmsnorm_fwd_kernel", "num_warps": NUM_WARPS})
