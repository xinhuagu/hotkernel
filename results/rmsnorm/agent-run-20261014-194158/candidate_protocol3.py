"""Protocol-3 re-measurement wrapper for the kernel the autonomous loop left in
candidate_after.py (agent-run-20261014-194158, claude-fable-5-1, measured there under
protocol 2). The kernel code is verbatim; only the harness changed: the function is handed to
the baseline's run(), which owns shape, dtype, reference, tolerances, rotation and timer.
Run from the repo root: python results/rmsnorm/agent-run-20261014-194158/candidate_protocol3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

import torch
import triton
import triton.language as tl
from rmsnorm_baseline import EPS, run

RUN = "agent-run-20261014-194158"


@triton.jit
def _rmsnorm_fused_kernel(
    x_ptr, w_ptr, out_ptr,
    stride_row,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per row. The whole row (N bf16 values) is loaded once into registers,
    # the sum of squares is reduced in fp32, and the normalized row is written once.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.rsqrt(ms + eps)
    y = x * inv * w
    tl.store(out_ptr + row * stride_row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm_candidate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    rows, n = x.shape
    _rmsnorm_fused_kernel[(rows,)](
        x, w, out, x.stride(0), EPS,
        N=n, BLOCK=triton.next_power_of_2(n), num_warps=8,
    )
    return out


if __name__ == "__main__":
    run(rmsnorm_candidate,
        impl=f"kernel found by autonomous loop ({RUN}) under protocol 2, re-measured under protocol 3",
        extra={"triton": triton.__version__, "kernel": "_rmsnorm_fused_kernel", "num_warps": 8,
               "origin_run": RUN})
