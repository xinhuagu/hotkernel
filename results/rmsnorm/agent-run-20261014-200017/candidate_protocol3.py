"""Protocol-3 re-measurement wrapper for the kernel the autonomous loop left in
candidate_after.py (agent-run-20261014-200017, claude-opus-5-5, measured there under
protocol 2). The kernel code is verbatim; only the harness changed: the function is handed to
the baseline's run(), which owns shape, dtype, reference, tolerances, rotation and timer.
Run from the repo root: python results/rmsnorm/agent-run-20261014-200017/candidate_protocol3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

import torch
import triton
import triton.language as tl
from rmsnorm_baseline import EPS, run

RUN = "agent-run-20261014-200017"


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    # One program per row: the whole row stays in registers, one read and one write.
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(y_ptr + row * n_cols + offs, (x * rstd * w).to(tl.bfloat16), mask=mask)


def rmsnorm_candidate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    rows, cols = x.shape
    _rmsnorm_kernel[(rows,)](x, w, y, cols, EPS, BLOCK=triton.next_power_of_2(cols), num_warps=16)
    return y


if __name__ == "__main__":
    run(rmsnorm_candidate,
        impl=f"kernel found by autonomous loop ({RUN}) under protocol 2, re-measured under protocol 3",
        extra={"triton": triton.__version__, "kernel": "_rmsnorm_kernel", "num_warps": 16,
               "origin_run": RUN})
