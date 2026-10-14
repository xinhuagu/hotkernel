"""Protocol-3 re-measurement wrapper for the kernel the autonomous loop left in
candidate_after.py (agent-run-20261014-200625, claude-opus-5-5, measured there under
protocol 2). The kernel code is verbatim; only the harness changed: the function is handed to
the baseline's run(), which owns shape, dtype, reference, tolerances, rotation and timer.
Run from the repo root: python results/softmax/agent-run-20261014-200625/candidate_protocol3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

import torch
import triton
import triton.language as tl
from softmax_baseline import run

RUN = "agent-run-20261014-200625"


@triton.jit
def _softmax_row_kernel(x_ptr, y_ptr, stride, n_cols, BLOCK: tl.constexpr):
    # One program per row; the whole row lives in registers, so it is read once and written once.
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * stride + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def softmax_candidate(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    y = torch.empty_like(x)
    _softmax_row_kernel[(rows,)](x, y, x.stride(0), cols, BLOCK=triton.next_power_of_2(cols),
                                 num_warps=8)
    return y


if __name__ == "__main__":
    run(softmax_candidate,
        impl=f"kernel found by autonomous loop ({RUN}) under protocol 2, re-measured under protocol 3",
        extra={"triton": triton.__version__, "kernel": "_softmax_row_kernel", "num_warps": 8,
               "origin_run": RUN})
