"""Protocol-3 re-measurement wrapper for the kernel the autonomous loop left in
candidate_after.py (agent-run-20261014-201507, claude-opus-5-5, measured there under
protocol 2). The kernel code is verbatim, autotune space included; only the harness changed:
the function is handed to the baseline's run(), which owns shape, dtype, reference,
tolerances, rotation and timer. The chosen autotune config is printed in the JSON line.
Run from the repo root: python results/gemv/agent-run-20261014-201507/candidate_protocol3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

import torch
import triton
import triton.language as tl
from gemv_baseline import run

RUN = "agent-run-20261014-201507"


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm, bn, w, s in [
            (2, 1024, 4, 3), (4, 512, 4, 3), (4, 1024, 4, 3), (4, 1024, 8, 3),
            (8, 512, 4, 3), (8, 512, 8, 3), (8, 1024, 8, 2), (16, 256, 4, 3),
        ]
    ],
    key=["M", "N"],
)
@triton.jit
def _gemv_kernel(a_ptr, x_ptr, y_ptr, M, N, stride_am,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # One program owns BLOCK_M full rows; fp32 accumulation, one reduction at the end.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    a_ptrs = a_ptr + rows[:, None] * stride_am + cols[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, N, BLOCK_N):
        a = tl.load(a_ptrs + k, eviction_policy="evict_first")
        x = tl.load(x_ptr + k + cols)
        acc += a.to(tl.float32) * x.to(tl.float32)[None, :]
    tl.store(y_ptr + rows, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty))


def gemv_candidate(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    assert a.stride(1) == 1 and m % 16 == 0 and n % 1024 == 0
    y = torch.empty(m, device=a.device, dtype=a.dtype)
    grid = lambda meta: (m // meta["BLOCK_M"],)  # noqa: E731
    _gemv_kernel[grid](a, x, y, m, n, a.stride(0))
    return y


def _chosen_config() -> str:
    best = getattr(_gemv_kernel, "best_config", None)
    return str(best) if best is not None else "unknown"


if __name__ == "__main__":
    # Trigger autotuning once so the chosen config can be reported; run() times it afterwards.
    a0 = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    x0 = torch.randn(8192, device="cuda", dtype=torch.bfloat16)
    gemv_candidate(a0, x0)
    torch.cuda.synchronize()
    del a0, x0
    run(gemv_candidate,
        impl=f"kernel found by autonomous loop ({RUN}) under protocol 2, re-measured under protocol 3",
        extra={"triton": triton.__version__, "kernel": "_gemv_kernel (autotuned)",
               "config": _chosen_config(), "origin_run": RUN})
