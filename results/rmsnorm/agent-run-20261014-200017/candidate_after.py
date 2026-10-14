"""Blank candidate for the rmsnorm target: the file the agent edits in the kernel-speedrun loop.

It starts as a copy of the baseline implementation so the first verify() is a fair
"inconclusive" and every later verdict measures only what the agent changed. Shape, dtype,
reference, tolerances, iteration count and the timer are imported from the baseline and
must stay that way; verify() returns `invalid` if the contract changes.
"""

import json
import sys

import torch
import triton
import triton.language as tl
from rmsnorm_baseline import (
    COLS,
    DTYPE,
    EPS,
    ITERS,
    ROWS,
    SEED,
    TIMING,
    rmsnorm_reference,
    time_kernel,
)


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


def main() -> None:
    torch.manual_seed(SEED)
    x = torch.randn(ROWS, COLS, device="cuda", dtype=DTYPE)
    w = torch.randn(COLS, device="cuda", dtype=DTYPE)

    out = rmsnorm_candidate(x, w)
    ref = rmsnorm_reference(x, w)
    ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2)

    ms = time_kernel(lambda: rmsnorm_candidate(x, w), iters=ITERS)

    bytes_moved = 2 * x.numel() * x.element_size()
    print(json.dumps({
        "ms": round(ms, 5), "ok": bool(ok), "shape": [ROWS, COLS], "dtype": str(DTYPE),
        "iters": ITERS, "seed": SEED, "timing": TIMING, "min_bytes": bytes_moved,
        "achieved_gbs_lower_bound": round(bytes_moved / (ms / 1e3) / 1e9, 1),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "impl": "agent candidate",
    }))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
