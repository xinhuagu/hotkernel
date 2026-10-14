"""Blank candidate for the softmax target: the file the agent edits in the kernel-speedrun loop.

It starts as the baseline's own call (torch.softmax), so the first verify() is a fair
"inconclusive" and every later verdict measures only what the agent changed. Shape, dtype,
reference, tolerances, iteration count and the timer are imported from the baseline and must
stay that way; verify() returns `invalid` if the contract changes.
"""

import json
import sys

import torch
import triton
import triton.language as tl
from softmax_baseline import (
    ATOL,
    COLS,
    DTYPE,
    ITERS,
    ROWS,
    RTOL,
    SEED,
    TIMING,
    softmax_reference,
    time_kernel,
)


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


def main() -> None:
    torch.manual_seed(SEED)
    x = torch.randn(ROWS, COLS, device="cuda", dtype=DTYPE)

    out = softmax_candidate(x)
    ref = softmax_reference(x)
    ok = torch.allclose(out.float(), ref.float(), rtol=RTOL, atol=ATOL)

    ms = time_kernel(lambda: softmax_candidate(x), iters=ITERS)

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
