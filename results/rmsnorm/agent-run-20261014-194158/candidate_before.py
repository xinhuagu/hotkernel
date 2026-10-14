"""Blank candidate for the rmsnorm target: the file the agent edits in the kernel-speedrun loop.

It starts as a copy of the baseline implementation so the first verify() is a fair
"inconclusive" and every later verdict measures only what the agent changed. Shape, dtype,
reference, tolerances, iteration count and the timer are imported from the baseline and
must stay that way; verify() returns `invalid` if the contract changes.
"""

import json
import sys

import torch
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


def rmsnorm_candidate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Starting point: identical to the baseline. Replace with something faster.
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


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
