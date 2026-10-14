"""Blank candidate for the softmax target: the file the agent edits in the kernel-speedrun loop.

It starts as the baseline's own call (torch.softmax), so the first verify() is a fair
"inconclusive" and every later verdict measures only what the agent changed. Shape, dtype,
reference, tolerances, iteration count and the timer are imported from the baseline and must
stay that way; verify() returns `invalid` if the contract changes.
"""

import json
import sys

import torch
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


def softmax_candidate(x: torch.Tensor) -> torch.Tensor:
    # Starting point: identical to the baseline. Replace with something faster.
    return torch.softmax(x, dim=-1)


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
