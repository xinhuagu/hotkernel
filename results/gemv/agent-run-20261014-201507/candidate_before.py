"""Blank candidate for the gemv target: the file the agent edits in the kernel-speedrun loop.

It starts as the baseline's own call (torch.mv), so the first verify() is a fair
"inconclusive" and every later verdict measures only what the agent changed. Shape, dtype,
reference, tolerances, iteration count and the timer are imported from the baseline and must
stay that way; verify() returns `invalid` if the contract changes.
"""

import json
import sys

import torch
from gemv_baseline import (
    ATOL,
    DTYPE,
    ITERS,
    RTOL,
    SEED,
    TIMING,
    M,
    N,
    gemv_reference,
    time_kernel,
)


def gemv_candidate(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Starting point: identical to the baseline. Replace with something faster.
    return torch.mv(a, x)


def main() -> None:
    torch.manual_seed(SEED)
    a = torch.randn(M, N, device="cuda", dtype=DTYPE)
    x = torch.randn(N, device="cuda", dtype=DTYPE)

    out = gemv_candidate(a, x)
    ref = gemv_reference(a, x)
    ok = torch.allclose(out.float(), ref.float(), rtol=RTOL, atol=ATOL)

    ms = time_kernel(lambda: gemv_candidate(a, x), iters=ITERS)

    bytes_moved = a.numel() * a.element_size() + (x.numel() + out.numel()) * x.element_size()
    print(json.dumps({
        "ms": round(ms, 5), "ok": bool(ok), "shape": [M, N], "dtype": str(DTYPE),
        "iters": ITERS, "seed": SEED, "timing": TIMING, "min_bytes": bytes_moved,
        "flops": 2 * M * N,
        "achieved_gbs_lower_bound": round(bytes_moved / (ms / 1e3) / 1e9, 1),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "impl": "agent candidate",
    }))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
