"""Blank candidate for the gemv target: the file the agent edits in the kernel-speedrun loop.

It starts as the baseline's own call (torch.mv), so the first verify() is a fair
"inconclusive" and every later verdict measures only what the agent changed. Shape, dtype,
reference, tolerances, iteration count and the timer are imported from the baseline and must
stay that way; verify() returns `invalid` if the contract changes.
"""

import json
import sys

import torch
import triton
import triton.language as tl
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
    grid = lambda meta: (m // meta["BLOCK_M"],)
    _gemv_kernel[grid](a, x, y, m, n, a.stride(0))
    return y


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
