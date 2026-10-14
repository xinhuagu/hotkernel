"""Baseline target: matrix-vector product, A[8192, 8192] bf16 times x[8192] bf16, as a PyTorch
user writes it: torch.mv (cuBLAS underneath).

The baseline is the library call on purpose. If a candidate beats it, the row says "faster
than examples/gemv_baseline.py", measured the same way; it never says "beats cuBLAS". gemv is
bandwidth-bound: the 128 MiB matrix has to be read once per call.

Contract (see skills/kernel-speedrun/SKILL.md): a target prints ONE JSON line
{"ms": ..., "ok": ...}. Shape, dtype, seed, reference, tolerances, rotation and timer live in
run() below; a candidate defines a function with the same signature and calls
run(candidate, impl="..."). This file is never edited.
"""

import json
import sys

import torch
from _timing import PROTOCOL, TIMING, l2_bytes, rotation_k, time_kernel

M, N = 8192, 8192
DTYPE = torch.bfloat16
ITERS = 50
SEED = 0
RTOL, ATOL = 2e-2, 5e-2  # outputs are O(sqrt(N)) ~ 90; fp32-accumulated bf16 result


def gemv_reference(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (a.float() @ x.float()).to(a.dtype)


def gemv_baseline(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mv(a, x)


def run(fn, impl: str, extra: dict | None = None) -> None:
    """Check fn against the reference, time it under protocol 3, print the JSON line, exit."""
    torch.manual_seed(SEED)
    elem = torch.finfo(DTYPE).bits // 8
    pair_bytes = (M * N + N + M) * elem  # matrix + vector read, result written, per call
    k = rotation_k(pair_bytes)
    mats = [torch.randn(M, N, device="cuda", dtype=DTYPE) for _ in range(k)]
    x = torch.randn(N, device="cuda", dtype=DTYPE)

    out = fn(mats[0], x)
    ref = gemv_reference(mats[0], x)
    ok = torch.allclose(out.float(), ref.float(), rtol=RTOL, atol=ATOL)

    ms = time_kernel(fn, [(a, x) for a in mats], iters=ITERS)

    print(json.dumps({
        "ms": round(ms, 5), "ok": bool(ok), "shape": [M, N], "dtype": str(DTYPE),
        "iters": ITERS, "seed": SEED, "protocol": PROTOCOL, "timing": TIMING,
        "rotation_k": k, "rotation_bytes": k * pair_bytes, "l2_bytes": l2_bytes(),
        "min_bytes": pair_bytes, "flops": 2 * M * N,
        "achieved_gbs_lower_bound": round(pair_bytes / (ms / 1e3) / 1e9, 1),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "impl": impl, **(extra or {}),
    }))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    run(gemv_baseline, impl="torch.mv")
