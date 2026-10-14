"""Baseline target for the kernel-speedrun loop: RMSNorm in plain PyTorch eager.

Contract (see skills/kernel-speedrun/SKILL.md): a target prints ONE JSON line
{"ms": <mean kernel ms per call>, "ok": <correctness vs fp32 reference>, ...}.

Everything that makes the number honest lives in run() below: the shape, dtype, seed, the
fp32 reference and its tolerances, the input rotation and the timer (examples/_timing.py,
protocol 3). A candidate only defines a function with the same signature and calls
run(candidate, impl="..."); it cannot change any of this without changing the contract fields
that verify() checks. This file is never edited.
"""

import json
import sys

import torch
from _timing import PROTOCOL, TIMING, l2_bytes, rotation_k, time_kernel

ROWS, COLS = 4096, 8192
DTYPE = torch.bfloat16
EPS = 1e-6
ITERS = 50
SEED = 0
RTOL, ATOL = 2e-2, 2e-2


def rmsnorm_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(x.dtype)


def rmsnorm_baseline(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Plain eager bf16: five kernels, several passes over memory. This is the number to beat.
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def run(fn, impl: str, extra: dict | None = None) -> None:
    """Check fn against the reference, time it under protocol 3, print the JSON line, exit."""
    torch.manual_seed(SEED)
    elem = torch.finfo(DTYPE).bits // 8
    pair_bytes = 2 * ROWS * COLS * elem  # one input read + one output write per call
    k = rotation_k(pair_bytes)
    xs = [torch.randn(ROWS, COLS, device="cuda", dtype=DTYPE) for _ in range(k)]
    w = torch.randn(COLS, device="cuda", dtype=DTYPE)

    out = fn(xs[0], w)
    ref = rmsnorm_reference(xs[0], w)
    ok = torch.allclose(out.float(), ref.float(), rtol=RTOL, atol=ATOL)

    ms = time_kernel(fn, [(x, w) for x in xs], iters=ITERS)

    print(json.dumps({
        "ms": round(ms, 5), "ok": bool(ok), "shape": [ROWS, COLS], "dtype": str(DTYPE),
        "iters": ITERS, "seed": SEED, "protocol": PROTOCOL, "timing": TIMING,
        "rotation_k": k, "rotation_bytes": k * pair_bytes, "l2_bytes": l2_bytes(),
        "min_bytes": pair_bytes, "flops": 4 * ROWS * COLS,
        "achieved_gbs_lower_bound": round(pair_bytes / (ms / 1e3) / 1e9, 1),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "impl": impl, **(extra or {}),
    }))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    run(rmsnorm_baseline, impl="eager bf16: pow, mean, rsqrt, mul, mul")
