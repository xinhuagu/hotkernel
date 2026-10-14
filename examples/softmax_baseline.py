"""Baseline target: row softmax, 4096x8192 bf16, as a PyTorch user writes it: torch.softmax.

This is deliberately the library call, not a hand-rolled max/exp/sum/div chain: a speedup
over a composite nobody uses would not mean anything. torch.softmax on bf16 is a single fused
kernel with fp32 accumulation, so a candidate has to be genuinely better, and "no gain" is
an honest outcome for this target.

Contract (see skills/kernel-speedrun/SKILL.md): a target prints ONE JSON line
{"ms": ..., "ok": ...}. Shape, dtype, seed, reference, tolerances, rotation and timer live in
run() below; a candidate defines a function with the same signature and calls
run(candidate, impl="..."). This file is never edited.
"""

import json
import sys

import torch
from _timing import PROTOCOL, TIMING, l2_bytes, rotation_k, time_kernel

ROWS, COLS = 4096, 8192
DTYPE = torch.bfloat16
ITERS = 50
SEED = 0
RTOL, ATOL = 2e-2, 1e-6  # bf16 output of values in (0, 1); zeros allowed


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def softmax_baseline(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def run(fn, impl: str, extra: dict | None = None) -> None:
    """Check fn against the reference, time it under protocol 3, print the JSON line, exit."""
    torch.manual_seed(SEED)
    elem = torch.finfo(DTYPE).bits // 8
    pair_bytes = 2 * ROWS * COLS * elem  # one input read + one output write per call
    k = rotation_k(pair_bytes)
    xs = [torch.randn(ROWS, COLS, device="cuda", dtype=DTYPE) for _ in range(k)]

    out = fn(xs[0])
    ref = softmax_reference(xs[0])
    ok = torch.allclose(out.float(), ref.float(), rtol=RTOL, atol=ATOL)

    ms = time_kernel(fn, [(x,) for x in xs], iters=ITERS)

    print(json.dumps({
        "ms": round(ms, 5), "ok": bool(ok), "shape": [ROWS, COLS], "dtype": str(DTYPE),
        "iters": ITERS, "seed": SEED, "protocol": PROTOCOL, "timing": TIMING,
        "rotation_k": k, "rotation_bytes": k * pair_bytes, "l2_bytes": l2_bytes(),
        "min_bytes": pair_bytes, "flops": 5 * ROWS * COLS,
        "achieved_gbs_lower_bound": round(pair_bytes / (ms / 1e3) / 1e9, 1),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "impl": impl, **(extra or {}),
    }))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    run(softmax_baseline, impl="torch.softmax")
