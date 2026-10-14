"""Baseline target for the kernel-speedrun loop: RMSNorm in plain PyTorch eager.

Contract (see skills/kernel-speedrun/SKILL.md): a target prints ONE JSON line
{"ms": <mean kernel ms per call>, "ok": <correctness vs fp32 reference>, ...}.

Everything that makes the number honest lives in TARGET below and in _hotkernel_contract.py
(the vendored contract, protocol 3): the inputs (shapes, dtype, seed), the fp32 reference and
its tolerances, the input rotation and the timer. A candidate only defines a function with the
same signature and calls run(candidate, impl="..."); it cannot change any of this without
changing the contract fields that verify() checks. This file is never edited.
"""

import torch
from _hotkernel_contract import Target

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


TARGET = Target(name="rmsnorm", dtype=DTYPE, shapes={"x": (ROWS, COLS), "w": (COLS,)}, rotate=("x",),
                reference=rmsnorm_reference, rtol=RTOL, atol=ATOL, iters=ITERS, seed=SEED,
                flops=4 * ROWS * COLS, reference_name="rmsnorm_reference (fp32)")


def run(fn, impl: str, extra: dict | None = None) -> None:
    """Check fn against the reference, time it under protocol 3, print the JSON line, exit."""
    TARGET.run(fn, impl, extra)


if __name__ == "__main__":
    run(rmsnorm_baseline, impl="eager bf16: pow, mean, rsqrt, mul, mul")
