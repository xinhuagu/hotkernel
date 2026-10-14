"""Baseline target: matrix-vector product, A[8192, 8192] bf16 times x[8192] bf16, as a PyTorch
user writes it: torch.mv (cuBLAS underneath).

The baseline is the library call on purpose. If a candidate beats it, the row says "faster
than examples/gemv_baseline.py", measured the same way; it never says "beats cuBLAS". gemv is
bandwidth-bound: the 128 MiB matrix has to be read once per call.

Contract (see skills/kernel-speedrun/SKILL.md): a target prints ONE JSON line
{"ms": ..., "ok": ...}. Inputs, seed, reference, tolerances, rotation and timer live in TARGET
and in _hotkernel_contract.py (protocol 3); a candidate defines a function with the same
signature and calls run(candidate, impl="..."). This file is never edited.
"""

import torch
from _hotkernel_contract import Target

M, N = 8192, 8192
DTYPE = torch.bfloat16
ITERS = 50
SEED = 0
RTOL, ATOL = 2e-2, 5e-2  # outputs are O(sqrt(N)) ~ 90; fp32-accumulated bf16 result


def gemv_reference(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (a.float() @ x.float()).to(a.dtype)


def gemv_baseline(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mv(a, x)


TARGET = Target(name="gemv", dtype=DTYPE, shapes={"a": (M, N), "x": (N,)}, rotate=("a",),
                reference=gemv_reference, rtol=RTOL, atol=ATOL, iters=ITERS, seed=SEED,
                flops=2 * M * N, reference_name="fp32 matmul")


def run(fn, impl: str, extra: dict | None = None) -> None:
    """Check fn against the reference, time it under protocol 3, print the JSON line, exit."""
    TARGET.run(fn, impl, extra)


if __name__ == "__main__":
    run(gemv_baseline, impl="torch.mv")
