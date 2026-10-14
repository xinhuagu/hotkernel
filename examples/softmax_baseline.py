"""Baseline target: row softmax, 4096x8192 bf16, as a PyTorch user writes it: torch.softmax.

This is deliberately the library call, not a hand-rolled max/exp/sum/div chain: a speedup
over a composite nobody uses would not mean anything. torch.softmax on bf16 is a single fused
kernel with fp32 accumulation, so a candidate has to be genuinely better, and "no gain" is
an honest outcome for this target.

Contract (see skills/kernel-speedrun/SKILL.md): a target prints ONE JSON line
{"ms": ..., "ok": ...}. Inputs, seed, reference, tolerances, rotation and timer live in TARGET
and in _hotkernel_contract.py (protocol 3); a candidate defines a function with the same
signature and calls run(candidate, impl="..."). This file is never edited.
"""

import torch
from _hotkernel_contract import Target

ROWS, COLS = 4096, 8192
DTYPE = torch.bfloat16
ITERS = 50
SEED = 0
RTOL, ATOL = 2e-2, 1e-6  # bf16 output of values in (0, 1); zeros allowed


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def softmax_baseline(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


TARGET = Target(name="softmax", dtype=DTYPE, shapes={"x": (ROWS, COLS)}, rotate=("x",),
                reference=softmax_reference, rtol=RTOL, atol=ATOL, iters=ITERS, seed=SEED,
                flops=5 * ROWS * COLS, reference_name="torch.softmax in fp32")


def run(fn, impl: str, extra: dict | None = None) -> None:
    """Check fn against the reference, time it under protocol 3, print the JSON line, exit."""
    TARGET.run(fn, impl, extra)


if __name__ == "__main__":
    run(softmax_baseline, impl="torch.softmax")
