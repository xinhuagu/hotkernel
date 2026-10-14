"""Blank candidate for the rmsnorm target: the file the agent edits in the kernel-speedrun loop.

It starts as the baseline implementation, so the first verify() is a fair "inconclusive" and
every later verdict measures only what the agent changed. Define rmsnorm_candidate(x, w) and
leave the run() call alone: shape, dtype, seed, reference, tolerances, rotation and timer are
the baseline's, and verify() reports `invalid` if the contract changes.
"""

import torch
from rmsnorm_baseline import EPS, run


def rmsnorm_candidate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Starting point: identical to the baseline. Replace with something faster.
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


if __name__ == "__main__":
    run(rmsnorm_candidate, impl="agent candidate")
