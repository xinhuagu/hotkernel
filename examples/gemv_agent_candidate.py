"""Blank candidate for the gemv target: the file the agent edits in the kernel-speedrun loop.

It starts as the baseline's own call (torch.mv), so the first verify() is a fair
"inconclusive" and every later verdict measures only what the agent changed. Define
gemv_candidate(a, x) and leave the run() call alone: shape, dtype, seed, reference,
tolerances, rotation and timer are the baseline's, and verify() reports `invalid` if the
contract changes.
"""

import torch
from gemv_baseline import run


def gemv_candidate(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Starting point: identical to the baseline. Replace with something faster.
    return torch.mv(a, x)


if __name__ == "__main__":
    run(gemv_candidate, impl="agent candidate")
