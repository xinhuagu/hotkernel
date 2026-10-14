"""Blank candidate for the softmax target: the file the agent edits in the kernel-speedrun loop.

It starts as the baseline's own call (torch.softmax), so the first verify() is a fair
"inconclusive" and every later verdict measures only what the agent changed. Define
softmax_candidate(x) and leave the run() call alone: shape, dtype, seed, reference,
tolerances, rotation and timer are the baseline's, and verify() reports `invalid` if the
contract changes.
"""

import torch
from softmax_baseline import run


def softmax_candidate(x: torch.Tensor) -> torch.Tensor:
    # Starting point: identical to the baseline. Replace with something faster.
    return torch.softmax(x, dim=-1)


if __name__ == "__main__":
    run(softmax_candidate, impl="agent candidate")
