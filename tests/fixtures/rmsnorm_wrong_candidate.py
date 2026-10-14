"""A candidate that is WRONG, kept to prove the negative path of verify()/record().

It follows the contract (it hands its function to the baseline's run(), so shape, dtype, seed,
reference, tolerances, rotation and timer are the baseline's) but skips the weight multiply,
so `ok` is false whatever `ms` says. verify() must return verdict "incorrect" with no speedup,
and record() must write "incorrect" in the Speedup cell.
Run from the repo root: python tests/fixtures/rmsnorm_wrong_candidate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

import torch
from rmsnorm_baseline import EPS, run


def rmsnorm_wrong(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Bug on purpose: the weight is never applied.
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS).to(x.dtype)


if __name__ == "__main__":
    run(rmsnorm_wrong, impl="deliberately wrong (no weight)")
