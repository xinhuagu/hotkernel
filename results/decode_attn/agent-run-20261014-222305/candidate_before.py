"""Candidate for the "decode_attn" target: the file the kernel-speedrun loop edits.

It starts as the baseline itself, so the first verify() is a fair "inconclusive" and every
later verdict measures only what changed. Define decode_attn_candidate(x, k, v) and leave the
run() call alone: inputs, seed, reference, tolerances, rotation and timer are the baseline's,
and verify() reports `invalid` if the contract changes. Triton and torch.compile are fair game.
"""

from decode_attn_baseline import baseline_fn, run


def decode_attn_candidate(x, k, v):
    # Starting point: the baseline. Replace with something faster; keep the signature.
    return baseline_fn(x, k, v)


if __name__ == "__main__":
    run(decode_attn_candidate, impl="agent candidate")
