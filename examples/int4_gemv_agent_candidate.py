"""Blank candidate for the int4_gemv target: the file the agent edits in the kernel-speedrun
loop.

It starts as the baseline's own call (torch._weight_int4pack_mm), so the first verify() is a
fair "inconclusive" and every later verdict measures only what the agent changed. Define
int4_gemv_candidate(x, w_q, scales, zeros, w_packed, scales_and_zeros) and leave the run()
call alone: inputs, quantization, seed, reference, tolerances, rotation and timer are the
baseline's, and verify() reports `invalid` if the contract changes.

Either weight layout may be read, never both per call: w_q is (n, k/2) uint8 with the even
column in the high nibble, scales and zeros are (n, k/128) bf16 and dequant(q) = (q - 8) *
scale + zero; w_packed and scales_and_zeros are the same codes in tinygemm's layout. Triton
and torch.compile are fair game.
"""

from int4_gemv_baseline import baseline_fn, run


def int4_gemv_candidate(x, w_q, scales, zeros, w_packed, scales_and_zeros):
    # Starting point: identical to the baseline. Replace with something faster.
    return baseline_fn(x, w_q, scales, zeros, w_packed, scales_and_zeros)


if __name__ == "__main__":
    run(int4_gemv_candidate, impl="agent candidate")
