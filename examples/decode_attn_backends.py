"""Time one SDPA backend on the decode_attn target's exact contract, so the baseline can be
chosen by measurement: `bench(script="examples/decode_attn_backends.py", n=5,
args=["--backend", "cudnn"])` for each of flash, cudnn, efficient, math. Each run prints the
target's JSON line with `ok` against the fp32 reference, under the protocol-3 timer, with the
same inputs, seed and rotation as the baseline, because it reuses the baseline's TARGET.
The comparison and the choice are written to results/decode_attn/baseline_choice.json by
scripts/choose_decode_attn_baseline.py.
"""

import argparse

from decode_attn_baseline import TARGET
from decode_attn_sdpa import BACKENDS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    a = p.parse_args()
    TARGET.run(BACKENDS[a.backend], impl=f"torch sdpa backend {a.backend}",
               extra={"backend": a.backend})


if __name__ == "__main__":
    main()
