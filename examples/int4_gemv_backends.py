"""Time one int4 weight-only path on the int4_gemv target's exact contract, so the baseline is
chosen by measurement: `bench(script="examples/int4_gemv_backends.py", n=5,
args=["--path", "int4pack_k4"])` for each path in int4_gemv_paths.PATHS. Each run prints the
target's JSON line with `ok` against the fp32 dequantize reference, under the protocol-3
timer, with the same raw inputs, seed, rotation and min_bytes as the baseline; only the
inner_k_tiles of the packed layout follows the path. The comparison and the choice are
written to results/int4_gemv/baseline_choice.json by scripts/choose_int4_gemv_baseline.py.
"""

import argparse

from int4_gemv_baseline import make_target
from int4_gemv_paths import PATHS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", choices=sorted(PATHS), required=True)
    a = p.parse_args()
    fn, inner_k_tiles = PATHS[a.path]
    make_target(inner_k_tiles).run(fn, impl=f"int4 path {a.path}",
                                   extra={"path": a.path, "inner_k_tiles": inner_k_tiles})


if __name__ == "__main__":
    main()
