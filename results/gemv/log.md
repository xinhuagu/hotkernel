# gemv (8192x8192 bf16 matrix times 8192 bf16 vector): number to beat, 2026-10-14

Baseline is `torch.mv(a, x)`, the call a PyTorch user writes (cuBLAS `gemvx::kernel`
underneath). A faster candidate is "faster than examples/gemv_baseline.py, measured the same
way"; it is never written up as beating cuBLAS. Timer: `examples/_timing.py` (per-call CUDA
events, L2 flushed before each call).

`bench(examples/gemv_baseline.py, n=5)` on the RTX 5090 at 575 W (`bench_baseline_n5.json`):
mean 0.1046 ms, std 0.0038, min 0.0994, max 0.1095, all ok.

Traffic floor is the 128 MiB matrix read once (the vectors are negligible); at 1792 GB/s that
is 0.075 ms, so the library kernel sits at about 72% of DRAM bandwidth. 134 MFLOP of work
against 134 MB of traffic is an arithmetic intensity of 1 flop/byte: memory-bound by two
orders of magnitude.

(The numbers above were measured under protocol 2, the dirty-flush timer; those rows now live in
`superseded-dirty-l2-flush/`. Under protocol 3 `bench(baseline, n=5)` gives 0.0882 ms, std 0.0001:
cuBLAS already streams the matrix at about 1520 GB/s.)

## Autonomous loop and protocol-3 re-measurement, 2026-10-14

`agent-run-20261014-201507/` (claude-opus-5-5, 3 verify calls, 7 min, $1.39) wrote a Triton
row-block GEMV autotuned over (BLOCK_M, BLOCK_N, warps), fp32 accumulate, and recorded it at
1.11x under protocol 2; the session reproduced it at 1.10x. The kernel is kept verbatim in
`agent-run-20261014-201507/candidate_protocol3.py`, which also prints the chosen config
(BLOCK_M=16, BLOCK_N=256, 4 warps under protocol 3).

Under protocol 3, three independent `verify(n=7)`, all recorded by `record()`:

| kernel | verdicts | speedup | candidate ms | baseline (torch.mv) ms | GB/s | % of the 1501 GB/s copy ceiling |
|---|---|---|---|---|---|---|
| agent kernel | inconclusive, faster, faster | 1.03-1.05x | 0.0854-0.0883 | 0.0884-0.0912 | 1520-1572 | 101-105% |

Honest reading: once the timer stops charging foreign write-backs, the agent's GEMV is only
3-5% ahead of cuBLAS and one of three runs could not separate them. Both kernels are at or
above the copy ceiling because gemv traffic is almost all reads (no read/write turnarounds);
the read-only reference point is the Triton row sum at 1449 GB/s. "no gain" would be an
acceptable row here; the board shows the three rows as they came.
