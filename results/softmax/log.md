# softmax (4096x8192, bf16): number to beat, 2026-10-14

Baseline is `torch.softmax(x, dim=-1)`, the call a PyTorch user writes (a single fused library
kernel, `cunn_SoftMaxForwardReg`, fp32 accumulation), not a hand-rolled max/exp/sum/div chain.
Timer: `examples/_timing.py` (per-call CUDA events, L2 flushed before each call).

`bench(examples/softmax_baseline.py, n=5)` on the RTX 5090 at 575 W (`bench_baseline_n5.json`):
mean 0.1156 ms, std 0.0037, min 0.1093, max 0.1207, all ok.

Traffic floor is 64 MiB in + 64 MiB out = 134 MB; at 1792 GB/s that is 0.075 ms, so the
library kernel sits at about 65% of DRAM bandwidth. `torch_trace` on the baseline: the softmax
kernel is 70% of kernel time in the whole script, the rest is setup and the correctness check.

(The numbers above were measured under protocol 2, the dirty-flush timer; those rows now live in
`superseded-dirty-l2-flush/`. Under protocol 3 `bench(baseline, n=5)` gives 0.1091 ms, std 0.003.)

## Autonomous loop and protocol-3 re-measurement, 2026-10-14

`agent-run-20261014-200625/` (claude-opus-5-5, 4 verify calls, 6 min, $1.26) wrote a
one-program-per-row Triton softmax (whole row in registers, max/exp/sum in fp32, 8 warps) and
recorded it at 1.22x under protocol 2; the session reproduced it at 1.24x. The kernel is kept
verbatim in `agent-run-20261014-200625/candidate_protocol3.py`.

Under protocol 3, three independent `verify(n=7)`, all recorded by `record()`:

| kernel | verdicts | speedup | candidate ms | baseline (torch.softmax) ms | GB/s | % of the 1501 GB/s copy ceiling |
|---|---|---|---|---|---|---|
| agent kernel | faster x3 | 1.17-1.21x | 0.0903-0.0929 | 0.1068-0.1114 | 1445-1486 | 96-99% |

The agent's kernel runs at the card's measured streaming ceiling; `torch.softmax` runs at
1205-1257 GB/s (80-84%). The spec-sheet floor (0.0749 ms) is not reachable on this card.
