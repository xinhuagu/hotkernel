# rmsnorm (4096x8192, bf16): first candidate, 2026-10-14

Run by hand through the MCP server (stdio) on the reference card: RTX 5090, driver 617.42,
575 W, WSL2 Ubuntu 24.04, torch 2.11.0+cu128, Triton 3.6.0. Raw files in this directory and in
`results/rmsnorm_4096x8192_bf16/` (the `record()` log named after the leaderboard target).

## Number to beat

`bench(examples/rmsnorm_baseline.py, n=5)` with the final timer (L2 flushed before every
timed call, see below): mean 0.3183 ms, std 0.009 (`bench_baseline_n5.json`). Plain eager bf16:
`x.pow(2)`, `mean`, `rsqrt`, `mul`, `mul`. (First bench of the day, without the flush: 0.3245 ms.)

## Where the time goes

- `torch_trace`: 25 kernels; the bf16 elementwise multiply is 58% of kernel time, the `pow`
  kernel 21%, the mean reduction 5% (`results/profiles/torch_trace_rmsnorm_baseline.json`).
- `profile(tool="nsys")` agrees: elementwise multiply 59.3% / 112 calls / avg 110 us, pow 19.9%,
  reduce 7.6% (`results/profiles/nsys-20261014-002317.kern_sum.csv`).
- `profile(tool="ncu", ncu_kernel_regex="elementwise_kernel")`: the pow kernel runs at 86.8%
  DRAM throughput, 6% SM throughput, 85.8% achieved occupancy: memory-bound, as expected.
- `roofline(flops=3*4096*8192, bytes=2*4096*8192*2)`: arithmetic intensity 0.75 flop/byte
  against a ridge of 116.9; the attainable number is 1.34 TFLOPS, i.e. this is a bandwidth job.
  Minimum traffic is 64 MiB in + 64 MiB out = 134 MB; at 1792 GB/s that is 0.075 ms.

## Hypothesis 1: fuse everything into one pass

One Triton program per row: load the row once (bf16 -> fp32 in registers, BLOCK=8192,
8 warps), sum of squares, `1/sqrt(ms + eps)`, scale by `w`, store once. Candidate:
`examples/rmsnorm_candidate.py`, kernel `rmsnorm_fwd_kernel`.

### Measurement fix before the row counted

The first timing loop ran the kernel 50 times on the same 64 MiB input. The RTX 5090 has
96 MB of L2, so part of the input survived between iterations: the candidate's implied
bandwidth came out at 1.8 TB/s, above the 1.79 TB/s DRAM spec, and the baseline looked
faster than it is. Three logs were recorded that way (3.91x, 3.89x, and an `incorrect` row); they live in
`superseded-no-l2-flush/` and the board lists them under "Superseded, not counted".
`time_kernel()` in the baseline now zeroes a 256 MB buffer before every timed call, outside
the timed region (the `triton.testing.do_bench` trick), every JSON line says
`"timing": "per-call CUDA events, L2 flushed before each call"`, and the candidate imports
the timer instead of owning a timing loop.

`verify(candidate, baseline, n=7, warmup=2)` with the flush (`20261014-192810.json`; a first
flushed run at 19:19 gave 3.48x with a noisier baseline and was re-recorded only to correct
the `who` label):

| | mean ms | std ms | min ms | max ms |
|---|---|---|---|---|
| baseline | 0.3131 | 0.0081 | 0.2999 | 0.3203 |
| candidate | 0.0958 | 0.0021 | 0.0913 | 0.0971 |

Verdict `faster`, speedup 3.27x, |delta| 0.2173 ms against a 2*SE threshold of 0.0063 ms,
all 7 candidate runs `ok`. This is the LEADERBOARD row. 0.0958 ms for 134 MB is 1.40 TB/s,
78% of the card's 1.79 TB/s: a believable DRAM number, with headroom the earlier figure hid.
`who` on the row says what it is: hand-written by the session during tool bring-up, not a
product of the unattended loop; that case is still open.

## The negative path, on purpose

`tests/fixtures/rmsnorm_wrong_candidate.py` keeps the contract but never applies the weight.
`verify(wrong, baseline, n=7, warmup=2)` (`20261014-192838.json`): candidate `ok=false` in 7 of
7 runs, verdict `incorrect`, `speedup: null`, `significant: null`; the regenerated board shows
`incorrect` in the Speedup cell. The timing (0.5127 ms, slower than the baseline anyway) is in
the log but never becomes a speedup.

## What ncu and SASS say about the candidate

- `profile(tool="ncu", ncu_kernel_regex="rmsnorm_fwd_kernel")`: duration 50.4 us, DRAM
  throughput 82.2%, L2 47.3%, SM 13%, achieved occupancy 60.0% of a theoretical 66.7%,
  60 registers/thread, 6 waves/SM, `bound_hint: memory`; ncu rules fired: `SOLBottleneck`,
  `TheoreticalOccupancy` (`ncu_candidate_rmsnorm_fwd_kernel.json`).
- `disasm("triton-cache", kernel="rmsnorm")`: 336 instructions, 8x `LDG.E.128` and
  4x `STG.E.128` per thread (16-byte vector loads and stores), 8x `SHFL.BFLY` for the row
  reduction, `MUFU.SQRT` + `MUFU.RCP`, no `STL`/`LDL` (no register spills)
  (`candidate_sass.txt`).

## Protocol 3 (steady-state rotation, no flush): the rows that count, 2026-10-14 20:40-20:49

Everything above was measured under protocol 2 (dirty L2 flush, ~45-50 us charged to every call;
see docs/platform-matrix.md) and now lives in `superseded-dirty-l2-flush/`. Under protocol 3,
three independent `verify(n=7)` each, all recorded by `record()`, baseline 0.286-0.300 ms:

| kernel | verdicts | speedup | candidate ms | GB/s | % of the 1501 GB/s copy ceiling |
|---|---|---|---|---|---|
| hand-written `examples/rmsnorm_candidate.py` (8 warps) | faster x3 | 3.11-3.15x | 0.0916-0.0936 | 1434-1465 | 96-98% |
| agent-run-20261014-194158 kernel (8 warps) | faster x3 | 3.17-3.32x | 0.0885-0.0945 | 1420-1517 | 95-101% |
| agent-run-20261014-200017 kernel (16 warps) | faster x3 | 3.19-3.25x | 0.0884-0.0920 | 1459-1518 | 97-101% |
| `tests/fixtures/rmsnorm_wrong_candidate.py` | incorrect x3 | none | 0.466-0.496 | | |

All three correct kernels sit at the measured streaming ceiling of this card; the remaining
distance to the spec-sheet floor (0.0749 ms) is GDDR7's practical limit, not the kernel's.
`bench(baseline, n=5)` under protocol 3: 0.3013 ms, std 0.003.

## Next hypotheses (not run)

- Occupancy is capped at 66.7% by 60 registers/thread; `num_warps=4` or `16` changes the
  register/occupancy trade-off and is a one-line change.
- At 82% DRAM throughput the remaining headroom is small; a persistent-row or two-rows-per-
  program variant could hide the per-program launch tail, but the roofline says the kernel
  is already within 10% of attainable, which is the loop's stop condition.
