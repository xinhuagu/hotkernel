# int4_gemv agent run 20261014-224714

who: `agent:claude-opus-5-5, autonomous loop, results/int4_gemv/agent-run-20261014-224714`
Card: RTX 5090, driver 617.42, 575 W. 4 verify() calls used out of 20, stopped at about 23:02 (of a 30 min budget).

## Result

**No win recorded.** Every verify() came back `inconclusive` and every record() answered `keep: false`.
No row ever had `faster`, so there is no best candidate and no speedup to claim.

Roofline (min_bytes 35,684,352 B, measured_mem_gbs 1501): attainable_ms_measured = **0.02377 ms**.
The baseline (tinygemm `_weight_int4pack_mm`, inner_k_tiles=4) measured 0.032-0.037 ms mean and 0.0285 ms min, already ~83% of that floor at its min.

## Hypotheses (verify -> record)

| # | Change | Candidate mean (ms) | Baseline mean (ms) | Verdict | keep | Log |
|---|--------|------|------|---------|------|-----|
| 0 | control: unchanged candidate | 0.0338 | 0.0326 | inconclusive | false | results/int4_gemv/20261014-224818.json |
| 1 | Triton GEMV on w_q (uint8), group-factored dequant `s*(Σqx - 8Σx) + z*Σx`, reshape-reduce per group, direct `CompiledKernel.run` launch; BLOCK_N=16,BLOCK_KB=512,warps=8 | 0.0360 | 0.0326 | inconclusive | false | results/int4_gemv/20261014-225254.json |
| 2 | same kernel, smaller blocks (ncu: 512 blocks = 1 wave + a 2-block tail at 80 regs); BLOCK_N=4,BLOCK_KB=512,warps=2 | 0.0389 | 0.0369 | inconclusive | false | results/int4_gemv/20261014-225540.json |
| 3 | v3: tiles loaded as (rows, groups, 64 B) so the group sum is an in-thread/shuffle reduction (no shared-memory reshape and BAR per step); x read as bf16 pairs through an int32 view; BLOCK_N=4,BLOCK_KB=512,warps=2 | 0.0356 | 0.0321 | inconclusive | false | results/int4_gemv/20261014-225710.json |

The candidate file is left at hypothesis 3, the closest one: median 0.0298 ms against the baseline's 0.0288 ms in the same verify().
It is still not faster than the baseline.

## bench()-only probes (not verified, not on the board)

- v1 kernel: 4,256,4 ~0.043 mean; 4,512,4 ~0.043; 8,512,4 min 0.0345; 16,512,8 min 0.0337.
- Calling the compiled kernel's launcher directly (skipping JIT binding) made no visible difference to bench minima (~0.0356 at 4,256,4).
- Register double-buffering the next weight tile (v1, 16,512,8): mean 0.0368 and very low std, but no faster. 8,256,4 got 0.040. Reverted.
- v2 (per-element dequant, a 3D accumulator reduced once at the end): 4,512,2 min 0.037, 8,256,4 0.043. Worse because of extra ALU work. Reverted.
- v3 tilings: 4,512,2 min 0.0296 / mean 0.0306 (best); 8,512,4 min 0.0304; 4,1024,2 min 0.0332; 4,1024,4 min 0.0384; 2,512,1 min 0.0367.

## Profiling evidence

- torch_trace device time per call: tinygemm 27.7 us; v1 Triton (4,256,4) 30.9 us.
- ncu v1 (16,512,8): latency bound, DRAM 58%, 80 regs, theoretical occupancy 50%, 1 wave plus a 2-block tail.
  SASS showed LDG.E.128 for the weights but scalar LDG.E.U16 for x, and STS/LDSM/BAR.SYNC every iteration (the reshape-reduce).
- ncu v3 (4,512,2): memory bound, DRAM 65-68%, 48 regs, theoretical occupancy 83% but achieved 49%, waves_per_sm 0.6 (2048 blocks fill only part of one wave).

## Where I got stuck

- v3 reaches ~68% DRAM in ncu and about tinygemm's speed, but I found no tiling that adds bytes in flight without losing occupancy or adding a tail wave.
  Untried: split-K with a second reduction, a persistent grid sized to the SM count, and `tl.range(num_stages=...)` pipelining.
- Run-to-run noise is large in both scripts: single samples of 0.04-0.06 ms against typical 0.029 ms, std 0.004-0.012 ms at n=7.
  This puts the 2*SE threshold at 0.004-0.011 ms. At that noise a gain smaller than ~15-25% cannot come out `faster` at n=7. The best realistic gain here, from 0.029 to the 0.024 floor, is about 18%.

## Tool outputs that were unclear or made me guess

- The ncu note says bench()/verify() "flush L2 before every call", but the contract (protocol 3) says no flush, steady-state rotation. The two descriptions disagree.
- torch_trace's note likewise mentions an "L2 flush disabled here", which does not match protocol 3.
- The timer includes CPU dispatch gaps when the GPU idles between calls. I could not tell from the tools how much of the baseline/candidate gap was launch overhead and how much was kernel time.
  ncu/torch_trace kernel durations were the only way to separate the two.
- `CompiledKernel[grid]` needs a 3-tuple grid. A 1-tuple failed with an IndexError inside Triton.
