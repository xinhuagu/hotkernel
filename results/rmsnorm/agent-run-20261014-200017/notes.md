# rmsnorm agent run 20261014-200017

Agent: claude-opus-5-5, autonomous loop. Card: RTX 5090, driver 617.42, 575 W.
Shape 4096x8192 bf16. roofline(flops=1.34e8, bytes=1.34e8): memory-bound, attainable_ms 0.0749
(the 10% stop line is 0.0824 ms).

## Result

Best verdict this run: `faster`, 3.27x (candidate 0.0920 ms vs baseline 0.3008 ms, n=7),
log `results/rmsnorm/20261014-200159.json`. It did not beat the target's earlier best
(0.09065 ms, `results/rmsnorm/20261014-195124.json`, from a previous run), so every row here
answered `keep: false`. Stopped after three consecutive `keep: false` (3 of 10 verify calls used).
The best mean is still 23% above the roofline floor.

## Hypotheses

| # | Change | Verdict | Candidate ms | Speedup | keep | Log |
|---|---|---|---|---|---|---|
| 1 | Fused Triton kernel, one program per row, BLOCK=8192, num_warps=8, fp32 accumulate | faster | 0.0945 | 3.35x | false | 20261014-200116.json |
| 2 | Same, num_warps 8 -> 16 | faster | 0.0920 | 3.27x | false | 20261014-200159.json |
| 3 | #2 + evict_first on x/y, evict_last on w | faster | 0.0922 | 3.32x | false | 20261014-200245.json |

The speedup column reflects baseline noise. Hypothesis 2 has the lowest candidate mean but the
lowest speedup, because its baseline came out at 0.3008 ms against 0.3166 ms in run 1.

The candidate file was left at hypothesis 2 (16 warps, no cache hints).

## Where I got stuck

- `keep` compares against the best row across all runs for the target. The earlier best
  (0.09065 ms) came from a previous run whose code is not in the candidate file, so the advice
  "revert to the previous best" could not be followed. I reverted to my own lowest-mean variant
  instead. Every row I recorded fell within about 1-4 us of that earlier best, which is around
  the noise level. Under this rule, three `keep: false` answers came quickly even though all
  three rows were `faster`.
- The kernel runs at about 1.46 TB/s, roughly 81% of the 1.79 TB/s spec. I did not profile
  (ncu, disasm) to find out why, because of the 10-minute budget.

## Unclear tool outputs

- `record()` does not say that `previous_best_candidate_ms` can come from a different
  run/session. The advice "revert to the previous best" assumes that code is in the file.
- In verify(), the speedup comes from the baseline mean and moves with baseline noise (0.30-0.32 ms
  here), so speedups from different rows can't be compared directly. The candidate mean is the
  number `keep` uses.
