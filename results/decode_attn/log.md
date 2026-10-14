# decode_attn

## Baseline choice (2026-10-14, NVIDIA GeForce RTX 5090, sm_120, torch 2.11.0+cu128, cuDNN 91900)

q (1,32,1,128), k/v (1,8,32768,128), bf16; reference = fp32 math attention; protocol 3, bench n=5. Written by scripts/choose_decode_attn_baseline.py; raw numbers in baseline_choice.json.

| backend | runs on sm_120? (enable_gqa / repeat_interleave) | path timed | mean ms | std ms | min ms | correct | >= GB/s |
|---|---|---|---|---|---|---|---|
| flash | yes / yes | enable_gqa=True | 0.1100 | 0.0032 | 0.1062 | True | 1158.7 |
| cudnn | yes / yes | enable_gqa=True | 0.0979 | 0.0062 | 0.0907 | True | 1479.5 |
| efficient | no / yes | repeat_interleave KV x4 (enable_gqa=True is rejected for this shape) | 2.6212 | 0.0612 | 2.5189 | True | 53.3 |
| math | yes / yes | enable_gqa=True | 2.8955 | 0.0106 | 2.8829 | True | 46.4 |

Chosen baseline: **cudnn** (0.0979 ms mean), fastest mean over bench n=5 among backends that run on this GPU and pass the fp32 reference check (bf16 default tolerances).
Floor at the measured 1501 GB/s for 134225920 B: 0.0894 ms; the chosen baseline is at 91.3% of it.
- efficient / enable_gqa: error: No available kernel. Aborting execution.


## Autonomous run agent-run-20261014-222305 (stopped early)

claude-opus-5-5, from the blank candidate, launched with 30 min / 20 verify. The main session
changed the limits to 10 min / 5 verify while the run was in progress, and the process was
stopped at 9 min 51 s, after the 5th verify. 5 verify calls, all `inconclusive` (0.95–1.05x),
5 rows recorded in LEADERBOARD.md as "no gain". Best candidate: a Triton split-KV decode
kernel whose own device time is ~81 µs against cuDNN's ~90 µs, but with its combine kernel the
call takes 0.1056 ms against 0.1013 ms (n=15). Trajectory and the agent's last judgment:
agent-run-20261014-222305/notes.md.

Conclusion: cuDNN SDPA is at about 91% of the measured bandwidth ceiling for this decode
shape on sm_120 (mean 0.098–0.103 ms against the 0.0894 ms floor). The remaining headroom is
about 10%, and no candidate reached it in this run. Kept as a documented negative result;
not a headline target.
