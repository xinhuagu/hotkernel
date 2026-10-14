# int4_gemv

## Baseline choice (2026-10-14, NVIDIA GeForce RTX 5090, sm_120, torch 2.11.0+cu128, torchao 0.17.0+cu128)

W 8192x8192 int4 (group 128, bf16 scale and zero), x bf16 (8192,); reference = fp32 dequantize + fp32 matvec; protocol 3, bench n=5. Written by scripts/choose_int4_gemv_baseline.py; raw numbers in baseline_choice.json.

| path | runs? | mean ms | std ms | min ms | correct | >= GB/s | K |
|---|---|---|---|---|---|---|---|
| int4pack_k2 | yes | 0.03470 | 0.00470 | 0.02930 | True | 946.8 | 6 |
| int4pack_k4 | yes | 0.03140 | 0.00300 | 0.02870 | True | 1038.6 | 6 |
| int4pack_k8 | yes | 0.03610 | 0.01200 | 0.02760 | True | 599.9 | 6 |
| dequant_mv | yes | 1.43330 | 0.01920 | 1.41180 | True | 25.0 | 6 |

torchao Int4WeightOnlyConfig packing formats on an nn.Linear (own quantization, warm-L2 CUDA events, outside the contract):

- plain: does not run: Requires mslk >= 1.0.0
- preshuffled: does not run: Requires mslk >= 1.0.0
- plain_int32: does not run: Int4PlainInt32Tensor does not support device 'cuda' yet.
- tile_packed_to_4d: runs (Int4TilePackedTo4dTensor), 28.9 us warm, max abs err vs fp32 unquantized matvec 36.383 (absmax 352.946)
- for comparison, int4pack_k4 warm: 16.8 us, max abs err vs fp32 unquantized matvec 33.565

Quantizer vs torchao groupwise-affine on the same weight: {"codes_equal_fraction": 0.947841, "scales_and_zeros_max_abs_diff": 0.03125, "note": "same convention: (q - 8) * scale + zero, even column in the high nibble"}

Chosen baseline: **int4pack_k4** (0.03140 ms mean), fastest mean over bench n=5 among contract paths that run on this GPU and pass the fp32 dequantize reference check (bf16 default tolerances); torchao's module paths quantize with their own parameters, so they are compared on warm-L2 time only.
Floor at the measured 1501 GB/s for 35684352 B: 0.02377 ms; the chosen baseline is at 75.7% of it. Rotation K=6 with min_bytes 35684352 B: 214106112 B per cycle against 2 x L2 = 201326592 B -> ok.

