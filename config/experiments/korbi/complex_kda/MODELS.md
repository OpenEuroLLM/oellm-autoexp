# Complex-KDA sweep: models (generated)

MLP width matched to the dense `attn` arm at each size (grid 64). Params are total, tied embedding counted once; `aux.params_by_size` in the sweep is the same number, to compare against Megatron's startup print.

## s47M: d 384, 12 layers, 6 heads, dense ffn 1536

| arm | params | delta vs dense | ffn | ffn/d | tok/s/GPU @ mbs cap |
|---|---:|---:|---:|---:|---:|
| attn | 47,637,888 | +0.00% | 1536 | 4.00 | 335,853 |
| attn_qknorm | 47,639,424 | +0.00% | 1536 | 4.00 | 335,853 |
| kda_sig_lowrank | 48,025,800 | +0.81% | 1472 | 3.83 | 256,093 |
| ckda_shipped_lowrank | 48,025,800 | +0.81% | 1472 | 3.83 | 230,323 |
| kda_sig_hybrid_lowrank | 47,265,270 | -0.78% | 1408 | 3.67 | 266,594 |
| ckda_shipped_hybrid_lowrank | 47,265,270 | -0.78% | 1408 | 3.67 | 266,509 |

## s124M: d 576, 18 layers, 9 heads, dense ffn 2304

| arm | params | delta vs dense | ffn | ffn/d | tok/s/GPU @ mbs cap |
|---|---:|---:|---:|---:|---:|
| attn | 124,547,904 | +0.00% | 2304 | 4.00 | 161,497 |
| attn_qknorm | 124,550,208 | +0.00% | 2304 | 4.00 | 161,497 |
| kda_sig_lowrank | 125,451,234 | +0.73% | 2240 | 3.89 | 132,105 |
| ckda_shipped_lowrank | 125,451,234 | +0.73% | 2240 | 3.89 | 128,073 |
| kda_sig_hybrid_lowrank | 124,144,574 | -0.32% | 2176 | 3.78 | 139,079 |
| ckda_shipped_hybrid_lowrank | 124,144,574 | -0.32% | 2176 | 3.78 | 132,361 |

## s302M: d 896, 20 layers, 14 heads, dense ffn 3584

| arm | params | delta vs dense | ffn | ffn/d | tok/s/GPU @ mbs cap |
|---|---:|---:|---:|---:|---:|
| attn | 302,010,240 | +0.00% | 3584 | 4.00 | 77,921 |
| attn_qknorm | 302,012,800 | +0.00% | 3584 | 4.00 | 77,921 |
| kda_sig_lowrank | 303,660,440 | +0.55% | 3520 | 3.93 | 67,145 |
| ckda_shipped_lowrank | 303,660,440 | +0.55% | 3520 | 3.93 | 64,802 |
| kda_sig_hybrid_lowrank | 302,961,170 | +0.31% | 3456 | 3.86 | 70,368 |
| ckda_shipped_hybrid_lowrank | 302,961,170 | +0.31% | 3456 | 3.86 | 69,782 |

## s588M: d 1280, 20 layers, 20 heads, dense ffn 5120

| arm | params | delta vs dense | ffn | ffn/d | tok/s/GPU @ mbs cap |
|---|---:|---:|---:|---:|---:|
| attn | 588,729,600 | +0.00% | 5120 | 4.00 | 50,420 |
| attn_qknorm | 588,732,160 | +0.00% | 5120 | 4.00 | 50,420 |
| kda_sig_lowrank | 586,324,880 | -0.41% | 4992 | 3.90 | 35,999 |
| ckda_shipped_lowrank | 586,324,880 | -0.41% | 4992 | 3.90 | 37,532 |
| kda_sig_hybrid_lowrank | 587,745,260 | -0.17% | 4928 | 3.85 | 41,315 |
| ckda_shipped_hybrid_lowrank | 587,745,260 | -0.17% | 4928 | 3.85 | 37,580 |

## s983M: d 1536, 24 layers, 24 heads, dense ffn 6144

| arm | params | delta vs dense | ffn | ffn/d | tok/s/GPU @ mbs cap |
|---|---:|---:|---:|---:|---:|
| attn | 983,311,872 | +0.00% | 6144 | 4.00 | 31,809 |
| attn_qknorm | 983,314,944 | +0.00% | 6144 | 4.00 | 31,809 |
| kda_sig_lowrank | 979,996,224 | -0.34% | 6016 | 3.92 | 27,225 |
| ckda_shipped_lowrank | 979,996,224 | -0.34% | 6016 | 3.92 | 25,646 |
| kda_sig_hybrid_lowrank | 984,364,080 | +0.11% | 5952 | 3.88 | 29,309 |
| ckda_shipped_hybrid_lowrank | 984,364,080 | +0.11% | 5952 | 3.88 | 27,347 |

## s1p7B: d 2048, 24 layers, 32 heads, dense ffn 8192

| arm | params | delta vs dense | ffn | ffn/d | tok/s/GPU @ mbs cap |
|---|---:|---:|---:|---:|---:|
| attn | 1,713,735,680 | +0.00% | 8192 | 4.00 | 19,774 |
| attn_qknorm | 1,713,738,752 | +0.00% | 8192 | 4.00 | 19,774 |
| kda_sig_lowrank | 1,709,707,520 | -0.24% | 8064 | 3.94 | 15,394 |
| ckda_shipped_lowrank | 1,709,707,520 | -0.24% | 8064 | 3.94 | 14,978 |
| kda_sig_hybrid_lowrank | 1,712,287,424 | -0.08% | 7936 | 3.88 | 17,229 |
| ckda_shipped_hybrid_lowrank | 1,712,287,424 | -0.08% | 7936 | 3.88 | 16,812 |
