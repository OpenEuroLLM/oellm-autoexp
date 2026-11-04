# Parallel HF export: < 2 min, byte-identical

## Objective
Export a v4_2ph (mHC) checkpoint to HF in < 3 min (v2 sequential path with the fast reader: ~8 min, 006/007).

## Method
`oellm_autoexp.hf_export.export_parallel`: the parent reads the DCP metadata once (8 s), plans 355 work units
(98 layers + non-layer + 256 Engram-table row ranges of ~1 GiB), and forks 32 worker processes (largest first).
Each worker reads its chunks directly (offset/length from the metadata, 4 threads per worker), converts with the
shared `export_ambitious.convert_layer` / `convert_nonlayer` (refactored out of `convert`; experts now
converted one expert at a time, same arithmetic), and writes its own safetensors file; Engram tables go into one
preallocated file per table via `pwrite`. FP8 experts: same `fp8_tiles` (E4M3, 1x128 fp32 scales) on CPU.
No GPU needed: the job is I/O bound (654 GB fp32 in, 248 GB out).

## Results (run o134c.0026, as01r2b04, only 16 CPUs in the step)
| checkpoint | wall (process start to end) | in-python | read rate |
|---|---:|---:|---:|
| 5800 | 115.9 s | 107.2 s | 7.8 GB/s |
| 5900 | 102.3 s | 100.0 s | 7.7 GB/s |

- 5800 vs the existing v2 export (hf_exports/ambitious_v4_2ph_fp8_5800, itself bit-verified vs the 006 baseline):
  **1,949 tensors / 248,189,032,600 bytes bitwise equal**, all config / tokenizer / model-code files equal (compare_raw.py, 31 s).
- 5900 written to `/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_5900` (.export_done, .check_done; check_export clean).
  Score / paired validation not run (export_main_run_v2.sh would do them, skipping the export).
- Parent max RSS 9.3 GiB. Shard layout differs from the sequential exporter (101 files), tensors identical.

## Usage
`HF_EXPORTER=parallel` in `scripts/korbi/hf_export/export_main_run_v2.sh`, or directly
`python -m oellm_autoexp.hf_export.export_parallel --ckpt-dir ... --config ... --out-dir ... --tokenizer ... --expert-dtype fp8`.
More CPUs in the step (kernel_tests default 16) may raise the rate further.

Artifacts: results/run1/.
