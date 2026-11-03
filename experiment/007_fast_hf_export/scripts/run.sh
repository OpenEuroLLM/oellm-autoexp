#!/bin/bash
set -euo pipefail
export HF_EXPORT_READER=fast HF_EXPORT_READ_THREADS=8 EXPERT_DTYPE=fp8
RESULT=$PWD/experiment/007_fast_hf_export/results
BASE=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700
CROOT=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports_fast007
mkdir -p "$RESULT"
python -m pytest --confcutdir=tests/unit -p no:cacheprovider -o addopts='' -q tests/unit/test_hf_export_reader.py 2>&1 | tee "$RESULT/unit.log"
python experiment/007_fast_hf_export/scripts/reader_benchmark.py 2>&1 | tee "$RESULT/reader_benchmark.log"
# Time the same export + shape checks + validation scoring + paired scoring as baseline.
/usr/bin/time -v -o "$RESULT/pipeline_time.txt" \
  bash scripts/korbi/hf_export/export_main_run.sh ambitious_v4 3700 "$CROOT" \
  2>&1 | tee "$RESULT/pipeline.log"
CAND=$CROOT/ambitious_v4_2ph_fp8_3700
# Baseline weights/config are immutable once export checks finish. Its scoring
# can continue while values are compared; no changes to the active eval pipeline.
for i in $(seq 1 240); do [ -f "$BASE/.check_done" ] && break; sleep 15; done
test -f "$BASE/.check_done"
python experiment/007_fast_hf_export/scripts/compare_exports.py "$BASE" "$CAND" \
  --out "$RESULT/value_comparison.json" 2>&1 | tee "$RESULT/value_comparison.log"
echo 'FAST_EXPORT007_VERIFIED'
