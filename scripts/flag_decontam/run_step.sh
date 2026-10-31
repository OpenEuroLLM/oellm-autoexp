#!/bin/bash
# One step of the flag decontamination inside the training image on a MareNostrum 5 node
# (config/experiments/korbi/decontam/flag_decon_mn5.yaml runs it once per node under srun).
# docs/flag_post_decontamination.md has the order and what each step writes.
#
#   run_step.sh <step> [extra args for the step's script]
#     selftest  pytest of the pipeline on this node (CPU and GPU hashes agree)
#     build     reference table from $REFS
#     roundtrip tokenizer round trip on a few shards
#     count     n-gram counts over a sample of the ordinary (web / code) shards
#     prune     drop the n-grams the count found common
#     scan      the full scan of every shard of the mix (all nodes of the job share it)
#     recheck   recompute candidate pairs against corrected n-gram sets (no rescan)
#     ngrams    per-benchmark n-gram hit rates over the whole mix (_report_ngrams.tsv)
#     compare   our decisions vs an independently filtered shard (compare_pure.py; --orig / --pure / --out)
#     explain   their any-n-gram rule on our table for the compare groups (explain_pure.py; --orig / --cmp)
#     absrule   documents dropped only by the absolute-run rule (abs_rule_report.py)
#     ovl_refs  a sample of one dataset's documents as $REFS (doc_overlap.py refs; then build, scan --shards B)
#     ovl_report  how much of the sampled documents the scanned shards contain (doc_overlap.py report)
#     finalize  drop decisions + new tree in $OUT (FINALIZE_ARGS, e.g. --dry-run)
#     verify    new tree loads in Megatron, kept documents are unchanged
#     audit     recall of the token matcher vs. a text matcher (agenttrove, megamath; 100k docs each)
set -euo pipefail
step=$1; shift || true
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-/gpfs/projects/ehpc533/training-data-collection/flag}
WORK=${WORK:-/gpfs/scratch/ehpc533/users/korbip/flag_decon_v5}
OUT=${OUT:-/gpfs/projects/ehpc533/training-data-collection/flag-decon-v1}
REFS=${REFS:-$WORK/refs/refs.jsonl.gz}
mkdir -p "$WORK"
cd "$HERE/../.."

# shard lists: every shard of the mix; the count sample leaves out the sources most likely to
# hold benchmark copies (their copies must not make the benchmark's n-grams look "common")
t=.tmp.${SLURM_NODEID:-0}.$$                       # every node writes them: rename atomically
awk -v r="$ROOT" 'NF {print r "/" $2}' "$ROOT/datamix.txt" > "$WORK/shards_all.txt$t"
grep -v -E '/(agenttrove|dolmino|swallow-math|finemath|openwebmath|megamath|nemotron-pretraining-specialized|nemotron-mind|mixture-vitae|finephrase)[^/]*/' \
    "$WORK/shards_all.txt$t" > "$WORK/shards_count.txt$t"
grep -E '/(agenttrove|megamath|dclm)-' "$WORK/shards_all.txt$t" | awk -F/ '!seen[$7]++' > "$WORK/shards_roundtrip.txt$t"
for f in shards_all shards_count shards_roundtrip; do mv -f "$WORK/$f.txt$t" "$WORK/$f.txt"; done
echo "[$step] node ${SLURM_NODEID:-0}/${SLURM_NNODES:-1} on $(hostname), work $WORK"
case $step in
selftest)  python scripts/flag_decontam/test_flag_decontam.py ;;
build)     python scripts/flag_decontam/build_table.py --work "$WORK" --refs "$REFS" "$@" ;;
roundtrip) python scripts/flag_decontam/verify.py roundtrip --shards "$WORK/shards_roundtrip.txt" "$@" ;;
count)     # counts/ keeps one file per node and GPU: remove it before counting with a new table
           python scripts/flag_decontam/scan.py --mode count --work "$WORK" --root "$ROOT" \
               --shards "$WORK/shards_count.txt" --sample-tokens $((1 << 28)) "$@" ;;
prune)     python scripts/flag_decontam/build_table.py --work "$WORK" --prune "$@" ;;
scan)      python scripts/flag_decontam/scan.py --mode scan --work "$WORK" --root "$ROOT" \
               --shards "$WORK/shards_all.txt" "$@" ;;
recheck)   python scripts/flag_decontam/recheck.py --work "$WORK" --root "$ROOT" "$@" ;;
ngrams)    python scripts/flag_decontam/ngram_report.py --work "$WORK" "$@" ;;
compare)   python scripts/flag_decontam/compare_pure.py --work "$WORK" --root "$ROOT" "$@" ;;
explain)   python scripts/flag_decontam/explain_pure.py --work "$WORK" "$@" ;;
absrule)   python scripts/flag_decontam/abs_rule_report.py --work "$WORK" --root "$ROOT" "$@" ;;
ovl_refs)  python scripts/flag_decontam/doc_overlap.py refs --out "$REFS" "$@" ;;
ovl_report) python scripts/flag_decontam/doc_overlap.py report --work "$WORK" "$@" ;;
finalize)  python scripts/flag_decontam/finalize.py --work "$WORK" --root "$ROOT" --out "$OUT" "$@" ;;
verify)    python scripts/flag_decontam/verify.py tree --out "$OUT" --root "$ROOT" --procs 56 "$@" ;;
audit)     for s in agenttrove-0.0/megatron-lm/data/shard_0000 megamath-0.0.0/megatron-lm/shard_0000; do
               python scripts/flag_decontam/verify.py audit --work "$WORK" --root "$ROOT" --docs 100000 \
                   --shard "$ROOT/${s}_text_document" "$@"
           done ;;
*) echo "unknown step $step" >&2; exit 2 ;;
esac
echo "[$step] done"
