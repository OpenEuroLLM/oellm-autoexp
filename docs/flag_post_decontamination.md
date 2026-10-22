# Decontamination of the pre-tokenised "flag" mix on MareNostrum 5

Goal: remove from the flag training mix (`/gpfs/projects/ehpc533/training-data-collection/flag`,
452 shards, 54 TB of int32 `.bin`, ~13.5T tokens) every document that contains a benchmark item
of the evaluations we run (FLAG suite: `config/eval_tasks/flag_evals.yaml`, 438 evals over
lm-eval-harness, lighteval and Evalchemy, incl. the post-training ones: GSM8K, MATH500, AIME,
AMC, GPQA, HumanEval, MBPP, LiveCodeBench, IFEval, ...). Trigger: `agenttrove-0.0` contains
GSM8K and MATH500 although the collection README marks it decontaminated ("Contamination ✅");
the README marks the math / code sources as never checked ("➖").

Decisions (user, 2026-10-22): decontaminate against **all** benchmarks; **drop whole
documents** whose overlap is too high (measure the impact later); the original data stays as it
is, the result goes to a new folder **`/gpfs/projects/ehpc533/training-data-collection/flag-decon-v1`**.

## 1. Constraints found on the cluster

- **No CPU budget.** ehpc533 and cns144 have MareNostrum5 ACC hours only (`bsc_acct`: GPP
  "undefined"). The `gpdata` partition (10 data nodes, 224 cores, 2 TB) refuses our QOS
  ("Invalid qos specification"); `accinteractive` is one shared node (alogin3), queued for days;
  login nodes kill a process after **300 s of CPU time** (`ulimit -t 300`). So: GPU nodes,
  `acc_debug` (starts within minutes, <= 8 nodes, 1 job, 2 h). A quarter node (20 cores + 1
  GPU) is billed as 1 GPU-h.
- **The tokenizer allows matching on token ids.** `tokenizer-256k/tokenizer.model` is a
  SentencePiece BPE with `split_by_whitespace=True`, `split_digits=True`, byte fallback, identity
  normaliser, no dummy prefix: no piece spans a space, so a word tokenises identically wherever it
  stands except for its leading `▁` (space vs newline vs start). A lookup table that maps every id
  to the id of its piece with whitespace stripped and drops whitespace-only / control pieces gives
  a canonical stream in which n-gram matching equals word-sequence matching up to whitespace.
  No detokenisation of 13.5T tokens.
- **Megatron reads documents through their pointers.** `GPTDataset` reads each document with
  `IndexedDataset.get(doc, offset, length)` (`gpt_dataset.py:390-396` -> `indexed_dataset.py:848`),
  which seeks to `sequence_pointers[doc]`. An `.idx` listing only the kept documents, with their
  original pointers, works against the original `.bin`: the new tree is symlinks plus rewritten
  `.idx` files (<= 225 GB even if every shard lost a document; 190 TB free on projects).
  (Only slice reads `dataset[a:b]` require contiguous sequences; pretraining does not use them.)
- **Measured speed** (agenttrove shard, login node, one core): 3.0 GB/s cold single-stream GPFS
  read, 31 M tok/s numpy 13-gram hashing, 94 M tok/s bitmap lookup. On GPUs the scan is bound by
  the GPFS read.
- **Eval data is on JUPITER only** (MN5 has no internet and no eval datasets cached): the FLAG
  suite's HF cache `/e/scratch/e-sta-openeurollm/poeppel1/flag-evals/hf_home` (hub + `datasets`
  arrow cache, 24 GB) and the data Evalchemy / human-eval bundle inside the vLLM eval image
  (`/opt/evalchemy/eval/chat_benchmarks/*/data`, `/opt/human-eval/data`). Only the extracted
  text (132 MB) is moved. JUPITER's own key is not accepted by MN5 (`transfer1.bsc.es`:
  "Permission denied (publickey)"), so the copy needs the user's MN5 key on JUPITER (or agent
  forwarding of it).

## 2. Method

1. **References** (`scripts/flag_decontam/extract_refs.py`, JUPITER, inside the vLLM eval image):
   every string field of every row of every cached eval dataset (nested lists / dicts flattened)
   that looks like text or code (>= 4 words with letters, 25 .. 100k chars), deduplicated by
   exact text -> `refs.jsonl.gz`, one **segment** per line (`src`, `split`, `file`, `text`), plus
   `refs.jsonl.gz.manifest.tsv` (every file, its split, included or not, rows, errors).
   Splits: test / validation / dev / devtest / unknown included; `train` excluded when the source
   has other splits (few-shot pools and training sets are legitimate data), included when train
   is the only split (GPQA, multiblimp, ...). `cache-*.arrow` (`.map()` outputs) skipped.
2. **Table** (`build_table.py build`): segments tokenised with SentencePiece, canonicalised;
   distinct n-grams with n = 13 (n = 8 for segments of 8..19 canonical tokens; shorter ones are
   unmatchable and only counted). N-grams in more than 10 distinct segments are prompt / template
   text and dropped. Output: sorted int64 hashes + CSR hash -> segments.
3. **Background pruning** (`scan.py --mode count` + `build_table.py --prune`): count every table
   n-gram over the first 2^28 tokens of each shard of the ordinary sources (all but agenttrove,
   dolmino, swallow-math, finemath, openwebmath, megamath, nemotron-pretraining-specialized,
   nemotron-mind, mixture-vitae, finephrase, whose benchmark copies must not make a benchmark's
   n-grams look common); n-grams seen more than 50 times are common phrases and dropped.
4. **Scan** (`scan.py --mode scan`, all GPUs of all nodes): per unit of whole sequences
   (2^28 tokens) read the tokens, canonicalise on the GPU, hash the 8- and 13-gram windows that
   stay inside one document, `searchsorted` against the table; keep every (document, segment) pair
   with >= 3 distinct matched n-grams -> `$WORK/hits/<shard>/<first seq>.npz`. Resumable.
5. **Finalize** (`finalize.py`): a pair is contamination when the document contains
   >= max(3, ceil(0.5 * n_grams(segment))) of the segment's n-grams; a document with any such
   pair is dropped. Segments that would drop > 20000 documents mix-wide are reported
   (`generic_segments.tsv`) and not applied. Writes the tree (below) and the report.
6. **Verify** (`verify.py`): `roundtrip` (decode -> re-encode -> same canonical stream: the
   premise of token matching), `tree` (Megatron's `IndexedDataset` opens every rewritten shard,
   kept documents are token-identical, counts add up), `audit` (recall against a plain lower-cased
   word-13-gram text matcher on 5000 agenttrove documents, math / code / QA benchmarks).

Output tree `flag-decon-v1/` mirrors `flag/` for the 452 shards of `datamix.txt`:
`<prefix>.bin` -> symlink to the original; `<prefix>.idx` new (or a symlink when nothing is
dropped); `<shard>.info.json` with the new idx size / md5 / counts and a `decontamination` entry;
`<prefix>.removed.npz` (doc, seq, seg, matched, n_grams of every flagged pair); `datamix.txt`
(unchanged weights); `_report/` (report.json, sources.tsv, benchmarks.tsv, top_segments.tsv,
generic_segments.tsv).

Training on it: `scripts/korbi/flag_datamix_mn5.py --root .../flag-decon-v1 --out
$HYBRID_ENV/data/flag_decon_v1_datamix_mn5.txt --require-complete`, and a data config pointing
`data_args_path` there (step 9).

## 3. Scripts and how to run them

All in `scripts/flag_decontam/` (committed; edited locally only, synced with
`./sync_to_marenostrum.sh decon` to `$R/autoexp_decon`, never written on the cluster).
Every MN5 step goes through autoexp: `config/experiments/korbi/decontam/flag_decon_mn5.yaml`
runs `scripts/flag_decontam/run_step.sh <step>` once per node in the training image
(`MegatronTraining-hybrid-ep_x86_64_202610161948.sif`: torch, sentencepiece, pytest).

| file | what |
|---|---|
| `extract_refs.py` | JUPITER: benchmark text -> refs.jsonl.gz + manifest |
| `common.py` | canonical lookup table, n-gram hash, Megatron .idx read / write / filter |
| `build_table.py` | reference table (build) and background pruning (--prune) |
| `scan.py` | GPU scan: --mode count / scan, one worker per GPU, parts by SLURM_NODEID |
| `finalize.py` | thresholds, drop decisions, new tree, report (--dry-run: report only) |
| `verify.py` | roundtrip / tree / audit checks |
| `run_step.sh` | the step dispatcher used by the autoexp config |
| `test_flag_decontam.py` | synthetic end-to-end test (chunking, doc boundaries, whitespace twins, thresholds, idx rewrite, Megatron reads it) |

Local test: `CUDA_VISIBLE_DEVICES= python -m pytest scripts/flag_decontam/test_flag_decontam.py`
(4 passed, 2026-10-22; the local GPU is unusable until the driver reboot, so the CPU/GPU hash
agreement is checked by the `selftest` step on MN5).

Steps (R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019, WORK=/gpfs/scratch/ehpc533/users/korbip/flag_decon_v1):

```bash
# 0. JUPITER: extract the references (the script copied next to its output; same md5 as committed)
ssh jupiter
D=/e/scratch/e-sta-openeurollm/$USER/flag-decon; mkdir -p $D; cd $D   # extract_refs.py copied here
apptainer exec --cleanenv -B /e /e/project1/e-sta-openeurollm/container/oellm-eval-vllm.sif \
    python extract_refs.py --hf-home /e/scratch/e-sta-openeurollm/$USER/flag-evals/hf_home --out refs.jsonl.gz
# 1. JUPITER -> MN5. Directly (needs the MN5 key on JUPITER, or `ssh -A jupiter` with it loaded in
#    the local agent):
rsync -av --progress $D/refs.jsonl.gz* eit847786@transfer1.bsc.es:/gpfs/scratch/ehpc533/users/korbip/flag_decon_v1/refs/
#    or, as done on 2026-10-22 (132 MB: 14 s each way), relayed through the workstation:
scp jupiter:$D/refs.jsonl.gz jupiter:$D/refs.jsonl.gz.manifest.tsv ./refs/ && \
    rsync -a ./refs/ marenostrum_new:/gpfs/scratch/ehpc533/users/korbip/flag_decon_v1/refs/
#    sha256 on both sides: refs.jsonl.gz fdeceb5b868d9ffb..., manifest aa9b1aee69c77a31...
# 2. local: sync the code snapshot
./sync_to_marenostrum.sh decon
# 3..8. MN5 login node, in $R/autoexp_decon (one step per launch, each waits for the previous)
run() { scripts/korbi/mn5_autoexp_bg.sh decon_$1 scripts/run_autoexp.py \
          --config-name experiments/korbi/decontam/flag_decon_mn5 --yes aux.step=$1 "${@:2}"; }
run selftest
run build                        # tokenise + table (CPU, one node)
run roundtrip
run count aux.nodes=2 aux.time=01:00:00
run prune
run scan  aux.nodes=8 aux.time=02:00:00      # rerun to resume if it hits the wall time
run finalize 'aux.args="--dry-run"'          # report in $WORK/_report: check before writing
run finalize
run verify
```

## 4. Results

(filled in as the steps run)

All jobs: account ehpc533, QOS acc_debug, snapshot `$R/autoexp_decon`. GPU-h = nodes x 4 x elapsed.

| step | job | nodes x elapsed | GPU-h | result |
|---|---|---|---|---|
| extract (JUPITER login) | - | - | 0 | 2,406,959 unique segments from 4157 files, 132 MB (sha256 fdeceb5b...); 147 hub files with missing blobs (flores, xcopa pairs), covered by the arrow cache |
| selftest (1st) | 47820026 | 1 x 0:01 | 0.1 | failed: the image has no pytest -> tests run as a plain script |
| selftest | 47820072 | 1 x 0:01 | 0.1 | 4/4 passed on an H100 (CPU and GPU hashes agree) |
| build | 47820082 | 1 x 0:01:21 | 0.1 | lookup table 262144 ids -> 224382 canonical, 53 dropped; 82.3M (segment, n-gram) pairs, 64.5M unique n-grams, 115,807 template n-grams (> 10 segments) dropped; 1.53M segments at n = 13, 0.79M at n = 8, 80,325 too short (< 8 tokens: mostly multiple-choice options) |
| roundtrip | 47820487 | 1 x 0:01:51 | 0.1 | 900/900 documents (agenttrove, megamath, dclm) decode -> re-encode to the identical canonical stream |
| count (1st) | 47820821 | 2 x 0:02:24 | 0.3 | crashed: CUDA index assert on token id 262144 (`<pad>`, an added token outside the SentencePiece vocab: literal "<pad>" text) -> such ids are dropped and reported |
| count | 47821047 | 2 x 0:03:32 | 0.5 | 105.5B sample tokens (first 2^28 of each of 419 ordinary shards) in ~80 s per GPU (~165M tok/s/GPU); `<pad>` in 12 shards, 1-79 per 268M tokens |
| prune | (prune) | 1 x ~0:01 | 0.1 | n-grams seen > 50 times dropped: 51,995 (0.08 %); 64.45M kept. Histogram of sample counts: > 0: 936k, > 10: 148k, > 100: 32k, > 10000: 388. Matchable segments (>= 3 n-grams): 2,183,686 -> 2,182,710 (976 lost to pruning) |

## 5. Caveats

- **Pairing with old runs breaks.** With `split 998,2,0` the validation documents and the
  shuffle order depend on the document count: runs on `flag-decon-v1` are not paired with runs
  on `flag` (validation loss and data order differ). Switch at a clean boundary.
- Megatron rebuilds the dataset index cache for the new paths (> 10 min per run).
- Benchmarks whose items are ordinary web / wiki text (FLORES, OpenSubtitles, SIB-200,
  Belebele passages, SQuAD / CoQA contexts, HellaSwag WikiHow contexts) drop the copies of those
  pages too: by decision, all benchmarks count. `benchmarks.tsv` shows what each costs.
- Recall limits: segments of < 8 canonical tokens (short answers, COPA-style premises) are not
  matchable; reformatted copies (LaTeX respacing, paraphrases, translations into other
  languages) are not exact n-gram matches. The `audit` step measures recall against a text matcher.
- The collection's README marks agenttrove as decontaminated; tell its maintainer (bsc353976).
