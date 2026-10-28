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

## 2. Method (final: v5 references, v3 matcher, 70 % rule)

The history of how this was arrived at (v1 n-gram coverage -> v2 longest runs -> v3 word-level
stream -> v4 / v5 more benchmarks) is in section 4; this section describes what produced
`flag-decon-v1`.

1. **References** (`extract_refs.py`, JUPITER, inside the vLLM eval image). Every string field of
   every row of every eval dataset (nested lists / dicts flattened; lists of >= 8 one-word strings
   joined: WikiAnn-style token lists) that looks like text or code (>= 4 words with letters,
   25 .. 100k chars), deduplicated by exact text -> one **segment** per line. Sources: the FLAG
   suite's HF cache, the data bundled in the Evalchemy / human-eval image, and `hf_extra`
   (`fetch_extra.py`: oellm-eval groups outside FLAG, the external table's benchmarks, the gated
   sets). Splits: test / validation / dev / devtest / unknown; `train` only when it is a source's
   only split. `--append-to` extends a refs file keeping segment ids; `--attribute` writes every
   (source, split, file, row, field) of each segment plus an items file (segments by field, short
   fields: answers, labels, options).
2. **Table** (`build_table.py`). Segments tokenised with the training SentencePiece model and mapped
   to the **word-level canonical stream** (`common.build_lut(words=True)`: each piece lower-cased,
   non-alphanumerics removed, pieces without letters / digits dropped). Segments need >= 20 canonical
   tokens and >= 80 characters. Per segment n = the smallest of 13 / 24 / 40 whose n-grams span >= 30
   characters (byte-fallback scripts get long n-grams). The JSON-escaped form of each segment
   (`\n`, `\"`; and `\uXXXX` only for texts with <= 3 non-ASCII characters) is hashed under the same
   segment id. N-grams in > 10 distinct segments (templates) are dropped.
3. **Background pruning** (`scan.py --mode count`, `build_table.py --prune`): n-grams seen > 50 times
   in the first 2^28 tokens of each of the 419 ordinary shards (105.5B tokens; the suspect math /
   synthetic sources excluded) are common phrases and dropped. Prune once per build.
4. **Scan** (`scan.py --mode scan`, 8 nodes x 4 GPUs, ~1 h): per unit of whole sequences (2^28
   tokens) canonicalise on the GPU, hash the 13 / 24 / 40-gram windows inside one document, look
   them up (`searchsorted`), map each matching window to its segments and compute per (sequence,
   segment) the **longest run** of consecutive matching windows (gaps <= n // 2 bridged; a changed
   word splits the run). Stored: pairs with runs >= 20 tokens, plus the distinct table n-grams each
   unit contains. Token ids outside the vocabulary (`<pad>` = 262144, literal text) are dropped and
   reported.
5. **Finalize** (`finalize.py --frac 0.7`): a document is dropped when one of its sequences has a
   segment whose longest run covers >= 70 % of the segment's canonical tokens, or is >= 200 tokens
   long, and is >= 20 tokens. Segments that would drop > 20,000 documents are held back as generic
   (one AlpacaEval prompt). Reports: per source, per benchmark (one / all attributions), by field
   role, top segments.
6. **Verify** (`verify.py`): `tree` (Megatron's `IndexedDataset` opens every rewritten shard, kept
   documents token-identical, counts add up), `roundtrip` (decode -> re-encode -> same canonical
   stream), `audit` (recall against a JSON-unescaping word-level text matcher on 100k documents of
   agenttrove and megamath); `ngram_report.py` (per-benchmark n-gram hit rates, comparable to
   n-gram-overlap tables); `inspect_hits.py` (samples of matches: segment next to the document text).

Output tree `flag-decon-v1/` mirrors `flag/` for the 452 shards of `datamix.txt`: `<prefix>.bin` ->
symlink to the original; `<prefix>.idx` new (or a symlink when nothing is dropped); `<shard>.info.json`
with the new idx size / md5 / counts and a `decontamination` entry; `<prefix>.removed.npz` (doc, seq,
seg, run, windows, runs, n_tokens of every flagged pair); `datamix.txt` (unchanged weights); `_report/`.
Training: `config/backend/megatron/data_flag_decon_v1_mn5.yaml` with
`$HYBRID_ENV/data/flag_decon_v1_datamix_mn5.txt` (`scripts/korbi/flag_datamix_mn5.py --root
.../flag-decon-v1 --out ... --require-complete`).

## 3. Scripts and how to run them

All in `scripts/flag_decontam/` (committed; edited locally only, synced to MN5 snapshots with
`./sync_to_marenostrum.sh <name>`, never written on the cluster). Every MN5 step goes through
autoexp: `config/experiments/korbi/decontam/flag_decon_mn5.yaml` (`aux.version`, `aux.work`,
`aux.refs`, `aux.out`) runs `scripts/flag_decontam/run_step.sh <step>` once per node in the training
image (`MegatronTraining-hybrid-ep_x86_64_202610161948.sif`).

| file | what |
|---|---|
| `fetch_extra.py` | JUPITER: download benchmarks outside the FLAG cache (lm-eval tasks, Hub ids, parquet branches, files) |
| `extract_refs.py` | JUPITER: benchmark text -> refs (+ manifest); `--append-to`; `--attribute` (all sources, fields, items) |
| `common.py` | word-level lookup table, n-gram hash, longest runs, Megatron .idx read / write / filter |
| `build_table.py` | reference table (build) and background pruning (--prune) |
| `scan.py` | GPU scan: --mode count / scan, one worker per GPU, parts by SLURM_NODEID, resumable |
| `recheck.py` | recompute candidate pairs against corrected n-gram sets without a rescan (v2 fix; overlay file) |
| `finalize.py` | drop decisions, tree, reports (--dry-run: several --frac values, reports only) |
| `ngram_report.py` | per-benchmark n-gram hit rates (--by-config) |
| `inspect_hits.py` | sample matches: benchmark segment vs. the document around its longest run |
| `compare_pure.py` | our decisions vs an independently filtered shard (step `compare`; section 4.7) |
| `explain_pure.py` | their any-n-gram rule on our table per compare group + samples (step `explain`) |
| `abs_rule_report.py` | documents dropped only by the >= 200-token branch, by source and item (step `absrule`) |
| `verify.py` | tree / roundtrip / audit checks |
| `run_step.sh` | the step dispatcher used by the autoexp config |
| `test_flag_decontam.py` | synthetic end-to-end test (runs without pytest; `CUDA_VISIBLE_DEVICES= python ...` locally) |

Steps (R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019, D=/e/scratch/e-sta-openeurollm/$USER/flag-decon on JUPITER):

```bash
# JUPITER login node (internet), in $D with fetch_extra.py, extract_refs.py and oellm-eval's
# custom_lm_eval_tasks copied there; S = the vLLM eval image, H1 = the FLAG suite's HF cache
S=/e/project1/e-sta-openeurollm/container/oellm-eval-vllm.sif; H1=/e/scratch/e-sta-openeurollm/$USER/flag-evals/hf_home
apptainer exec --cleanenv -B /e --env HF_HOME=$D/hf_extra --env HF_TOKEN_PATH=$HOME/.cache/huggingface/token \
    $S python fetch_extra.py --oellm-tasks $D/custom_lm_eval_tasks
apptainer exec --cleanenv -B /e $S python extract_refs.py --hf-home $H1 --out refs.jsonl.gz      # first time
apptainer exec --cleanenv -B /e $S python extract_refs.py --hf-home $H1 --hf-home $D/hf_extra \
    --append-to refs_v4.jsonl.gz --out refs_v5.jsonl.gz                                          # extensions
apptainer exec --cleanenv -B /e $S python extract_refs.py --hf-home $H1 --hf-home $D/hf_extra \
    --attribute refs_v5.jsonl.gz --out refs_v5.attrib.jsonl.gz
# JUPITER -> MN5 (relayed through the workstation; check sha256 on both sides)
scp jupiter:$D/refs_v5* ./refs_v5/ && rsync -a ./refs_v5/ marenostrum_new:/gpfs/scratch/ehpc533/users/korbip/flag_decon_v5/refs/
# local: code snapshot
./sync_to_marenostrum.sh decon10
# MN5 login node, in $R/autoexp_decon10 (one step per launch, each after the previous finished)
run() { scripts/korbi/mn5_autoexp_bg.sh decon5_$1 scripts/run_autoexp.py \
          --config-name experiments/korbi/decontam/flag_decon_mn5 --yes aux.step=$1 "${@:2}"; }
run selftest
run build aux.time=00:45:00
run count aux.nodes=2 aux.time=01:00:00
run prune
run scan aux.nodes=8 aux.time=02:00:00          # rerun to resume if it hits the wall time
run ngrams aux.time=00:45:00
run finalize 'aux.args="--dry-run --frac 0.7,0.8,0.9,1.0"'      # $WORK/_report/frac_*/: check first
run finalize 'aux.args="--frac 0.7"' aux.out=/gpfs/projects/ehpc533/training-data-collection/flag-decon-v1.new aux.time=02:00:00
run verify aux.out=/gpfs/projects/ehpc533/training-data-collection/flag-decon-v1.new aux.time=02:00:00
run audit aux.time=01:30:00
# swap in (login node), then the datamix file
T=/gpfs/projects/ehpc533/training-data-collection; mv $T/flag-decon-v1 $T/flag-decon-v1.old && mv $T/flag-decon-v1.new $T/flag-decon-v1
python3 scripts/korbi/flag_datamix_mn5.py --root $T/flag-decon-v1 --out $R/data/flag_decon_v1_datamix_mn5.txt --require-complete
# inspection, any time (login node, ~30 s): samples of matches
singularity exec --cleanenv -B /gpfs ../container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif \
    python scripts/flag_decontam/inspect_hits.py --shard <shard key substring> --n 10 [--min-cov 0.7] [--bench gsm8k]
```

## 4. Results and history

v1 (n-gram coverage) -> v2 (longest runs) -> v3 (word-level stream) -> v4 / v5 (more benchmarks). All jobs: account ehpc533, QOS acc_debug, snapshot `$R/autoexp_decon`. GPU-h = nodes x 4 x elapsed.

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
| scan | 47821245 | 8 x 0:42:44 | 22.8 | 13.55T tokens, ~175M tok/s/GPU; 50,718 unit files, 311M (sequence, segment) pairs with >= 3 n-grams (7.46 GB); mixture-vitae shards 3+4: 55 % of them |
| finalize dry run (1st) | 47823965 | 1 x 0:47 | 3.1 | cancelled: per-pair Python loops over 310M pairs -> vectorised |
| finalize dry run | 47825946 | 1 x ~0:20 | 1.3 | --frac 0.2 / 0.5 / 0.8: 9.28M / 3.95M / 1.48M docs, 96.3B / 44.1B / 16.2B tokens (0.71 / 0.33 / 0.12 %); 45 / 8 / 2 generic segments held back |

### 4.1 Inspection of the matches (inspect_hits.py, 2026-10-23): the v1 matcher over-flags

```bash
cd $R/autoexp_decon_inspect   # login node, ~30 s per call
singularity exec --cleanenv -B /gpfs ../container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif \
  python scripts/flag_decontam/inspect_hits.py --shard <shard key substring> --n 10 [--min-cov 0.5] [--bench gsm8k]
```

- **Scattered matches.** Coverage counts a segment's n-grams anywhere in the document. In
  byte-fallback scripts (Tamil, Georgian, Telugu, Hindi) 13 tokens are ~4 characters, and a
  40k-100k-token Wikipedia article "contains" 50-70 % of a short xcopa / include-base-44 sentence
  spread over tens of thousands of tokens. Most of mixture-vitae's 170M pairs, and the long-document
  sources' drops at 0.5 (finepdfs 1.68 % of its tokens, finepdfs-edu 1.34 %, hplt 0.32 %), come from this.
- **Short generic segments.** Global-MMLU / ARC answer options ("la dure-mère et l'arachnoïde.")
  and list fragments (`['Yes', 'Yes', 'No', 'Yes']`) match ordinary text and code (Global-MMLU:
  1.88M pairs at 0.5).
- **Real copies.** SQuAD (100 % of segments found) and CoQA (85 %) passages (Wikipedia); an
  OpenBookQA question quoted in a paper (agenttrove); MATH500 / HumanEval / MBPP items (33-48 % of
  segments found somewhere).
- **agenttrove: no verbatim GSM8K** (6 incidental pairs at coverage <= 0.03), 22 MATH500 pairs at
  >= 0.5 of 577 such pairs in the shard. Either rewritten variants (a sampled MATH problem with
  changed numbers) or copies inside JSON-escaped text (`\n`, `\"`, `’`) the token matcher misses.

MATH500 (asked 2026-10-23): found three ways -- agenttrove holds *modified* test problems inside
agent templates ("angle BAC" -> "exterior angle BAC", cos -> sin, a changed polynomial, the full
`[asy]` code of a solution); megamath holds verbatim contest problems from the web ("Four distinct
circles are drawn in a plane...", coverage 1.0); short MATH500 expressions (`x^8 + x^7 + ... + 1`)
match ordinary algebra pages (false positives).

### 4.2 v2 matcher (user, 2026-10-23: go ahead; "re-phrases and modified questions (cos -> sin)
should probably not be filtered out, although long 1:1 matches are bad"; GSM8K in agenttrove
was word of mouth)

- **Score = the longest verbatim run** of a segment in a sequence (`common.runs`): consecutive
  matching windows, gaps of <= n // 2 windows bridged (pruned common n-grams inside a copied
  passage); a changed word removes n consecutive windows and splits the run. Scattered n-grams
  are many short runs. scan.py stores (seq, seg, longest run in canonical tokens, matching
  windows, runs) for runs >= 30 tokens.
- **Drop rule** (finalize.py): longest run >= --frac (default 0.9) of the segment's canonical
  tokens, or >= --abs-run (200) tokens (a long 1:1 excerpt of a passage), and >= 30 tokens.
  Modified / rephrased items stay.
- **n by characters:** the smallest of 13 / 24 / 40 whose n-grams span >= 30 characters
  (median; a byte-fallback piece counts 1/3 character), so Tamil / Georgian / Telugu text gets long
  n-grams instead of 4-character ones.
- **Segments >= 20 canonical tokens and >= 80 characters** (no n = 8 path): answer options,
  short premises and list fragments are not matched alone.
- **JSON-escaped variants** of every segment (`json.dumps`, with and without `ensure_ascii`)
  hashed under the same segment id: copies inside agent trajectories (`\n`, `\"`, `’`).
- **Audit** (`verify.py audit`): text matcher on JSON-unescaped, lower-cased words with the same
  rule (longest common word run >= 0.9 of the segment or >= 150 words), against the token flags.
- Work dir `flag_decon_v2` (v1 kept in `flag_decon_v1`), same refs; snapshot `$R/autoexp_decon2`;
  job logs under `.../flag_decon/v2/<step>`. Tests: `test_runs`, `test_variants`, and planted
  verbatim (dropped) / one-word-modified (kept) / scattered (kept) / 220-of-300-token passage
  excerpt (dropped via --abs-run) cases in `test_end_to_end`.

| step (v2) | job | nodes x elapsed | GPU-h | result |
|---|---|---|---|---|
| selftest | (decon2_selftest) | 1 x ~0:01 | 0.1 | 6/6 passed on an H100 |
| build (1st) | 47851767 | 1 x 0:43 | 2.9 | cancelled: one multithreaded sp.encode call per segment oversubscribed the node -> one single-threaded call per batch |
| build | 47854517 | 1 x 0:02:38 | 0.2 | 1.40M matchable segments (1.31M n=13, 81k n=24, 9.6k n=40), 1.0M too short (357k Global-MMLU options, m_hellaswag endings, subtitles, ARC options); 130.5M n-grams (escaped variants), 519k template n-grams dropped |
| count + prune | 47854719 / 47855122 | 2 x ~0:04 / 1 x ~0:01 | 0.6 | 424k of 130M n-grams seen in the 105B sample, 10.5k dropped (> 50), 1 segment made unmatchable |
| scan | 47855150 | 8 x 1:01:39 | 32.9 | 13.55T tokens, ~120M tok/s/GPU (3 n, 2x table); 14.5M pairs with runs >= 30 tokens (0.58 GB) |
| dry run | 47859725 | 1 x 0:06:55 | 0.5 | --frac 0.7/0.8/0.9/1.0: 4.95B / 4.38B / 3.73B / 3.19B tokens; stackv2 2.79B of the 3.73B at 0.9 -> sampled: `\uXXXX` runs |
| recheck (1st, 2nd) | 47860904 / 47863568 | 1 x 0:31 / 1 x 1:00 | 6.1 | cancelled / timed out: per-segment rehash, then rewriting 50k unit files on GPFS (0.4 s each) -> one overlay file |
| recheck | 47867711 | 1 x 0:11:35 | 0.8 | 793k candidate pairs (>= 0.7 or >= 200 tokens); 401k with a changed segment rechecked, 346k of them fall below 30 tokens (escape false positives), 55k stay candidates |
| dry run (fixed) | 47869067 | 1 x ~0:05 | 0.3 | **0.7/0.8/0.9/1.0: 229k/175k/144k/93k docs, 2.86B/2.44B/1.91B/1.52B tokens (0.021/0.018/0.014/0.011 %)** |
| audit | 47869329 | 1 x 0:04:29 | 0.3 | 100k docs each: agenttrove 9/17 text-flagged docs also token-flagged, megamath 3/17 (see below) |

Corrected per-source drops at 0.9 (all classes, before the answer-aware policy): stackv2 1.8k docs /
1.02B tokens (0.14 %: GitHub repos shipping benchmark data files -- a BoolQ TSV, ARC data, MultiPL-E
prompts, Universal Dependencies treebanks behind multiblimp, up to 1.5M tokens each), dclm 43k / 231M
(0.006 %), finepdfs 2.0k / 171M, hplt 15k / 134M, mixture-vitae 35k / 82M (0.019 %), nemotron-cc
19k / 73M, megamath 14k / 72M (0.023 %), finemath 3.1k / 13M (0.034 %), openwebmath 0.8k / 7M
(0.05 %), agenttrove ~170 docs / ~2.7M (~0.013 %).

Audit misses, inspected: **megamath -- real**: formatting variants (LaTeX `\[...\]` vs `$...$`,
`$a,b,c,d,$` vs `$a$, $b$`, code `f( num)` vs `f(num)`, re-wrapped lists) change the token
sequence; and segments of 20-29 canonical tokens can never flag (the scan stores runs >= 30, the rule
requires >= 30 -- a bug: an MMLU question of 24 tokens copied with its answer). **agenttrove --
mostly text-matcher false alarms**: generic code (`import numpy as np ...` against a 49k-token
HumanEvalPlus test dump via the 150-word rule; the 15-word `balance = 0 / for op in operations` loop).

Attribution (extract_refs.py --attribute, JUPITER, 2026-10-23): 687,819 of 2,406,959 segments occur
in more than one benchmark; 1,908,405 items (rows with their segments by field and their short
fields); no included text missing from the refs. refs.attrib.jsonl.gz (sha256 012a4653...),
refs.attrib.items.jsonl.gz (42efdad4...) next to the refs on MN5.

Proposed v3 (pending the user): word-level canonical stream (lowercase, punctuation stripped from
pieces, pure-punctuation pieces dropped), runs >= 20 tokens stored and flaggable, the escape fix in
the build, then answer-aware classes (A answer/solution, B question + answer nearby, C question
only, D context only, E completion-style gold continuation; drop A/B/E?) and a normalized-code check
on code candidates; audit with a tightened text rule.

### 4.3 GSM8K in agenttrove, and how v1's inspection missed it (2026-10-23)

v2 scan, whole agenttrove shard, longest verbatim run >= 0.9 of the question: **48 GSM8K test
questions in 48 documents** (43 credited to CohereLabs/global-mgsm `en/test`, 5 to Qwen/PolyMath
`en`), none credited to `openai/gsm8k` (its other ~1,070 test questions and all solutions: 0).
They sit in "Solve the following math problem step by step and write the final answer to
answer.txt" tasks. All fall in the MGSM / PolyMath subset: agenttrove likely took its math tasks
from those GSM8K-derived sets. megamath holds many more verbatim GSM8K test questions
("Question: ... Answer: ..." pages, credited to openai/gsm8k).

The v1 report "no verbatim GSM8K in agenttrove" was wrong, for three reasons:
1. `extract_refs.py` dedupes identical texts and credits each to the first source in alphabetical
   file order; MGSM-en is GSM8K's first 250 test questions verbatim and `CohereLabs/...` sorts
   before `openai/...`, so those questions are stored as MGSM only.
2. `inspect_hits.py --bench gsm8k` filters by source *name*, so MGSM / PolyMath never showed.
3. A negative claim without a positive control: the printed v1 summary listed `global-mgsm` pairs
   at >= 0.5 and was not connected to GSM8K.

Rules from this: before reporting a benchmark absent, search a known item of it and its derived
benchmarks (MGSM-en ⊂ GSM8K test, MATH500 ⊂ MATH test, Global-MMLU-en ≈ MMLU, ARC-MT ⊂ ARC,
PolyMath partly GSM8K); query by content or by all attributions. Planned: re-extract with an
`also_in` list of every source of a segment (same line order, so table and scan stay valid) and
report per benchmark family.

### 4.4 v3 (word-level canonical stream, runs >= 20) and the external n-gram table (2026-10-23)

| step (v3) | job | nodes x elapsed | GPU-h | result |
|---|---|---|---|---|
| selftest / build | 47884202 / 47884279 | 1 x ~0:02 each | 0.3 | LUT 262144 ids -> 186,434 word keys, 3,015 punctuation-only pieces dropped; 1.34M matchable segments (1.30M n=13, 23k n=24, 9.4k n=40), 1.07M too short; 58.5M n-grams (escaped variants mostly collapse), 61k template n-grams dropped |
| count / prune | 47884403 / 47884421 | 2 x ~0:04 / 1 x ~0:01 | 0.6 | 336k n-grams seen in the 105B sample, 8.2k dropped (> 50), 2 segments made unmatchable |
| scan | 47884465 | 8 x 0:53:50 | 28.7 | 13.55T tokens, ~141M tok/s/GPU; 3.36M of 58.5M table n-grams occur in the mix |
| ngrams | 47888458 | 1 x short | 0.1 | `_report_ngrams.tsv`, `_report_ngrams_by_config.tsv` (all attributions) |
| dry run | 47889027 | 1 x short | 0.3 | **--frac 0.7/0.8/0.9/1.0: 430k/326k/274k/183k docs, 5.98B/4.83B/3.79B/2.96B tokens (0.044/0.036/0.028/0.022 %)** -- about 2x v2 (formatting variants and 20-29-token questions now found) |

Cross-match with the n-gram overlap table another group shared (`dump/image23.png`, OCR in
`docs/flag_decon/external_ngram_overlap_image23.tsv`; its n-gram size and corpus are not stated):
`docs/flag_decon/crossmatch_image23_v3.tsv`. 14 of its 40 benchmarks are in our suite (the other 26 --
RACE, MathQA, TriviaQA, NQ, MMLU-Pro/Redux, PubMedQA, SciQ, SVAMP, AQuA-RAT, AlpacaEval, AdvBench,
MedMCQA, WikiAnn, Tatoeba, MultiLoKo, DeepSeek LeetCode, the Spanish / Catalan / French / Basque bench
parts, SciRIFF, NorIdiom -- are not in our references and not decontaminated). Agreement: the same
benchmarks top both lists (BoolQ, WSC); HellaSwag (17.1 vs 19.3 %) and LiveCodeBench (29.2 vs 27.6 %)
nearly match. Ours are higher for ARC (13 vs > 99 %), SocialIQA (21 vs 99 %), MMLU (12 vs 30 %),
Include (12 vs 28 %), MBPP (10 vs 30 %), HumanEval (13 vs 31-71 %), because (1) we count only matchable
segments (>= 20 word tokens, >= 80 chars): for ARC / SocialIQA the long items only, which are the ones
copied; (2) the whole flag mix incl. GitHub repos that ship benchmark data files; (3) our 13-gram is
over word fragments (~8-10 words) and case / punctuation-insensitive. HumanEvalPlus is lower (1.6 %)
because its segments include the huge Plus test-input dumps. LiveCodeBench by release batch shows the
expected time pattern: 38.7 % (v1) .. 6.4 % (v6) of n-grams hit. The stronger measure -- matchable
items present >= 90 % verbatim somewhere: ARC 99.95 %, SQuAD 99.7 %, SocialIQA 97.9 %, WSC 97.6 %,
CoQA 84.5 %, BoolQ 70.9 %, HumanEval (openai) 58.4 %, GSM8K 54.3 %, MBPP 47-49 %, AIME24 48.3 %,
MATH500 28.7 %, MMLU 26.8 %, LiveCodeBench 15.6 %, HellaSwag 13.4 %, Include 9.3 %.

### 4.5 v4: benchmarks outside the FLAG suite (user, 2026-10-24: "we should also include the other benchmarks")

Sources: (1) every oellm-eval task group outside FLAG (`oellm/resources/task-groups.yaml`: pawsx-eu,
xnli-eu, mmlu-prox-eu, mmmlu-eu, generic-multilingual = xwinograd / xstorycloze / all-language xcopa,
belebele-eu-cf, the v0.02 reference corrections over gsm8k / squad_v2 / gpqa); (2) the 26 benchmarks of
the external table we did not have. Fetched on the JUPITER login node (internet) into a separate cache
`/e/scratch/e-sta-openeurollm/poeppel1/flag-decon/hf_extra` (14 GB) with `scripts/flag_decontam/fetch_extra.py`
(lm-eval task names where lm-eval has them -- `get_task_dict` with oellm-eval's custom task dir --,
else Hub ids; the Spanish / Catalan / French / Basque bench groups subtask by subtask; script datasets
from the Hub's parquet branch; DeepSeek-Coder's LeetCode contest file from GitHub; AlpacaEval's json):

```bash
D=/e/scratch/e-sta-openeurollm/$USER/flag-decon; cd $D   # fetch_extra.py, extract_refs.py, oellm-eval's custom_lm_eval_tasks copied here
apptainer exec --cleanenv -B /e --env HF_HOME=$D/hf_extra --env HF_TOKEN_PATH=$HOME/.cache/huggingface/token \
    /e/project1/e-sta-openeurollm/container/oellm-eval-vllm.sif python fetch_extra.py --oellm-tasks $D/custom_lm_eval_tasks
H1=/e/scratch/e-sta-openeurollm/$USER/flag-evals/hf_home
apptainer exec ... python extract_refs.py --hf-home $H1 --hf-home $D/hf_extra --append-to refs.jsonl.gz --out refs_v4.jsonl.gz
apptainer exec ... python extract_refs.py --hf-home $H1 --hf-home $D/hf_extra --attribute refs_v4.jsonl.gz --out refs_v4.attrib.jsonl.gz
```

Fetch result (3 rounds): 133 lm-eval tasks and 372 Hub dataset configs; 101 of 110 language-bench
subtasks. Not fetched: **gated** -- walledai/AdvBench, facebook/multiloko, Helsinki-NLP/tatoeba_mt
(eng-nor), gplsi/cocoteros_va, gplsi/truthfulqa_va (need access approval for the JUPITER HF token);
**not found** -- NorIdiom (no Hub id known); covered otherwise -- lm-eval paws_es / xnli_es (Hub ids
paws-x / xnli), xcopa / medmcqa / mmlu_redux / pubmedqa (Hub ids), the v0.02 tasks (their data).
Not used: a full LeetCode dump (greengerong/leetcode, all problems -- the benchmark is the contest
file). refs_v4: 5,818,224 segments = the 2,406,959 of refs.jsonl.gz unchanged (same ids) + 3,411,265
new (largest: MMLU-ProX 1.39M, WikiAnn 468k, FLORES all 200 languages 323k -- the language-bench
groups load it --, MMMLU 275k, SciRIFF 210k, XNLI 125k, XStoryCloze 100k, PAWS-X, MMLU-Pro, EusExams,
RACE, TriviaQA, XLSum-es, AlpacaEval, MathQA, CoQCat, MedMCQA, ARC-ca / -eu, OrangeSum, wikitext-fr,
EusProficiency, ...); 2,829,542 in more than one benchmark; 5,448,773 items. On MN5 in
`flag_decon_v4/refs/` (sha256 refs_v4 f254da37..., attrib 690cd150..., items 4b796887...). Note:
wikitext-fr, XLSum, OrangeSum, WikiAnn and FLORES are Wikipedia / news text: decontaminating against
them also drops copies of those articles (the report shows the cost per benchmark).

v4 run (snapshot `$R/autoexp_decon9`, work `flag_decon_v4`): build 47917924 -- 5,818,224 segments,
2,824,237 matchable (2.28M n=13, 165k n=24, 384k n=40), 2.99M too short, 365.1M n-grams, 679k template
n-grams dropped; count / prune 47918121 / 47918166 -- 2.81M n-grams seen in the sample, 22k dropped,
175 segments made unmatchable; scan 47918178 (8 nodes, ~1 h, ~127M tok/s/GPU); ngrams 47920439;
dry run 47921047: **--frac 0.7/0.8/0.9/1.0: 1.42M/1.20M/1.04M/0.84M docs, 17.6B/13.5B/11.5B/9.7B
tokens (0.130/0.100/0.085/0.072 %)** -- 3x v3, from the added benchmarks: SciRIFF (373k doc-segment
pairs), MathQA (280k; verbatim on aptitude / worked-solution pages in megamath -- the problems come
from GMAT/GRE banks via AQuA-RAT), WikiAnn (238k; Wikipedia sentences), SciQ (52k), AQuA-RAT (34k),
XLSum-es (33k), TriviaQA (26k), MMLU-Pro (21k). One AlpacaEval segment held back as generic (would
drop 44k docs). Per source at 0.9: stackv2 7.0k docs / 3.29B tokens (0.46 %), megamath 206k / 2.29B
(0.72 %), hplt 234k / 1.48B, dclm 239k / 1.29B, finepdfs 12k / 1.05B, nemotron-cc 100k / 0.97B,
mixture-vitae 135k / 0.45B, finemath 26k / 0.13B (0.33 %), finewiki 28k / 84M (0.34 %), openwebmath
7.6k / 67M (0.48 %). By role (0.9): answer 200k docs / 2.98B tokens, tests 5.4k / 0.96B, question
383k / 4.12B, options 81k / 0.74B, context 69k / 0.42B, other (field not recognised: e.g. WikiAnn's
joined token lists) 303k / 2.31B. Cross-match of all 40 benchmarks of the external table:
`docs/flag_decon/crossmatch_image23_v4.tsv` (34 fetched; AdvBench, MultiLoKo, Tatoeba gated, NorIdiom
not found, COPA too short).

The v4 report adds `roles.tsv`: dropped documents / tokens by the most severe field role of their
flagged segments (answer > tests > question > options > context > other, from the field paths of the
attribution), as the basis for the answer-aware policy.

### 4.6 v5 and the written tree (2026-10-24)

v5 = v4 + the datasets fetched after access approval: AdvBench, MultiLoKo (all languages), NorIdiom
(`mimir-project/noridiom`, `Sprakbanken/Norwegian_idioms`), the Valencian catalan_bench subtasks
(gplsi/cocoteros_va, truthfulqa_va, ...), Tatoeba MT (`Helsinki-NLP/tatoeba_mt`, parquet branch
`eng-nor`). refs_v5: 5,892,001 segments (+73,777), 2.86M matchable, 385M n-grams; work
`flag_decon_v5`, snapshot `$R/autoexp_decon10`; scan 47944498 (8 nodes ~1 h). The 152 benchmarks shared
with v4 change by < 0.5 pt in n-gram hit rate; new: MultiLoKo 70.0 % (external 11.1 %), Tatoeba 38.3 %
(11.6 %), AdvBench 1.6 % (24.7 %: its prompts are mostly < 20 tokens), NorIdiom not matchable (idioms
< 20 tokens). Dry run 47948407: **--frac 0.7/0.8/0.9/1.0: 1.47M/1.25M/1.08M/0.88M docs, 18.0B/13.9B/
11.9B/10.1B tokens (0.133/0.103/0.088/0.074 %)**; +42k docs / +0.37B tokens vs v4 at 0.9, almost all
MultiLoKo (7,868 of 29,673 segments found >= 90 % verbatim, 26.6k docs, role context).

**Decision (user, 2026-10-24): the 70 % rule** ("to be on the safe side"): a document is dropped when
its longest verbatim run of a benchmark item covers >= 70 % of the item, or is >= 200 tokens, and
>= 20 tokens. Lowering the rule catches truncated / end-modified copies; a change in the middle still
splits the run, so modified items stay; short items (20-28 tokens) need (nearly) full copies at any
rule. `flag-decon-v1` = v5 at 0.7: **1.47M documents, 18.0B tokens (0.133 % of 13.5T) dropped**.
Written to flag-decon-v1.new (job 47950366, 1 node, ~50 min: 441 shards rewritten, 11 symlinked),
verified (job 47953404: `verify.py tree` split over 8 nodes x 56 processes, 3:04 min -- 441/441
shards, every kept document checked token-identical on 200 random documents per shard, counts add
up, 0 errors), then swapped in for the first tree (v4 @ 0.9, never used, deleted). Datamix
`$R/data/flag_decon_v1_datamix_mn5.txt` regenerated: 452/452 shards complete (sizes = info.json),
weights identical to `flag_datamix_mn5.txt`; no broken symlinks; `flag/` untouched. Training config:
`config/backend/megatron/data_flag_decon_v1_mn5.yaml`. Total decontamination cost: ~250 GPU-h on
ehpc533 (v1-v5, mostly the 8-node scans, ~30 GPU-h each).

### 4.7 Comparison with the OpenEuroLLM "-pure" sets of agenttrove and openthoughts-3 (2026-10-28)

`flag/agenttrove-0.0-pure` and `flag/openthoughts-3-pure` were filtered independently by the
OpenEuroLLM pretraining-decontamination pipeline (NeMo Curator TaskDecontamination, commit
413335e): word n-grams of 98 English benchmark tasks (13 words, 8 for short tasks); a document is
removed whole (`remove-split-docs`) when it contains **any one** benchmark n-gram that occurs <= 10
times in the dataset (`--match-threshold 10`). They are re-tokenised twice (`openeurollm-256k-without-bos`,
same tokenizer as `flag`, and `...-reason-without-bos`); the comparison uses the first. Steps
(snapshots decon13-16): `compare` (compare_pure.py: document hashes, exact and canonical), `explain`
(explain_pure.py: their any-n-gram rule on OUR table, for each group and a 5,000-document control kept
by both), `absrule` (abs_rule_report.py: our documents dropped only by the >= 200-token branch).

**Their pure sets are exact subsets**: every pure document matches an original document token for
token (0 modified, 0 new). The `.bin` files compared are complete (size implied by `.idx`; fingerprint
in the scratchpad); second copies (`*.partial`) by the maintainer stopped growing at 15:24 and are
leftovers.

| | agenttrove-0.0 (1.57M docs, 20.2B tok) | openthoughts-3 (1.20M docs, 20.2B tok) |
|---|---|---|
| removed by them | 4,192 docs / 52.4M tok (0.26 %) | 5,013 / 84.2M (0.42 %) |
| removed by us (v5 @ 0.7, no hold-back) | 265 / 3.0M (0.015 %) | 746 / 12.3M (0.061 %) |
| both | 130 / 0.87M | 132 / 1.86M |
| theirs only | 4,062 / 51.5M | 4,881 / 82.3M |
| ours only | 135 / 2.2M | 614 / 10.5M |

Agreement: they also remove 49 % (agenttrove) / 18 % (openthoughts) of our removals; we remove 3.1 % /
2.6 % of theirs.
Both catch the GSM8K/MGSM/PolyMath and MMLU(-Pro) copies (the "both" groups: 74 / 81 of 130 / 132
contain >= 70 % of an item's n-grams).

**Theirs only: mostly the single-n-gram rule, partly benchmarks we lack.** With our table:

| theirs-only docs | agenttrove | openthoughts | control (kept by both) |
|---|---|---|---|
| contain any table n-gram | 75 % | 40 % | 7.5 % |
| >= 10 n-grams of one item | 37 % | 6.5 % | 1.2 / 0.1 % |
| >= 30 % of one item's n-grams | 67 docs | 49 docs | - |

- agenttrove: the matched items are mostly LiveCodeBench (2,117 docs), BigCodeBench (454), HumanEval(+)
  (230), AlpacaEval (125). Samples: generic code (`is_prime`, sklearn imports, number-to-words arrays,
  days-per-month lists matching LCB stdin test cases) and Codeforces boilerplate ("You can output the
  answer in any case ..."): one rare n-gram suffices for them. A minority are real copies we miss:
  e.g. an LCB/Codeforces problem copied with `$$$` math and its "Note" section moved into the middle;
  the reordering splits the copy into runs < 70 % each (our rule takes the longest run only).
- openthoughts-3: 60 % contain no n-gram of our 5.9M items, so their trigger is a benchmark we do not
  index (their list has full Hendrycks MATH, GSM-Symbolic, GSM8K-Platinum, DeepMind Math, DS-1000,
  LBPP, MedQA, LogiQA, BBH, SimpleQA, SuperGPQA, HLE, MT-Bench, ...) or an 8-word match below our
  20-token minimum. The rest are partial phrases (AIME tail "where m and n are positive integers ...
  Find m+n", power-of-two tables); none hold >= 50 % of one of our items.

**Ours only:** 135 / 614 documents holding >= 70 % of an item or a >= 200-token run that their
threshold misses because the n-gram is common in the dataset (> 10 occurrences: repeated items count
against themselves) or the benchmark is not in their list (AlpacaEval, BigCodeBench, SciRIFF, MBPP
partly, MixEval). openthoughts' 496 MixEval documents are a **false positive of ours**: one MixEval
free-form item (515 tokens) whose `[asy]` block (`rr_cartesian_axes` plotting helper) recurs in many
MATH-style problems gives a 401-token run -> the absolute branch.

**The absolute branch over the whole mix** (abs_rule_report.py, `_report_abs_rule.json`): 183,888
documents / 3.49B tokens of the 18.0B are dropped only by `run >= 200` (megamath 1.19B, stack-v2 0.89B,
nemotron-cc 0.45B, dclm 0.33B, hplt 0.32B). Its top items are long ones whose boilerplate is shared:
the HumanEval+/MBPP+ test harness (`def is_floats(x)` ..., 470 + ~1,200 docs), SciRIFF instruction
templates (>= 3,800 docs over the 13 SciRIFF items in the top 25), the MixEval `[asy]` item (457), an AlpacaEval loop snippet (984);
MultiLoKo Marathi Wikipedia passages (901 + 249) are genuine copies of the source pages (by design).

Conclusions:
- Our filter is **not** a superset of theirs: it is stricter per item (long verbatim runs only) and broader
  in benchmarks (5.9M items vs 98 tasks). Theirs removes 15x / 7x more documents here, mostly on one
  rare n-gram (high recall, many false positives on generic code / math phrases), and covers benchmarks
  we lack.
- Possible v6 improvements (none applied; flag-decon-v1 is unchanged): (1) add their extra benchmarks to the
  references; (2) a union-coverage criterion (sum of an item's runs >= 70 %) for reordered copies;
  (3) list boilerplate items (test harnesses, prompt templates, `[asy]` helpers) as generic or apply
  the absolute branch only to the item's non-template part; (4) for agenttrove/openthoughts, the
  pure sets can be combined with ours (keep a document only if both keep it) at a cost of 0.27 % /
  0.47 % of the tokens.

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
