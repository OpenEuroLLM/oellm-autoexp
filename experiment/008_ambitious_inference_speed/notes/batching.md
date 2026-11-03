# Saturated batch sweep

This extends the earlier 32-prompt probe to 1024 distinct fixed prompts. The sequence cap is aggregate per node: pure EP4/DP4 divides it by four, while TP4/DP1 uses the full cap in one scheduler. Submitted request count remains 1024 in every case; the scheduler admits work up to its sequence and token limits. Prefix caching is disabled.

Each request has 256 input tokens from the seeded validation corpus. Scoring requests full prompt logprobs and one generated token. After a warmup, two full scoring passes are recorded. Generation requests no prompt logprobs and exactly 32 output tokens regardless of EOS. After warming up on two concurrency waves, a full 1024-request generation pass is measured. Generation throughput includes its prefill time; it is not a decode-only kernel rate.

Controls and sweep:

- 32 sequences, 8192 aggregate scheduled tokens.
- 64, 128, 256 sequences at aggregate budgets 8192 and 32768.
- Same checkpoint, model implementation, prompts, eager mode, 85% memory target, and Engram mode throughout.
- Layout EP4/DP4 on as02r1b04; layout TP4/DP1 on as02r1b16. Each layout's sizes run on the same host. Cross-layout results can still include host variation.

The larger sequence cap is not proof that all sequences ran simultaneously. `scheduler_peaks_by_frontend` records observed running/waiting counts, cache occupancy and preemptions. Short scoring requests may hit the token budget before reaching the sequence cap. GPU peak allocated/reserved bytes are collected inside actual workers, and nvidia-smi independently samples device memory/utilization.

All cases use 4096 as the engine context limit, but the workload is only 256 input plus 32 output tokens. Successful admission at a batch size therefore establishes feasibility for this workload, not arbitrary 4096-token prompts or long generations. A practical production choice must consider throughput, task lengths, and memory margin together.

Driver: `../scripts/bench_batching.py`. Launcher: `../scripts/run_batching.sh`. Allocation 48460908, two independent nodes in one acc_debug job, two-hour wall cap. Snapshot `/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_infer008_batch`. Per-case stdout/stderr and incremental rank JSON remain in its `experiment/008_ambitious_inference_speed/results/batching/{ep4,tp4}/s<seqs>_t<tokens>/`. Engine algorithms remain at vLLM 5bd602f81c, combined model/kernel SHA256 `1a63b2cbca418278297c7ac4db7852c2c9979a471aa62c6f9cd6cc648746b46c`.

Downloaded artifacts are summarized by `../scripts/summarize_batching.py`, generating `../results/batching/summary.json` and `summary.md`. Case failures/timeouts are retained rather than silently omitted. Initial cold loading is excluded from throughput; startup seconds are reported separately.

Retry/provenance correction: allocation 48460908 yielded no measurements because the initial callable memory RPC was rejected. Its artifacts are isolated in `results/batching_attempt1`. Corrected allocation 48461862 runs on as04r3b02 / as04r3b04; each layout's `allocation.txt` records the assignment. The named worker extension preserves the default serialization policy. Cold loading repeats on these hosts and is again excluded from throughput.
