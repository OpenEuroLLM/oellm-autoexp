# Baseline, diagnosis, adaptation

1. Preserve native 983M / 80B Megatron checkpoints and original source. Check
   copied assets by SHA256. Preserve Megatron arguments and provenance. Use
   identical fresh optimizer initialization for adaptation of both arms.
2. Baseline: full Winogrande (0-shot), first 256 SQuAD v2 documents (10-shot),
   first 256 CoQA documents (0-shot). Separate 256-document recall readouts:
   SQuAD-completion, SWDE, FDA (0-shot). Bare contexts, context 4096, seed 2026.
   Initial limited evaluation is a screen, not a replacement for full results.
   Preserve raw per-example outputs and verify identical document hashes.
3. On matched documents, inspect answer rate, wrong answers, echo/loop rate,
   answer distance and prompt truncation. Reuse original recall_diagnose.py:
   context removal, natural counterfactual answer replacement, UUID copying,
   teacher-forced answer log-probability and rank. Prefer 64 stratified
   documents per task for diagnostics; reserve the remainder for confirmation.
4. Inference interventions implemented in scripts/interventions.py: remove
   negative alpha, cap beta at 1, both, lengthen/shorten memory decay, restrict
   attention to 512/1024. Run both arms; sign/reflection interventions MUST be
   no-ops on KDA. Record kernel calls and restore all patches on exit. These
   identify dependencies and are not trained replacement architectures.
   Capping beta changes both erasure and writing in this tied delta update;
   interpret it as a beta-range diagnostic, not an isolated rotation test.
   Memory-gain interventions likewise change the distribution seen by later
   layers; an immediate loss does not rule out learning a better replacement.
5. Native adaptation compatibility gate before training: original Megatron
   loader, exact saved model arguments, strict state-dict coverage, fixed-token
   logits/loss comparison against original model. Never substitute the current
   ambitious loader merely because both variants are called complex_kda.
6. First adaptation screen: unchanged continuation; gate-only adaptation;
   longer-memory alpha parameterization; independently controlled erase/write.
   Same data batches, fresh Adam state, token budgets and LR sweep. For a new
   gate, require a parameterization that reproduces the starting recurrence
   before optimizing it. Benchmark 1 node first; choose parallelism from that.
7. If state-level diagnostics show capacity/interference, add a small slow
   memory bank or sparse explicit KV retrieval path, with a parameter/compute
   matched control. Measure storage and inference cost alongside retrieval.
8. Repeat promising findings at a second seed and on held-out full tasks.
   Retain general-language loss and a short commonsense check to expose tradeoffs.
   Only validated improvements graduate to ambitious v4.

No large adaptation sweep is launched before baseline and compatibility gates.
No new work goes to o260c. Shrink only in an idle-step window after explicit
authorization of the concrete node count; verify GPU access before resuming.
