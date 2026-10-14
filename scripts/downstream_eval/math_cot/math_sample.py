#!/usr/bin/env python3
"""Sampled competition-math eval for 4k-context base models (runs inside oellm-
eval-vllm.sif).

The release evals (Evalchemy via lm-eval) are GREEDY: Evalchemy passes do_sample=False, which lm-eval turns
into temperature 0, so the 10 "repeats" differ only by batch-dependent bf16 noise. Here every problem gets
N independent samples (temperature 0.6, top_p 0.95, own seed), so per-problem success rates, pass@k and
loop rates under sampling become measurable.

Prompts and tokenization are Evalchemy's / lm-eval's own: the model is loaded through lm-eval's VLLM class,
the prompt is PROMPT.format(...) -> lm.apply_chat_template -> lm.tok_encode, and the budget is
4096 - prompt tokens (the models' trained context; nothing beyond it is valid).

    python3 math_sample.py --view VIEW --name CK --out DIR --shard I --nshards 4 [--tasks AIME24,AMC23] [--samples N]
    --check: greedy on the first AIME24 problems only, to compare against the release run's outputs.
    --prefill TEXT: appended after the prompt (e.g. " <think>\n") to force a worked solution; the
        generated text after it is what gets graded. Output name gets --tag.
    --nll: no generation; per-token NLL of each dataset's reference solution after the same prompt
        (prompt + " " + solution, scored with vLLM prompt_logprobs), for AIME24/AIME25/MATH500.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/opt/evalchemy")
os.chdir("/opt/evalchemy")

PROMPT = "Problem: {problem}\nMark your solution with \\boxed\nAnswer:"
DATA = {
    "AIME24": "eval/chat_benchmarks/AIME24/data/aime24.json",
    "AIME25": "eval/chat_benchmarks/AIME25/data/aime25.json",
    "AMC23": "eval/chat_benchmarks/AMC23/data/amc23.json",
    "MATH500": "eval/chat_benchmarks/MATH500/data/math500.jsonl",
}
NSAMP = {"AIME24": 16, "AIME25": 16, "AMC23": 16, "MATH500": 4}
CTX = 4096


def load(task):
    txt = open(DATA[task]).read().strip()
    try:
        rows = json.loads(txt)
        rows = rows if isinstance(rows, list) else [rows]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in txt.splitlines() if line.strip()]
    return [
        dict(
            problem=r.get("problem", r.get("question")),
            answer=str(r.get("expected_answer", r.get("answer"))),
            solution=r.get("reference_solution") or r.get("solution"),
        )
        for r in rows
    ]


def distinct20(ids):
    tail = list(ids)[-1000:]
    ng = [tuple(tail[i : i + 20]) for i in range(len(tail) - 19)]
    return len(set(ng)) / max(len(ng), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--prefill", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--nll", action="store_true")
    ap.add_argument(
        "--tasks", default=",".join(DATA), help="comma-separated subset of " + ",".join(DATA)
    )
    ap.add_argument(
        "--samples", type=int, default=0, help="samples per problem (0: the per-task default)"
    )
    a = ap.parse_args()
    a.prefill = a.prefill.encode().decode("unicode_escape")

    from lm_eval.models.vllm_causallms import VLLM
    from vllm import SamplingParams

    lm = VLLM(
        pretrained=a.view,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
        seed=1234,
        trust_remote_code=True,
        max_length=CTX,
    )
    if a.nll:
        return solution_nll(a, lm)

    jobs = []  # (task, idx, answer, prompt_ids, n)
    tasks = ["AIME24"] if a.check else a.tasks.split(",")
    unknown = sorted(set(tasks) - set(DATA))
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; known: {list(DATA)}")
    for t in tasks:
        rows = load(t)[:8] if a.check else load(t)
        for i, r in enumerate(rows):
            if not a.check and (sum(map(ord, t)) + i) % a.nshards != a.shard:
                continue
            prompt = lm.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(problem=r["problem"])}]
            )
            ids = lm.tok_encode(prompt + a.prefill)
            jobs.append((t, i, r["answer"], ids, 1 if a.check else (a.samples or NSAMP[t])))

    params = []
    for t, i, _, ids, n in jobs:
        budget = CTX - len(ids)
        if a.check:
            params.append(
                SamplingParams(
                    temperature=0.0,
                    max_tokens=budget,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
            )
        else:
            params.append(
                SamplingParams(
                    n=n,
                    temperature=a.temperature,
                    top_p=a.top_p,
                    max_tokens=budget,
                    seed=10007 * (sum(map(ord, t)) % 97) + i,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
            )
    outs = lm.model.generate([{"prompt_token_ids": j[3]} for j in jobs], params)

    os.makedirs(a.out, exist_ok=True)
    part = "check" if a.check else f"{'+'.join(tasks)}.shard{a.shard}"
    fn = f"{a.out}/{a.name}{a.tag}.{part}.jsonl"
    with open(fn, "w") as f:
        for (t, i, ans, ids, n), o in zip(jobs, outs):
            f.write(
                json.dumps(
                    dict(
                        ck=a.name + a.tag,
                        prefill=a.prefill,
                        task=t,
                        idx=i,
                        answer=ans,
                        prompt_tokens=len(ids),
                        samples=[
                            dict(
                                text=c.text,
                                n_tokens=len(c.token_ids),
                                finish=c.finish_reason,
                                distinct20=round(distinct20(c.token_ids), 4),
                            )
                            for c in o.outputs
                        ],
                    )
                )
                + "\n"
            )
    print("wrote", fn, len(jobs), "prompts", flush=True)


def solution_nll(a, lm):
    from vllm import SamplingParams

    rows = []
    for t in [t for t in a.tasks.split(",") if t in ("AIME24", "AIME25", "MATH500")]:
        for i, r in enumerate(load(t)):
            if not r["solution"]:
                continue
            prompt = lm.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(problem=r["problem"])}]
            )
            p_ids = lm.tok_encode(prompt)
            full = lm.tok_encode(prompt + " " + r["solution"].strip())[: CTX - 1]
            if (
                full[: len(p_ids)] != p_ids
            ):  # tokenization boundary moved: score from the first differing token
                k = next(j for j, (x, y) in enumerate(zip(p_ids, full)) if x != y)
            else:
                k = len(p_ids)
            rows.append((t, i, k, full))
    outs = lm.model.generate(
        [{"prompt_token_ids": f} for _, _, _, f in rows],
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
    )
    os.makedirs(a.out, exist_ok=True)
    fn = f"{a.out}/{a.name}.nll.jsonl"
    with open(fn, "w") as f:
        for (t, i, k, full), o in zip(rows, outs):
            lp = [o.prompt_logprobs[j][full[j]].logprob for j in range(k, len(full))]
            f.write(
                json.dumps(
                    dict(
                        ck=a.name,
                        task=t,
                        idx=i,
                        n_tokens=len(lp),
                        nll=-sum(lp),
                        nll_first256=-sum(lp[:256]),
                        n_first256=min(256, len(lp)),
                    )
                )
                + "\n"
            )
    print("wrote", fn, len(rows), "solutions", flush=True)


if __name__ == "__main__":
    main()
