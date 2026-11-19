#!/usr/bin/env python3
"""Realistic long-context evals of an HF export as it is (its sparse attention, config.sparse_attention): LongBench v1
(THUDM/LongBench, and its length-balanced LongBench-E files as ``<task>_e``) and InfiniteBench
(xinrongzhang2022/InfiniteBench), base-model formats. Batch size 1, chunked prefill through the model's cache
(long_generate.py), so any --max-len fits.

    long_realistic_eval.py --model <hf export> --data <long_evals dir> --out <dir> [--tasks narrativeqa ... en_mc]
                           [--max-len 65536] [--n 50] [--select random|spread|longest] [--min-tokens 0] [--seed 0]
    long_realistic_eval.py --data <long_evals dir> --stats [--tasks ...]       # parse-only (no torch): rows, lengths

``--data``: the directory holding LongBench/data/<task>.jsonl and InfiniteBench/<file>.jsonl (MN5:
/gpfs/projects/ehpc533/users/korbip/data/long_evals).

Formats and metrics (implemented here, no rouge / fuzzywuzzy / jieba needed):
  LongBench    the official prompts (config/dataset2prompt.json) and generation lengths (dataset2maxlen.json), greedy;
               the official metrics: qa_f1 (narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique,
               triviaqa), ROUGE-L of the `rouge` package (gov_report, qmsum, multi_news, samsum: summary-level union
               LCS, sentences split at '.'), classification (trec), retrieval (passage_retrieval_en), count
               (passage_count), code_sim (lcc, repobench-p: fuzzywuzzy's ratio = difflib's, first code line).
               trec / triviaqa / samsum are scored on the first line, as LongBench's eval.py does.
  InfiniteBench the base-model ("yarn-mistral") prompts of the official src/prompt.py.
               en_qa (longbook_qa_eng): generation, qa_f1. en_dia (longdialogue_qa_eng): generation, a gold name
               inside the answer. en_sum (longbook_sum_eng): generation (1200 tokens), ROUGE-L.
               en_mc (longbook_choice_eng) / code_debug: log-likelihood, no generation. The context is prefilled
               once; then (a) ``letter`` (the score): the official prompt ending "The letter of the correct answer
               is" / "The correct option is:", the option whose letter " A".." D" is the most likely; (b) ``text``:
               the question (en_mc) or "The function with the deliberate error is" (code_debug) and each option's
               text as the continuation, scored from a copy of the cache (sum of log-probs; ``text_norm`` per byte).
Base models do not stop: short-answer tasks stop at the end of the answer's first line (``line``), lcc / repobench-p
at the first code line (the one code_sim scores), qmsum at a blank line; summaries run to the generation limit.

Prompts longer than --max-len tokens are cut in the middle (LongBench's / InfiniteBench's convention: the instruction
and the question at the ends stay), the generation comes on top. --n rows per task, the same for every
model: ``random`` (a seeded sample), ``spread`` (evenly over the rows sorted by context length) or ``longest``;
--min-tokens skips shorter rows.

Writes <out>/samples.jsonl (one row per sample: prompt / full length, prediction, score) and <out>/summary.json:
the score (x100) per task, and per length bucket 0-8k / 8-16k / 16-32k / 32-64k / 64-128k / 128k+ of the full
(untruncated) prompt (``buckets``) and of the prompt fed (``buckets_fed``). Rows already in samples.jsonl are
skipped (resumable).
"""

from __future__ import annotations

import argparse
import collections
import difflib
import itertools
import json
import random
import re
import string
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------------------------------------------------

# LongBench config/dataset2prompt.json (v1), verbatim
LB_PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",  # noqa: E501
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",  # noqa: E501
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",  # noqa: E501
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",  # noqa: E501
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",  # noqa: E501
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",  # noqa: E501
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",  # noqa: E501
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",  # noqa: E501
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",  # noqa: E501
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",  # noqa: E501
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",  # noqa: E501
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",  # noqa: E501
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",  # noqa: E501
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",  # noqa: E501
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

# InfiniteBench src/prompt.py yarn_mistral_templates (the base-model prompts), verbatim; *_text: the option-text
# variant of the two multiple-choice tasks (ours: the official prompt minus the options and the letter request)
IB_PROMPTS = {
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",  # noqa: E501
    "longbook_choice_eng_text": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nAnswer:",
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",  # noqa: E501
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",  # noqa: E501
    "code_debug_text": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nThe function with the deliberate error is",  # noqa: E501
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely",  # noqa: E501
}


class Task:
    def __init__(self, suite: str, file: str, metric: str, max_new: int, stop: str | None, prompt: str):
        self.suite, self.file, self.metric, self.max_new, self.stop, self.prompt = (suite, file, metric, max_new,
                                                                                    stop, prompt)


# name -> (suite, file stem, metric, max new tokens (dataset2maxlen / DATA_NAME_TO_MAX_NEW_TOKENS), stop rule, prompt)
TASKS = {
    "narrativeqa": Task("lb", "narrativeqa", "qa_f1", 128, "line", "narrativeqa"),
    "qasper": Task("lb", "qasper", "qa_f1", 128, "line", "qasper"),
    "multifieldqa_en": Task("lb", "multifieldqa_en", "qa_f1", 64, "line", "multifieldqa_en"),
    "hotpotqa": Task("lb", "hotpotqa", "qa_f1", 32, "line", "hotpotqa"),
    "2wikimqa": Task("lb", "2wikimqa", "qa_f1", 32, "line", "2wikimqa"),
    "musique": Task("lb", "musique", "qa_f1", 32, "line", "musique"),
    "gov_report": Task("lb", "gov_report", "rouge", 512, None, "gov_report"),
    "qmsum": Task("lb", "qmsum", "rouge", 512, "para", "qmsum"),
    "multi_news": Task("lb", "multi_news", "rouge", 512, None, "multi_news"),
    "trec": Task("lb", "trec", "classification", 64, "line", "trec"),
    "triviaqa": Task("lb", "triviaqa", "qa_f1", 32, "line", "triviaqa"),
    "samsum": Task("lb", "samsum", "rouge", 128, "line", "samsum"),
    "passage_count": Task("lb", "passage_count", "count", 32, "line", "passage_count"),
    "passage_retrieval_en": Task("lb", "passage_retrieval_en", "retrieval", 32, "line", "passage_retrieval_en"),
    "lcc": Task("lb", "lcc", "code_sim", 64, "code", "lcc"),
    "repobench-p": Task("lb", "repobench-p", "code_sim", 64, "code", "repobench-p"),
    "en_mc": Task("ib", "longbook_choice_eng", "mc", 0, None, "longbook_choice_eng"),
    "en_qa": Task("ib", "longbook_qa_eng", "qa_f1", 40, "line", "longbook_qa_eng"),
    "en_dia": Task("ib", "longdialogue_qa_eng", "dia", 40, "line", "longdialogue_qa_eng"),
    "en_sum": Task("ib", "longbook_sum_eng", "rouge", 1200, None, "longbook_sum_eng"),
    "code_debug": Task("ib", "code_debug", "mc", 0, None, "code_debug"),
}
ALIASES = {"longbook_choice_eng": "en_mc", "longbook_qa_eng": "en_qa", "longdialogue_qa_eng": "en_dia",
           "longbook_sum_eng": "en_sum", "en.mc": "en_mc", "en.qa": "en_qa", "en.dia": "en_dia", "en.sum": "en_sum",
           "code.debug": "code_debug"}
DEFAULT_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
                 "trec", "triviaqa", "samsum", "passage_retrieval_en", "lcc", "repobench-p", "en_mc", "en_qa",
                 "code_debug"]
FIRST_LINE = {"trec", "triviaqa", "samsum"}  # LongBench eval.py scores these on the prediction's first line
BUCKETS = [(0, 8192, "0-8k"), (8192, 16384, "8-16k"), (16384, 32768, "16-32k"), (32768, 65536, "32-64k"),
           (65536, 131072, "64-128k"), (131072, float("inf"), "128k+")]


def task_of(name: str):
    """(canonical name, Task, LongBench-E?): ``<lb task>_e`` reads the LongBench-E file of the task."""
    name = ALIASES.get(name, name)
    if name in TASKS:
        return name, TASKS[name], False
    if name.endswith("_e") and name[:-2] in TASKS and TASKS[name[:-2]].suite == "lb":
        return name, TASKS[name[:-2]], True
    raise SystemExit(f"unknown task {name!r}; known: {sorted(TASKS)} (+ <LongBench task>_e)")


def task_path(data: Path, name: str) -> Path:
    _, t, e = task_of(name)
    stem = t.file + ("_e" if e else "")
    cands = ([data / "LongBench" / "data" / f"{stem}.jsonl", data / "data" / f"{stem}.jsonl"] if t.suite == "lb"
             else [data / "InfiniteBench" / f"{stem}.jsonl"]) + [data / f"{stem}.jsonl"]
    for p in cands:
        if p.exists():
            return p
    raise SystemExit(f"{name}: none of {[str(c) for c in cands]} exists")


def ib_answer(row: dict) -> tuple[str, int]:
    """InfiniteBench's get_answer for the multiple-choice tasks: (option text, index of the gold option)."""
    ans = row["answer"]
    if isinstance(ans, list) and len(ans) == 2 and ans[1] in "ABCD" and len(ans[1]) == 1:
        return ans[0], "ABCD".index(ans[1])
    text = ans[0] if isinstance(ans, list) else ans
    return text, row["options"].index(text)


def load_rows(data: Path, name: str) -> list[dict]:
    """The rows of a task, as {idx, id, context, fields (the prompt's other slots), answers, all_classes, options,
    gold}."""
    _, t, _ = task_of(name)
    rows = []
    with task_path(data, name).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            r = json.loads(line)
            if t.suite == "lb":
                rows.append({"idx": i, "id": r.get("_id", str(i)), "context": r["context"],
                             "fields": {"input": r["input"]}, "answers": list(r["answers"]),
                             "all_classes": r.get("all_classes"), "length": r.get("length")})
                continue
            fields = {"question": r["input"], "input": r["input"]}
            rec = {"idx": i, "id": r.get("id", i), "context": r["context"], "fields": fields,
                   "answers": r["answer"] if isinstance(r["answer"], list) else [r["answer"]]}
            if t.metric == "mc":
                text, gold = ib_answer(r)
                rec.update(options=list(r["options"]), gold=gold, answers=[text])
                fields.update({f"OPTION_{c}": o for c, o in zip("ABCD", r["options"])})
            rows.append(rec)
    return rows


def prompt_parts(name: str, row: dict) -> tuple[str, str, dict]:
    """(head, context, {variant: tail}): the prompt is head + context + tail; variants 'gen', or 'letter' / 'text'
    for the multiple-choice tasks."""
    _, t, _ = task_of(name)
    templates = {"gen": LB_PROMPTS[t.prompt]} if t.suite == "lb" else (
        {"letter": IB_PROMPTS[t.prompt], "text": IB_PROMPTS[t.prompt + "_text"]} if t.metric == "mc"
        else {"gen": IB_PROMPTS[t.prompt]})
    head, tails = None, {}
    for k, tpl in templates.items():
        h, tail = tpl.split("{context}")
        assert head in (None, h), "the variants share the text before the context"
        head, tails[k] = h, tail.format(**row["fields"])
    return head.format(**row["fields"]), row["context"], tails


def select(rows: list[dict], n: int, seed: int, name: str, how: str) -> list[dict]:
    """--n rows in a model-independent order: a seeded sample, spread evenly over the context length, or the
    longest (rows sorted by context characters)."""
    if how == "random":
        order = list(rows)
        random.Random(f"{seed}:{name}").shuffle(order)
        return order
    by_len = sorted(rows, key=lambda r: (len(r["context"]), r["idx"]))
    if how == "longest":
        return by_len[::-1]
    if n >= len(by_len):
        return by_len
    picks = sorted({round(k * (len(by_len) - 1) / max(n - 1, 1)) for k in range(n)})
    return [by_len[k] for k in picks] + [r for k, r in enumerate(by_len) if k not in set(picks)]


def bucket(tokens: int) -> str:
    return next(lab for lo, hi, lab in BUCKETS if lo <= tokens < hi)


def middle_truncate(ids: list[int], budget: int) -> list[int]:
    if len(ids) <= budget:
        return ids
    budget = max(budget, 0)
    return ids[: budget // 2] + ids[len(ids) - (budget - budget // 2):] if budget else []


# ----------------------------------------------------------------------------------------------------------------
# metrics (LongBench metrics.py / InfiniteBench compute_scores.py)
# ----------------------------------------------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, gold: str, **_) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def _lcs_words(x: list[str], y: list[str]) -> set:
    """The words of the `rouge` package's reconstructed LCS of x and y (its _recon_lcs tie-break), iteratively."""
    n, m = len(x), len(y)
    t = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        xi, ti, tp = x[i - 1], t[i], t[i - 1]
        for j in range(1, m + 1):
            ti[j] = tp[j - 1] + 1 if xi == y[j - 1] else max(tp[j], ti[j - 1])
    out, i, j = set(), n, m
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            out.add(x[i - 1])
            i, j = i - 1, j - 1
        elif t[i - 1][j] > t[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return out


def rouge_l(pred: str, gold: str, **_) -> float:
    """The `rouge` package's (1.0.1) rouge-l f, as LongBench calls it (Rouge().get_scores([pred], [gold])): sentences
    split at '.', summary-level union LCS over word sets (exclusive=True); 0 where it raises (an empty side)."""
    hyp = [" ".join(s.split()) for s in pred.split(".") if len(s) > 0]
    ref = [" ".join(s.split()) for s in gold.split(".") if len(s) > 0]
    if not hyp or not ref:
        return 0.0
    words = lambda sents: list(itertools.chain(*[s.split(" ") for s in sents]))  # noqa: E731
    m, n = len(set(words(ref))), len(set(words(hyp)))
    union, llcs = set(), 0
    for r in ref:
        rw = r.split(" ")
        before = len(union)
        for h in hyp:
            union |= _lcs_words(rw, h.split(" "))
        llcs += len(union) - before
    r_lcs, p_lcs = llcs / m, llcs / n
    return 2.0 * ((p_lcs * r_lcs) / (p_lcs + r_lcs + 1e-8))


def code_sim(pred: str, gold: str, **_) -> float:
    """LongBench code_sim_score: the first line without ` # //, fuzz.ratio (difflib's ratio without
    python-Levenshtein) / 100."""
    line = ""
    for ln in pred.lstrip("\n").split("\n"):
        if "`" not in ln and "#" not in ln and "//" not in ln:
            line = ln
            break
    if line == gold:
        return 1.0
    if not line or not gold:
        return 0.0
    return int(round(100 * difflib.SequenceMatcher(None, line, gold).ratio())) / 100


def classification(pred: str, gold: str, all_classes=None, **_) -> float:
    matches = [c for c in all_classes if c in pred]
    for term in matches:  # LongBench's loop verbatim: it removes while iterating (and so skips the next class)
        if term in gold and term != gold:
            matches.remove(term)
    return 1.0 / len(matches) if gold in matches else 0.0


def retrieval(pred: str, gold: str, **_) -> float:
    gid = re.findall(r"Paragraph (\d+)", gold)[0]
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(x == gid for x in nums) / len(nums)


def count(pred: str, gold: str, **_) -> float:
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(x == str(gold) for x in nums) / len(nums)


def dia(pred: str, gold: str, **_) -> float:
    """InfiniteBench longdialogue_qa_eng: the gold name inside the (upper-cased) answer."""
    return float(gold.upper() in pred.strip().upper())


METRICS = {"qa_f1": qa_f1, "rouge": rouge_l, "code_sim": code_sim, "classification": classification,
           "retrieval": retrieval, "count": count, "dia": dia}


def score_pred(name: str, pred: str, row: dict) -> float:
    _, t, _ = task_of(name)
    base = name[:-2] if name.endswith("_e") else name
    if base in FIRST_LINE:
        pred = pred.lstrip("\n").split("\n")[0]
    fn = METRICS[t.metric]
    return max(fn(pred, g, all_classes=row.get("all_classes")) for g in row["answers"])


def stop_rule(rule: str | None):
    """text -> done? for the generation of a base model, which does not emit EOS after the answer."""
    if rule == "line":
        return lambda text: "\n" in text.lstrip()
    if rule == "para":
        return lambda text: "\n\n" in text.strip()
    if rule == "code":  # a complete line code_sim would score
        return lambda text: any("`" not in ln and "#" not in ln and "//" not in ln
                                for ln in text.lstrip("\n").split("\n")[:-1])
    return None


def cut(rule: str | None, text: str) -> str:
    """The answer of a generation stopped by ``rule``."""
    if rule == "line":
        return text.lstrip().split("\n")[0]
    if rule == "para":
        return text.strip().split("\n\n")[0]
    return text


# ----------------------------------------------------------------------------------------------------------------
# model side (torch imported lazily: the loader and metrics stay usable without it)
# ----------------------------------------------------------------------------------------------------------------


def encode(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def continuation_logprob(model, cache, last_logits, cont: list[int]) -> float:
    """log p(cont | the state in cache, whose last position predicted last_logits); the cache stays untouched (the
    continuation runs on a copy)."""
    import copy

    import torch

    lp = float(torch.log_softmax(last_logits[0].float(), -1)[cont[0]])
    if len(cont) > 1:
        dev = next(model.parameters()).device
        with torch.no_grad():
            out = model(input_ids=torch.tensor([cont[:-1]], device=dev), past_key_values=copy.deepcopy(cache),
                        use_cache=True)
        lps = torch.log_softmax(out.logits[0].float(), -1)
        lp += float(sum(lps[i, cont[i + 1]] for i in range(len(cont) - 1)))
    return lp


def eval_row(model, tok, lg, name: str, row: dict, max_len: int, chunk: int) -> dict:
    """One sample: the prompt tokenized whole (multiple choice: head + context whole, each tail on its own) and, when
    longer than max_len, cut in the middle (LongBench's pred.py on tokens), then generation or option scoring."""
    import copy

    _, t, _ = task_of(name)
    head, ctx, tails = prompt_parts(name, row)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    tids = {k: encode(tok, v) for k, v in tails.items()}
    rec = {"task": name, "idx": row["idx"], "id": row["id"], "max_len": max_len}
    t0 = time.time()
    if t.metric == "mc":
        conts = {"letter": [encode(tok, f" {c}") for c in "ABCD"[: len(row["options"])]],
                 "text": [encode(tok, " " + o) for o in row["options"]]}
        longest = max(len(tids[k]) + max(len(c) for c in conts[k]) for k in tids)
        pre = encode(tok, head + ctx)
        full = len(bos) + len(pre) + longest
        pre = bos + middle_truncate(pre, max_len - len(bos) - longest)
        cache, _ = lg.prefill(model, pre, chunk=chunk)
        lps = {}
        for k in ("letter", "text"):
            c = copy.deepcopy(cache) if k == "letter" else cache  # 'text' continues the context's cache in place
            c, last = lg.prefill(model, tids[k], cache=c, chunk=chunk)
            lps[k] = [continuation_logprob(model, c, last, cont) for cont in conts[k]]
            del c
        del cache
        argmax = lambda v: max(range(len(v)), key=lambda i: v[i])  # noqa: E731
        norm = [lp / max(len(o.encode("utf-8")), 1) for lp, o in zip(lps["text"], row["options"])]
        pick = {"letter": argmax(lps["letter"]), "text": argmax(lps["text"]), "text_norm": argmax(norm)}
        metrics = {k: float(v == row["gold"]) for k, v in pick.items()}
        rec.update(full_tokens=full, prompt_tokens=len(pre) + longest, gold=row["gold"], pick=pick,
                   logprobs={k: [round(x, 4) for x in v] for k, v in lps.items()}, score=metrics["letter"],
                   metrics=metrics)
    else:
        ids = encode(tok, head + ctx + tails["gen"])
        full = len(bos) + len(ids)
        ids = bos + middle_truncate(ids, max_len - len(bos))
        rule = stop_rule(t.stop)
        stop = None if rule is None else (lambda new: rule(tok.decode(new, skip_special_tokens=True)))
        gen = lg.greedy(model, ids, t.max_new, tok.eos_token_id, chunk=chunk, stop=stop)
        raw = tok.decode(gen, skip_special_tokens=True)
        pred = cut(t.stop, raw)
        s = score_pred(name, pred, row)
        rec.update(full_tokens=full, prompt_tokens=len(ids), gen_tokens=len(gen), pred=pred,
                   raw=raw if raw != pred else None, answers=row["answers"], score=s, metrics={"score": s})
    rec["truncated"] = rec["prompt_tokens"] < rec["full_tokens"]
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def summarize(recs: list[dict]) -> dict:
    """Per task: score x100, every metric, per length bucket of the full and of the fed prompt."""
    out = {}
    for name in dict.fromkeys(r["task"] for r in recs):
        rs = [r for r in recs if r["task"] == name]

        def agg(group):
            keys = list(dict.fromkeys(k for r in group for k in r["metrics"]))
            return {"score": round(100 * sum(r["score"] for r in group) / len(group), 2), "n": len(group),
                    "truncated": sum(r["truncated"] for r in group),
                    **({f"{k}": round(100 * sum(r["metrics"][k] for r in group) / len(group), 2) for k in keys}
                       if keys != ["score"] else {})}

        def by(key):
            b = collections.OrderedDict((lab, []) for _, _, lab in BUCKETS)
            for r in rs:
                b[bucket(r[key])].append(r)
            return {lab: agg(g) for lab, g in b.items() if g}

        out[name] = {**agg(rs), "mean_full_tokens": round(sum(r["full_tokens"] for r in rs) / len(rs)),
                     "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rs) / len(rs)),
                     "seconds": round(sum(r["seconds"] for r in rs), 1),
                     "buckets": by("full_tokens"), "buckets_fed": by("prompt_tokens")}
    return out


def stats(data: Path, tasks: list[str]) -> None:
    """Parse-only: rows, context characters (min / median / max), the fraction > 32k / 128k / 512k characters."""
    for name in tasks:
        rows = load_rows(data, name)
        lens = sorted(len(r["context"]) for r in rows)
        head, ctx, tails = prompt_parts(name, rows[0])
        assert ctx == rows[0]["context"] and all(isinstance(v, str) for v in tails.values())
        frac = lambda c: sum(x > c for x in lens) / len(lens)  # noqa: E731
        print(f"{name:22s} rows {len(rows):4d}  context chars min {lens[0]:8d} median {lens[len(lens) // 2]:8d} "
              f"max {lens[-1]:8d}  >32k {frac(32768):.2f} >128k {frac(131072):.2f} >512k {frac(524288):.2f}  "
              f"({task_path(data, name)})", flush=True)


def run(model, tok, tasks: list[str], data: Path, out: Path, max_len: int, n: int, how: str = "random",
        seed: int = 0, min_tokens: int = 0, chunk: int = 32768) -> dict:
    """Score ``tasks`` (resumable through <out>/samples.jsonl); returns and writes the summary."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import long_generate as lg

    out.mkdir(parents=True, exist_ok=True)
    samples = out / "samples.jsonl"
    recs = [json.loads(line) for line in samples.read_text().splitlines()] if samples.exists() else []
    bad = {r["max_len"] for r in recs} - {max_len}
    if bad:
        raise SystemExit(f"{samples} holds rows of --max-len {sorted(bad)}, not {max_len}: use another --out")
    done = {(r["task"], r["idx"]) for r in recs}
    with samples.open("a") as fh:
        for name in tasks:
            name = task_of(name)[0]
            taken = sum(1 for r in recs if r["task"] == name)
            for row in select(load_rows(data, name), n, seed, name, how):
                if taken >= n:
                    break
                if (name, row["idx"]) in done:
                    continue
                if min_tokens and len(encode(tok, row["context"])) < min_tokens:
                    continue
                rec = eval_row(model, tok, lg, name, row, max_len, chunk)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                recs.append(rec)
                taken += 1
                print(f"[long_realistic_eval] {name} #{row['idx']}: {rec['full_tokens']} -> {rec['prompt_tokens']} "
                      f"tokens, score {rec['score']:.3f}, {rec['seconds']} s", flush=True)
            got = [r for r in recs if r["task"] == name]
            if got:
                print(f"[long_realistic_eval] {name}: {100 * sum(r['score'] for r in got) / len(got):.2f} over "
                      f"{len(got)}", flush=True)
    return summarize(recs)  # every task in samples.jsonl (earlier invocations' too)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF export (trust_remote_code)")
    ap.add_argument("--data", type=Path, required=True, help="dir with LongBench/data/*.jsonl, InfiniteBench/*.jsonl")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                    help=f"of {sorted(TASKS)}, LongBench tasks also as <task>_e (LongBench-E)")
    ap.add_argument("--max-len", type=int, default=65536, help="prompt tokens; longer contexts truncated in the middle")
    ap.add_argument("--n", type=int, default=50, help="samples per task")
    ap.add_argument("--select", choices=["random", "spread", "longest"], default="random")
    ap.add_argument("--min-tokens", type=int, default=0, help="skip rows whose context is shorter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32768, help="prefill chunk (tokens)")
    ap.add_argument("--stats", action="store_true", help="parse the data only and print rows / lengths (no torch)")
    a = ap.parse_args()
    tasks = [task_of(t)[0] for t in a.tasks]
    if a.stats:
        stats(a.data, tasks)
        return
    if not a.model or not a.out:
        ap.error("--model and --out are required (unless --stats)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    sa = getattr(model.config, "sparse_attention", None)
    print(f"[long_realistic_eval] {a.model}: sparse_attention {json.dumps(sa)}", flush=True)
    summary = run(model, tok, tasks, a.data, a.out, a.max_len, a.n, a.select, a.seed, a.min_tokens, a.chunk)
    (a.out / "summary.json").write_text(json.dumps(
        {"model": a.model, "sparse_attention": sa, "max_len": a.max_len, "n": a.n, "select": a.select,
         "min_tokens": a.min_tokens, "seed": a.seed, "tasks": summary}, indent=2) + "\n")
    print(json.dumps({k: v["score"] for k, v in summary.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
