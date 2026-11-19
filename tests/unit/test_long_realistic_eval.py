"""scripts/korbi/campaign_midtraining/long_realistic_eval.py: the LongBench / InfiniteBench loaders and prompts, the
re-implemented official metrics, option scoring from a copy of the cache (= a one-shot forward), and an end-to-end
resumable run on a tiny remote-code export with sparse attention."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from oellm_autoexp.hf_export.oellm_hybrid import configuration_oellm_hybrid as _ch
from oellm_autoexp.hf_export.oellm_hybrid import modeling_oellm_hybrid as _mh

sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_hybrid", _ch)
sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_hybrid", _mh)

from oellm_autoexp.hf_export import export_ambitious as ea  # noqa: E402
from oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_ambitious import (  # noqa: E402
    OellmAmbitiousConfig,
)
from oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_ambitious import (  # noqa: E402
    OellmAmbitiousForCausalLM,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/korbi/campaign_midtraining"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lre = _load("long_realistic_eval")
lg = _load("long_generate")

SPARSE = dict(page_sparse_selector="page_mean", page_sparse_per_group=True, page_sparse_page=8, page_sparse_topk=2,
              page_sparse_window=8, page_sparse_sink_pages=1, page_sparse_select_block=8)
CHARS = "\n" + "".join(chr(c) for c in range(32, 127))  # ids 1..96; 0 = BOS / EOS


class CharTok:
    """A character tokenizer over the tiny model's 97 ids (the script needs encode / decode / bos / eos only)."""
    bos_token_id = eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [CHARS.index(c) + 1 if c in CHARS else CHARS.index("?") + 1 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(CHARS[i - 1] for i in ids if i > 0)


def _cfg(**kw):
    return OellmAmbitiousConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, head_dim=8, num_key_value_heads=2,
        block_types=["attention", "attention"], residual="mhc", mhc_expansion_rate=4, rms_norm_eps=1e-5,
        qk_layernorm=True, sparse_attention=SPARSE, **kw)


def _text(seed, n):
    import random

    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "river", "stone", "Paris", "garden", "music", "the", "of", "and"]
    return " ".join(rng.choice(words) for _ in range(n // 6)) + "."


def _write_data(root: Path):
    """A couple of rows per format, in the layout of the downloaded repos (LongBench/data, InfiniteBench)."""
    lb, ib = root / "LongBench" / "data", root / "InfiniteBench"
    lb.mkdir(parents=True)
    ib.mkdir(parents=True)

    def dump(path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def lbrow(i, n, inp, ans, classes=None):
        return {"input": inp, "context": _text(i, n), "answers": ans, "length": n // 6, "dataset": "x",
                "language": "en", "all_classes": classes, "_id": f"id{i}"}

    dump(lb / "hotpotqa.jsonl", [lbrow(i, 300 + 400 * i, "Where is the river?", ["Paris"]) for i in range(3)])
    dump(lb / "hotpotqa_e.jsonl", [lbrow(10 + i, 200, "Where?", ["garden"]) for i in range(2)])
    dump(lb / "trec.jsonl", [lbrow(20 + i, 400, "Question: What is it?\nType:", ["Entity"],
                                   ["Entity", "Location", "Description"]) for i in range(2)])
    dump(lb / "samsum.jsonl", [lbrow(30 + i, 400, "Dialogue: hi\nSummary:", ["They greet."]) for i in range(2)])
    dump(lb / "lcc.jsonl", [lbrow(40 + i, 500, "", ["x = compute(y)"]) for i in range(2)])
    opts = ["Walking Georgie", "Taking care of Totty", "Working in the dairy", "Light housework"]
    dump(ib / "longbook_choice_eng.jsonl", [{"id": i, "context": _text(50 + i, 1500), "input": "Which chore?",
                                             "answer": [opts[i]], "options": opts} for i in range(2)])
    dump(ib / "code_debug.jsonl", [{"id": 0, "context": "def f():\n    return 1\n" * 40, "input": "Which funtion?",
                                    "answer": ["cmd_gen"], "options": ["Resource.pkgname", "repack", "cmd_gen", "_init"]}])
    dump(ib / "longbook_qa_eng.jsonl", [{"id": 0, "context": _text(60, 2000), "input": "Who?", "answer": ['"Peyton"'],
                                         "options": []}])
    dump(ib / "longdialogue_qa_eng.jsonl", [{"id": 0, "context": "ACE: hi\n$$MASK$$: yo\n" * 30,
                                             "input": "Which character is $$MASK$$ ?", "answer": ["ACE", "ACE R"],
                                             "options": []}])


def test_metrics_follow_longbench_and_infinitebench():
    assert lre.qa_f1("The Eiffel tower.", "eiffel Tower") == 1.0
    assert lre.qa_f1("in Paris, France", "Paris") == pytest.approx(2 * (1 / 3) * 1 / (1 / 3 + 1))
    assert lre.qa_f1("", "Paris") == 0.0
    # rouge package (1.0.1) rouge-l: LCS words {the, cat, on, mat} of 5 distinct words on each side
    assert lre.rouge_l("the cat sat on the mat", "the cat was on the mat") == pytest.approx(0.8, abs=1e-6)
    # its summary-level union over word sets: every reference word is covered by some sentence's LCS
    assert lre.rouge_l("a b. c d", "a c. b d") == pytest.approx(1.0, abs=1e-6)
    assert lre.rouge_l("", "a b") == 0.0 and lre.rouge_l("...", "a b") == 0.0
    assert lre.code_sim("\n# a comment\nx = compute(y)\nmore", "x = compute(y)") == 1.0
    assert lre.code_sim("abcd", "abce") == 0.75  # difflib ratio 2*3/8
    assert lre.code_sim("", "x") == 0.0
    classes = ["Entity", "Location", "Description", "Abbreviation"]
    assert lre.classification("Location", "Location", all_classes=classes) == 1.0
    assert lre.classification("Entity or Location", "Location", all_classes=classes) == 0.5
    assert lre.retrieval("Paragraph 3", "Paragraph 3") == 1.0
    assert lre.retrieval("Paragraph 3 or 4", "Paragraph 3") == 0.5
    assert lre.count("There are 7 unique", 7) == 1.0
    assert lre.dia(" ace, of course", "ACE") == 1.0 and lre.dia("BOB", "ACE") == 0.0
    # trec / triviaqa / samsum: the first line only (LongBench eval.py)
    assert lre.score_pred("trec", "\nLocation\nEntity", {"answers": ["Location"], "all_classes": classes}) == 1.0


def test_stop_rules_and_answer_cut():
    line = lre.stop_rule("line")
    assert not line("\n Paris") and line(" Paris\nQuestion")
    assert lre.cut("line", "\n Paris\nQuestion: x") == "Paris"
    code = lre.stop_rule("code")
    assert not code("x = 1") and not code("# c\n") and code("# c\nx = 1\n")
    para = lre.stop_rule("para")
    assert not para("\n\nThey met") and para("They met.\n\nQuery")
    assert lre.cut("para", "They met.\n\nQuery") == "They met."
    assert lre.stop_rule(None) is None


def test_loader_prompts_selection_and_truncation(tmp_path):
    _write_data(tmp_path)
    rows = lre.load_rows(tmp_path, "hotpotqa")
    assert [r["idx"] for r in rows] == [0, 1, 2] and rows[0]["answers"] == ["Paris"]
    head, ctx, tails = lre.prompt_parts("hotpotqa", rows[0])
    assert head + ctx + tails["gen"] == lre.LB_PROMPTS["hotpotqa"].format(context=ctx, input="Where is the river?")
    assert lre.task_path(tmp_path, "hotpotqa_e").name == "hotpotqa_e.jsonl"
    mc = lre.load_rows(tmp_path, "en_mc")
    assert [r["gold"] for r in mc] == [0, 1]
    head, ctx, tails = lre.prompt_parts("en_mc", mc[1])
    assert tails["letter"].endswith("D. Light housework\n\nThe letter of the correct answer is")
    assert tails["text"] == "\n\nQuestion: Which chore?\nAnswer:"
    assert lre.task_of("longdialogue_qa_eng")[0] == "en_dia"
    with pytest.raises(SystemExit):
        lre.task_of("nonexistent")
    # the same rows for every model; spread covers the length range
    assert [r["idx"] for r in lre.select(rows, 2, 0, "hotpotqa", "random")] == \
        [r["idx"] for r in lre.select(rows, 2, 0, "hotpotqa", "random")]
    assert [r["idx"] for r in lre.select(rows, 2, 0, "hotpotqa", "spread")][:2] == [0, 2]
    assert [r["idx"] for r in lre.select(rows, 1, 0, "hotpotqa", "longest")][0] == 2
    assert lre.middle_truncate(list(range(10)), 4) == [0, 1, 8, 9]
    assert lre.middle_truncate(list(range(10)), 5) == [0, 1, 7, 8, 9]
    assert lre.middle_truncate(list(range(3)), 5) == [0, 1, 2]
    assert lre.bucket(8191) == "0-8k" and lre.bucket(70000) == "64-128k" and lre.bucket(200000) == "128k+"


def test_option_scoring_from_a_cache_copy_equals_one_forward():
    """The context's cache is prefilled once; each option is scored on a copy (the cache stays reusable) and the
    log-likelihood equals the one of a one-shot forward over context + option, with sparse attention."""
    torch.manual_seed(0)
    model = OellmAmbitiousForCausalLM(_cfg()).eval()
    ctx = torch.randint(1, 97, (70,), generator=torch.Generator().manual_seed(1)).tolist()
    conts = [[5, 9, 11], [7], [3, 3, 3, 3, 3]]
    cache, last = lg.prefill(model, ctx, chunk=16)
    for _ in range(2):  # scoring leaves the cache untouched
        for cont in conts:
            got = lre.continuation_logprob(model, cache, last, cont)
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.tensor([ctx + cont])).logits[0].float(), -1)
            ref = float(sum(lp[len(ctx) - 1 + i, t] for i, t in enumerate(cont)))
            assert got == pytest.approx(ref, abs=1e-4)


def test_greedy_stop_callback():
    torch.manual_seed(0)
    model = OellmAmbitiousForCausalLM(_cfg()).eval()
    ids = torch.randint(1, 97, (30,), generator=torch.Generator().manual_seed(2)).tolist()
    full = lg.greedy(model, ids, 8, None, chunk=7)
    assert lg.greedy(model, ids, 8, None, chunk=7, stop=lambda new: len(new) == 3) == full[:3]


def test_end_to_end_resumable_run_on_a_remote_code_export(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "modules"))
    torch.manual_seed(0)
    cfg = _cfg(auto_map={"AutoConfig": "configuration_oellm_ambitious.OellmAmbitiousConfig",
                         "AutoModelForCausalLM": "modeling_oellm_ambitious.OellmAmbitiousForCausalLM"})
    export = tmp_path / "export"
    OellmAmbitiousForCausalLM(cfg).eval().save_pretrained(export)
    for f in ea.MODEL_CODE:
        shutil.copyfile((ea._PKG / f).resolve(), export / f)
    for f in ("modeling_oellm_hybrid.py", "configuration_oellm_hybrid.py"):
        shutil.copyfile(ea._HERE / "oellm_hybrid" / f, export / f)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(export, trust_remote_code=True).eval()
    assert model.model.layers[0].mixer.sparse is not None
    data, out = tmp_path / "data", tmp_path / "out"
    _write_data(data)
    tasks = ["hotpotqa", "hotpotqa_e", "trec", "samsum", "lcc", "en_mc", "code_debug", "en_qa", "en_dia"]
    summary = lre.run(model, CharTok(), tasks, data, out, max_len=900, n=2)
    recs = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    per = {t: sum(r["task"] == t for r in recs) for t in tasks}
    assert per == {"hotpotqa": 2, "hotpotqa_e": 2, "trec": 2, "samsum": 2, "lcc": 2, "en_mc": 2, "code_debug": 1,
                   "en_qa": 1, "en_dia": 1}
    for r in recs:
        assert 0.0 <= r["score"] <= 1.0
        assert r["prompt_tokens"] <= max(900, r["full_tokens"]) and r["truncated"] == (r["full_tokens"] > 900)
        if r["task"] in ("en_mc", "code_debug"):
            assert set(r["pick"]) == {"letter", "text", "text_norm"} and len(r["logprobs"]["text"]) == 4
        else:
            assert r["gen_tokens"] <= lre.task_of(r["task"])[1].max_new
    assert any(r["truncated"] for r in recs) and not all(r["truncated"] for r in recs)
    assert set(summary) == set(tasks) and summary["en_mc"]["n"] == 2 and "text_norm" in summary["en_mc"]
    assert sum(b["n"] for b in summary["hotpotqa"]["buckets"].values()) == 2
    # resumable: a second invocation adds nothing; another --max-len into the same directory is refused
    lre.run(model, CharTok(), tasks, data, out, max_len=900, n=2)
    assert len((out / "samples.jsonl").read_text().splitlines()) == len(recs)
    with pytest.raises(SystemExit):
        lre.run(model, CharTok(), ["hotpotqa"], data, out, max_len=1000, n=2)


def test_stats_parses_without_torch_needed(tmp_path, capsys):
    _write_data(tmp_path)
    lre.stats(tmp_path, ["hotpotqa", "en_mc", "code_debug", "en_qa", "en_dia"])
    assert "en_mc" in capsys.readouterr().out
