"""Grouped FLAG evals: gen_grouped_config.py (one model load per group) and
oellm_rows_done.py (the per-row done markers of a grouped harness call)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _load(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("scripts/korbi/flag_eval/gen_grouped_config.py")
done = _load("scripts/korbi/flag_eval/oellm_rows_done.py")


def _rows():
    return yaml.safe_load((REPO / "config/eval_tasks/flag_evals.yaml").read_text())["tasks"]


def test_every_row_in_exactly_one_group_of_its_own_harness():
    rows = _rows()
    times = json.loads(gen.TIMES.read_text())
    groups = gen.build(rows, times, nodes=16, load_s=600)
    assert len(groups) == 16
    members = sorted(i for g in groups for i in g["rows"])
    assert members == list(range(len(rows)))
    for g in groups:
        assert {rows[i]["suite"] for i in g["rows"]} == {g["suite"]}
    # balanced within a harness: no lm-eval group more than 1.5x another
    lm = [g["seconds"] for g in groups if g["suite"] == "lm-eval-harness"]
    assert max(lm) < 1.5 * min(lm), lm


def test_split_nodes_proportional_and_one_each():
    assert gen.split_nodes({"lm-eval-harness": 90, "lighteval": 9, "evalchemy": 1}, 10) == {
        "lm-eval-harness": 8, "lighteval": 1, "evalchemy": 1,
    }
    assert sum(gen.split_nodes({"lm-eval-harness": 5, "lighteval": 5, "evalchemy": 5}, 7).values()) == 7


def test_rendered_config_lists_match_and_gate_on_the_export():
    rows = _rows()
    groups = gen.build(rows, {}, nodes=6, load_s=600)
    cfg = yaml.safe_load(gen.render(rows, groups, 6))
    assert cfg["defaults"][0] == "flag_eval_pipeline_mn5"
    points = cfg["sweep"]["groups"][0]["configs"]
    assert points[0]["aux.suite"] == "pipeline"
    total = 0
    for p in points[1:]:
        tasks, shots = p["aux.task"].split(","), p["aux.n_shot"].split(",")
        assert len(tasks) == len(shots)
        total += len(tasks)
        assert p["job.start_condition"]["path"].endswith("/.downstream_ready")
    assert total == len(rows)
    # the one task name with a space survives the comma list
    assert any("include_base_44_north macedonian" in p["aux.task"] for p in points[1:])


def test_rows_done_markers_per_harness(tmp_path):
    out_done = tmp_path / "done"
    # lighteval: keys task|n_shot in <results>/results/**/results_*.json
    res = tmp_path / "lighteval_group_x" / "results" / "m"
    res.mkdir(parents=True)
    (res / "results_1.json").write_text(json.dumps({"results": {"flores200:a|4": {}, "all": {}}}))
    rows = "flores200:a|4\nflores200:b|4"
    args = ["--suite", "lighteval", "--results", str(tmp_path / "lighteval_group_x"),
            "--lm-done", str(tmp_path / "nolm"), "--out-done", str(out_done), "--rows", rows]
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert done.main(args) == 0
    assert buf.getvalue().strip() == "flores200:b|4"
    assert (out_done / "flores200_a_4.ok").exists() and not (out_done / "flores200_b_4.ok").exists()
    # evalchemy: keys = task names, anywhere under the output dir
    ev = tmp_path / "evalchemy_group_y" / "sub"
    ev.mkdir(parents=True)
    (ev / "results_2.json").write_text(json.dumps({"results": {"AIME24": {}, "AMC23": {}}}))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        done.main(["--suite", "evalchemy", "--results", str(tmp_path / "evalchemy_group_y"),
                   "--lm-done", "x", "--out-done", str(out_done), "--rows", "AIME24|0\nAMC23|0"])
    assert buf.getvalue().strip() == ""
    # lm-eval: flag_lmeval.py's own marker <task>_<n_shot>.ok (raw task name, spaces kept)
    lm = tmp_path / "lmdone"
    lm.mkdir()
    (lm / "include_base_44_north macedonian_0.ok").write_text("{}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        done.main(["--suite", "lm-eval-harness", "--results", "x", "--lm-done", str(lm),
                   "--out-done", str(out_done),
                   "--rows", "include_base_44_north macedonian|0\narc_challenge|25"])
    assert buf.getvalue().strip() == "arc_challenge|25"
    assert (out_done / "include_base_44_north_macedonian_0.ok").exists()
