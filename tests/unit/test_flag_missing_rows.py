import csv

import yaml

from scripts.korbi.flag_eval import flag_missing_rows as fmr


def _write(tmp_path, tasks, rows):
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"tasks": tasks}))
    out = tmp_path / "x.tasks.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "n_shot", "metric"])
        w.writeheader()
        w.writerows(rows)
    return suite, out


def test_reports_rows_without_any_metric_at_their_n_shot(tmp_path):
    tasks = [{"name": "vllm/a/0", "task": "a", "n_shot": 0, "suite": "lm-eval-harness"},
             {"name": "vllm/b/5", "task": "b", "n_shot": 5, "suite": "lm-eval-harness"},
             {"name": "lighteval/flores200:x/4", "task": "flores200:x", "n_shot": 4, "suite": "lighteval"},
             {"name": "vllm/include_base_44_north_macedonian/0", "task": "include_base_44_north_macedonian",
              "n_shot": 0, "suite": "lm-eval-harness"}]
    rows = [{"task": "a", "n_shot": "0", "metric": "acc"}, {"task": "b", "n_shot": "0", "metric": "acc"},
            {"task": "include_base_44_north macedonian", "n_shot": "0", "metric": "acc"}]
    suite, path = _write(tmp_path, tasks, rows)
    out = tmp_path / "missing.csv"
    assert fmr.main(["--suite", str(suite), "--csv", str(path), "-o", str(out)]) == 0
    assert [r["name"] for r in csv.DictReader(out.open())] == ["vllm/b/5", "lighteval/flores200:x/4"]
    assert fmr.main(["--suite", str(suite), "--csv", str(path), "--strict"]) == 1
