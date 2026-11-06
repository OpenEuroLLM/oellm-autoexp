"""The typed eval launch (scripts/evals): spec loading and checks, the launch plan, the rendered pool
sweep and the harness commands of run_job.py (the same as mn5_oellm_run_one.sh / run_flag_fast_mn5.sh)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.evals import launch, run_job  # noqa: E402
from scripts.evals.spec import (  # noqa: E402
    EvalSpec, JobSpec, SpecError, eval_spec, load, merge, to_dict,
)


def _spec(tmp_path: Path, data: dict | None = None, sets: list[str] | None = None) -> EvalSpec:
    base = {"export": {"step": 13000},
            "paths": {"out": str(tmp_path / "out"), "exports": str(tmp_path / "exports"),
                      "views": str(tmp_path / "views"), "pool": str(tmp_path / "pool")}}
    return eval_spec(merge(base, data or {}), sets)


# ---------------------------------------------------------------- loading


def test_unknown_key_is_an_error_naming_its_path(tmp_path):
    with pytest.raises(SpecError, match=r"lm_eval\.engine: unknown key\(s\) \['max_seqs'\]"):
        _spec(tmp_path, {"lm_eval": {"engine": {"max_seqs": 128}}})


@pytest.mark.parametrize("data, match", [
    ({"lm_eval": {"workers": "10"}}, r"lm_eval\.workers: expected an integer"),
    ({"lm_eval": {"workers": True}}, r"lm_eval\.workers: expected an integer"),
    ({"lm_eval": {"engine": {"expert_parallel": 1}}}, r"expert_parallel: expected true/false"),
    ({"lighteval": {"model_code": {"engram_tables": "shm"}}}, r"engram_tables: 'shm' is not one of"),
    ({"rows": {"harnesses": ["lm-eval"]}}, r"rows\.harnesses\[0\]: 'lm-eval' is not one of"),
    ({"export": {"model": "ambitious_v5"}}, r"export\.model"),
])
def test_wrong_types_and_values_are_errors(tmp_path, data, match):
    with pytest.raises(SpecError, match=match):
        _spec(tmp_path, data)


def test_a_partial_section_keeps_that_sections_defaults(tmp_path):
    s = _spec(tmp_path, {"lm_eval": {"engine": {"max_num_seqs": 128}}})
    assert (s.lm_eval.engine.tp, s.lm_eval.engine.dp, s.lm_eval.engine.max_num_seqs) == (1, 4, 128)
    assert s.lm_eval.engine.request_batch_size == 256 and s.lm_eval.engine.response_cache
    # the grouped harnesses have their own defaults
    assert (s.lighteval.engine.tp, s.lighteval.engine.max_model_length, s.lighteval.max_rows) == (4, 32768, 12)
    assert s.evalchemy.engine.max_model_length is None
    assert s.evalchemy.exclude_tasks == ["LiveCodeBench"]


def test_set_overrides_parse_their_values_as_yaml(tmp_path):
    s = _spec(tmp_path, sets=["lm_eval.workers=12", "lighteval.enabled=false", "rows.exclude=[vllm/mmlu/5]"])
    assert s.lm_eval.workers == 12 and s.lighteval.enabled is False and s.rows.exclude == ["vllm/mmlu/5"]
    with pytest.raises(SpecError, match="expected key=value"):
        _spec(tmp_path, sets=["lm_eval.workers"])


@pytest.mark.parametrize("engine, match", [
    ({"tp": 2, "dp": 4}, r"tp x dp = 8 > 4"),
    ({"dp": 4, "expert_parallel": False}, r"dp > 1 needs expert_parallel"),
    ({"request_batch_size": None}, r"response_cache needs a request_batch_size"),
    ({"max_model_length": 65536}, r"max_model_length"),
])
def test_engine_rules(tmp_path, engine, match):
    with pytest.raises(SpecError, match=match):
        _spec(tmp_path, {"lm_eval": {"engine": engine}})


def test_grouped_harnesses_take_no_lm_eval_only_settings(tmp_path):
    with pytest.raises(SpecError, match=r"lighteval\.engine: request_batch_size"):
        _spec(tmp_path, {"lighteval": {"engine": {"request_batch_size": 64}}})


def test_export_name_follows_model_and_step(tmp_path):
    assert _spec(tmp_path).export.export_name == "ambitious_v4_2ph_hd256_fp8_13000"
    assert _spec(tmp_path, {"export": {"model": "ambitious_v4"}}).export.export_name == "ambitious_v4_2ph_fp8_13000"
    assert _spec(tmp_path, {"export": {"name": "q35c_gqa_32000"}}).export.export_name == "q35c_gqa_32000"
    with pytest.raises(SpecError, match="export: give step"):
        eval_spec({})


@pytest.mark.parametrize("path", sorted((REPO / "config/evals").glob("*.yaml")))
def test_shipped_specs_load_and_plan(path, tmp_path):
    data = yaml.safe_load(path.read_text())
    spec = eval_spec(data, ["export.step=13000", f"paths.out={tmp_path}", f"paths.exports={tmp_path}"])
    jobs, items, done = launch.plan(spec)
    assert jobs and not done
    assert any(j.kind == "export" for j, _ in jobs)


# ---------------------------------------------------------------- the plan


def test_full_plan_covers_every_row_once(tmp_path):
    spec = _spec(tmp_path)
    jobs, items, done = launch.plan(spec)
    rows = launch.select_rows(spec)
    lm = {r.name for r in rows if r.suite == "lm-eval-harness"}
    assert {i.split("@")[0] for i, _ in items} == lm
    grouped = [r.name for j, _ in jobs for r in j.rows]
    assert sorted(grouped) == sorted(r.name for r in rows if r.suite != "lm-eval-harness")
    assert not any("LiveCodeBench" in n for n in grouped)
    le = [j for j, _ in jobs if j.harness == "lighteval"]
    assert len(le) == 7 and all(len(j.rows) <= 12 for j in le)
    assert len([j for j, _ in jobs if j.kind == "lm_eval_worker"]) == 10
    # not ready: an export job first, every eval job gated on its marker
    assert jobs[0][0].kind == "export" and jobs[0][1]["gate"] is None
    gate = f"{tmp_path}/exports/ambitious_v4_2ph_hd256_fp8_13000/.downstream_ready"
    assert all(pool["gate"] == gate for j, pool in jobs[1:])


def test_done_rows_are_left_out_and_a_ready_export_runs_no_export_job(tmp_path):
    spec = _spec(tmp_path)
    root = tmp_path / "out/ambitious_v4_2ph_hd256_fp8_13000"
    (root / "done").mkdir(parents=True)
    (root / "done/mmlu_5.ok").touch()
    (root / "lighteval/done").mkdir(parents=True)
    (root / "lighteval/done/flores200_eng_Latn-mkd_Cyrl_4.ok").touch()
    exp = tmp_path / "exports/ambitious_v4_2ph_hd256_fp8_13000"
    exp.mkdir(parents=True)
    (exp / ".downstream_ready").touch()
    jobs, items, done = launch.plan(spec)
    assert {r.name for r in done} == {"vllm/mmlu/5", "lighteval/flores200:eng_Latn-mkd_Cyrl/4"}
    assert not any(i.startswith("vllm/mmlu/5") for i, _ in items)
    assert not any(r.task == "flores200:eng_Latn-mkd_Cyrl" for j, _ in jobs for r in j.rows)
    assert all(j.kind != "export" and pool["gate"] is None for j, pool in jobs)


def test_explicit_rows_and_unknown_names(tmp_path):
    spec = _spec(tmp_path, {"rows": {"names": ["vllm/mmlu/5", "vllm/gsm8k/4"], "harnesses": ["lm-eval-harness"]},
                            "lighteval": {"enabled": False}, "evalchemy": {"enabled": False},
                            "lm_eval": {"workers": 8}})
    jobs, items, _ = launch.plan(spec)
    workers = [j for j, _ in jobs if j.kind == "lm_eval_worker"]
    assert len(workers) == min(8, len(items))
    with pytest.raises(SpecError, match="not in"):
        launch.plan(_spec(tmp_path, {"rows": {"names": ["vllm/nope/0"]}}))


# ---------------------------------------------------------------- rendering


def test_rendered_sweep_has_only_literals_and_no_task_names(tmp_path):
    spec = _spec(tmp_path)
    jobs, _, _ = launch.plan(spec)
    cfg_path = launch.write_launch(spec, jobs, tmp_path / "launch")
    text = cfg_path.read_text()
    cfg = yaml.safe_load(text)
    points = cfg["sweep"]["groups"][0]["configs"]
    assert len(points) == len(jobs)
    assert cfg["defaults"][0] == f"/{launch.JOB_CONFIG}"
    assert cfg["hydra"]["searchpath"] == [f"file://{REPO / 'config'}"]
    for pt, (job, _) in zip(points, jobs):
        assert isinstance(pt["slurm.sbatch.time"], str) and pt["slurm.sbatch.time"].endswith(":00:00")
        assert pt["aux.spec"] == str(tmp_path / "launch/jobs" / f"{job.name}.json")
        assert launch.SAFE.match(pt["aux.spec"])
        assert load(JobSpec, json.loads(Path(pt["aux.spec"]).read_text())) == job
    assert "flores200" not in text and "mmlu" not in text


def test_job_spec_roundtrip(tmp_path):
    jobs, _, _ = launch.plan(_spec(tmp_path))
    for job, _ in jobs:
        assert load(JobSpec, json.loads(json.dumps(to_dict(job)))) == job


@pytest.mark.skipif(not (REPO / "oellm_autoexp").is_dir(), reason="needs the autoexp package")
def test_launch_composes_in_a_run_autoexp_dry_run(tmp_path):
    spec = _spec(tmp_path, {"rows": {"names": ["vllm/mmlu/5", "vllm/AMC23/0", "lighteval/xcopa:it/0"]}})
    jobs, _, _ = launch.plan(spec)
    d = tmp_path / "launch"
    launch.write_launch(spec, jobs, d)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, "scripts/run_autoexp.py", "--dry-run", "--dry-run-root", str(tmp_path / "dry"),
                        "-C", str(d), "--config-name", "launch"], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    scripts = sorted((tmp_path / "dry").rglob("job.sbatch"))
    assert len(scripts) == len(jobs)
    for job, _ in jobs:
        sb = next(s for s in scripts if s.parent.name == job.name).read_text()
        runtime = "plain" if job.kind == "export" else "vllm"
        assert f"bash scripts/evals/job.sh {runtime} {d}/jobs/{job.name}.json" in sb
        assert ("pylibs_hfx" in sb) == (job.kind == "export")


# ---------------------------------------------------------------- run_job commands


def _job(tmp_path, kind_filter, **spec_over):
    jobs, _, _ = launch.plan(_spec(tmp_path, spec_over or None))
    return next(j for j, _ in jobs if kind_filter(j))


def test_lighteval_group_command_matches_the_shell_runner(tmp_path):
    job = _job(tmp_path, lambda j: j.harness == "lighteval")
    steps = run_job.steps_of(job, "PY")
    call, done = steps
    rows = job.rows
    tasks = ",".join(r.task for r in rows)
    shots = ",".join(str(r.n_shot) for r in rows)
    key = "group_" + hashlib.sha1(f"{tasks}\n{shots}\n".encode()).hexdigest()[:10]  # mn5_oellm_run_one.sh
    assert call.argv[2:5] == ["vllm", f"model_name={job.export_dir},trust_remote_code=True,dtype=bfloat16,"
                              "gpu_memory_utilization=0.9,max_num_seqs=64,max_num_batched_tokens=32768,"
                              "max_model_length=32768,tensor_parallel_size=4",
                              ",".join(f"{r.task}|{r.n_shot}" for r in rows)]
    assert call.argv[-1].endswith(f"/lighteval/lighteval_{key}")
    assert call.env["OELLM_GRAPH_MAX_ROWS"] == "64" and call.env["OELLM_VLLM_ENFORCE_EAGER"] == "0"
    assert done.check and done.argv[done.argv.index("--results") + 1] == call.argv[-1]


def test_a_view_is_used_where_there_is_one(tmp_path):
    job = _job(tmp_path, lambda j: j.harness == "evalchemy")
    Path(job.view_dir).mkdir(parents=True)
    call = run_job.steps_of(job, "PY")[0]
    assert f"pretrained={job.view_dir}," in call.argv[call.argv.index("--model_args") + 1]


def test_evalchemy_runs_one_call_per_task_and_skips_done_rows(tmp_path):
    job = _job(tmp_path, lambda j: j.harness == "evalchemy")
    done_dir = Path(job.out) / job.export_name / "evalchemy/done"
    done_dir.mkdir(parents=True)
    (done_dir / f"{job.rows[0].task}_0.ok").touch()
    steps = run_job.steps_of(job, "PY")
    calls = [s for s in steps if "eval.eval" in s.argv]
    assert [c.argv[c.argv.index("--tasks") + 1] for c in calls] == [r.task for r in job.rows[1:]]
    args = calls[0].argv[calls[0].argv.index("--model_args") + 1]
    assert args.endswith(",tensor_parallel_size=4,enforce_eager=False,enable_expert_parallel=True")
    assert all(c.cwd.endswith("oellm_mn5_stage/evalchemy") for c in calls)


def test_lm_eval_worker_command(tmp_path):
    job = _job(tmp_path, lambda j: j.kind == "lm_eval_worker" and j.shard == 3)
    (s,) = run_job.steps_of(job, "PY")
    a = s.argv
    for flag, val in [("--tp", "1"), ("--dp", "4"), ("--ep", "1"), ("--max-num-seqs", "256"),
                      ("--request-batch-size", "256"), ("--shard", "3"), ("--gpu-mem", "0.85"),
                      ("--queue", f"{job.out}/{job.export_name}/queue_lm")]:
        assert a[a.index(flag) + 1] == val
    assert "--response-cache" in a and "--names" not in a
    assert s.env["OELLM_ENGRAM_TABLES"] == "uva" and s.env["OELLM_GRAPH_MAX_ROWS"] == "256"


def test_export_job_steps(tmp_path):
    job = _job(tmp_path, lambda j: j.kind == "export")
    steps = run_job.steps_of(job, "PY")
    assert steps[0].argv == ["bash", "scripts/korbi/hf_export/export_main_run_v2.sh", "ambitious_v4_hd256", "13000",
                             f"{tmp_path}/exports"]
    assert steps[0].env == {"EXPORT_ONLY": "1", "EXPERT_DTYPE": "fp8", "HF_EXPORTER": "parallel",
                            "OELLM_EXPORT_MAX_POSITIONS": "100000000"}
    assert steps[-1].argv == ["touch", f"{job.export_dir}/.downstream_ready"]
    assert any("prepare_base_model_view.py" in " ".join(s.argv) for s in steps)


def test_run_job_rejects_a_bad_spec(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "worker", "name": "x"}))
    assert run_job.main([str(bad)]) == 2
