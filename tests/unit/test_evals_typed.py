"""The typed eval launch (scripts/evals): spec loading and checks, the oellm-eval plan it builds, the
runtime for MN5, the rendered pool sweep, the manifest, and the harness commands the plan's units run
(the same as the old run_job.py / mn5_oellm_run_one.sh commands)."""

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
from oellm.runner import commands  # noqa: E402  (launch put submodules/oellm-eval on sys.path)
from oellm.runner.plan import read_plan  # noqa: E402
from oellm.runner.runtime import read_runtime  # noqa: E402
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
    ({"tp": 2, "dp": 4}, r"tp x dp x replicas = 8 > 4"),
    ({"tp": 1, "dp": 1, "replicas": 5, "expert_parallel": False}, r"tp x dp x replicas = 5 > 4"),
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
    export_job, plan, pools = launch.build(spec)
    assert export_job is not None and plan.units and set(pools) == {u.id for u in plan.units}


# ---------------------------------------------------------------- the plan


def test_task_group_rows_are_the_old_suite_files_rows(tmp_path):
    groups = launch.select_rows(_spec(tmp_path, {"evalchemy": {"exclude_tasks": []}}))
    suite = launch.select_rows(_spec(tmp_path, {"rows": {"suite_file": "config/eval_tasks/flag_evals.yaml"},
                                                "evalchemy": {"exclude_tasks": []}}))
    assert len(groups) == 438 and sorted(groups, key=lambda r: r.name) == sorted(suite, key=lambda r: r.name)


def test_full_plan_covers_every_row_once(tmp_path):
    spec = _spec(tmp_path)
    export_job, plan, pools = launch.build(spec)
    rows = launch.select_rows(spec)
    lm = {r.name for r in rows if r.harness == "lm-eval-harness"}
    assert {i.split("@")[0] for i, _ in plan.lm_items} == lm
    grouped = [r.name for u in plan.units for r in u.rows]
    assert sorted(grouped) == sorted(r.name for r in rows if r.harness != "lm-eval-harness")
    assert not any("LiveCodeBench" in n for n in grouped)
    le = [u for u in plan.units if u.kind == "lighteval"]
    assert len(le) == 7 and all(len(u.rows) <= 12 for u in le)
    assert sum(u.kind == "lm_eval_session" for u in plan.units) == 10
    assert plan.root == f"{tmp_path}/out/ambitious_v4_2ph_hd256_fp8_13000"
    assert plan.view_dir == f"{tmp_path}/views/ambitious_v4_2ph_hd256_fp8_13000"
    # not ready: an export job, every unit gated on its marker
    gate = f"{tmp_path}/exports/ambitious_v4_2ph_hd256_fp8_13000/.downstream_ready"
    assert export_job is not None and all(pool["gate"] == gate for pool in pools.values())
    assert pools["lm0"]["priority"] == 60 and pools["lm0"]["time_h"] == 12 and pools["le1"]["priority"] == 55


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
    export_job, plan, pools = launch.build(spec)
    assert not any(i.startswith("vllm/mmlu/5") for i, _ in plan.lm_items)
    assert not any(r.task == "flores200:eng_Latn-mkd_Cyrl" for u in plan.units for r in u.rows)
    assert export_job is None and all(pool["gate"] is None for pool in pools.values())
    assert plan.view_dir == str(exp), "a ready export without a view: the plain model"
    (tmp_path / "views/ambitious_v4_2ph_hd256_fp8_13000").mkdir(parents=True)
    assert launch.build(spec)[1].view_dir == f"{tmp_path}/views/ambitious_v4_2ph_hd256_fp8_13000"


def test_explicit_rows_and_unknown_names(tmp_path):
    spec = _spec(tmp_path, {"rows": {"names": ["vllm/mmlu/5", "vllm/gsm8k/4"], "harnesses": ["lm-eval-harness"]},
                            "lighteval": {"enabled": False}, "evalchemy": {"enabled": False},
                            "lm_eval": {"workers": 8}})
    _, plan, _ = launch.build(spec)
    assert sum(u.kind == "lm_eval_session" for u in plan.units) == min(8, len(plan.lm_items))
    with pytest.raises(SpecError, match="not in"):
        launch.build(_spec(tmp_path, {"rows": {"names": ["vllm/nope/0"]}}))


def test_a_limited_test_run_splits_no_rows(tmp_path):
    _, plan, _ = launch.build(_spec(tmp_path, {"rows": {"limit": 20}}))
    assert plan.lm_items and not any("@" in i for i, _ in plan.lm_items) and plan.root.endswith("_limit20")
    _, plan, _ = launch.build(_spec(tmp_path))
    assert any("@" in i for i, _ in plan.lm_items)


# ---------------------------------------------------------------- the commands of the units


def _unit(tmp_path, kind, data=None):
    _, plan, _ = launch.build(_spec(tmp_path, data))
    return plan, next(u for u in plan.units if u.kind == kind)


def test_lighteval_command_is_the_old_runners(tmp_path):
    plan, unit = _unit(tmp_path, "lighteval")
    Path(plan.view_dir).mkdir(parents=True)  # never for lighteval: it would drop the stop sequences
    (c,) = commands.unit_commands(plan, unit, unit.rows)
    tasks = ",".join(r.task for r in unit.rows)
    shots = ",".join(str(r.n_shot) for r in unit.rows)
    key = "group_" + hashlib.sha1(f"{tasks}\n{shots}\n".encode()).hexdigest()[:10]  # mn5_oellm_run_one.sh
    assert c.argv[3:6] == ["vllm", f"model_name={plan.model_dir},trust_remote_code=True,dtype=bfloat16,"
                           "gpu_memory_utilization=0.9,max_num_seqs=64,max_num_batched_tokens=32768,"
                           "max_model_length=32768,tensor_parallel_size=4",
                           ",".join(f"{r.task}|{r.n_shot}" for r in unit.rows)]
    assert c.argv[c.argv.index("--output-dir") + 1].endswith(f"/lighteval/lighteval_{key}")
    assert "--save-details" in c.argv and c.env["OELLM_EP"] == "1"
    assert c.env["OELLM_GRAPH_MAX_ROWS"] == "64" and c.env["OELLM_VLLM_ENFORCE_EAGER"] == "0"
    plan2, unit2 = _unit(tmp_path, "lighteval", {"lighteval": {"backend": "accelerate"}})
    (c2,) = commands.unit_commands(plan2, unit2, unit2.rows)
    assert c2.argv[3:5] == ["accelerate", f"model_name={plan2.model_dir},trust_remote_code=True,batch_size=32"]


def test_evalchemy_and_lm_eval_commands_are_the_old_runners(tmp_path):
    plan, ec = _unit(tmp_path, "evalchemy")
    calls = commands.unit_commands(plan, ec, ec.rows)
    assert [c.argv[c.argv.index("--tasks") + 1] for c in calls] == [r.task for r in ec.rows]
    args = calls[0].argv[calls[0].argv.index("--model_args") + 1]
    assert args.startswith(f"pretrained={plan.view_dir},")
    assert args.endswith(",tensor_parallel_size=4,enforce_eager=False,enable_expert_parallel=True")
    lm = next(u for u in plan.units if u.kind == "lm_eval_session")
    (s,) = commands.unit_commands(plan, lm, [], shard=3)
    a = s.argv
    for flag, val in [("--tp", "1"), ("--dp", "4"), ("--ep", "1"), ("--max-num-seqs", "256"),
                      ("--request-batch-size", "256"), ("--shard", "3"), ("--gpu-mem", "0.85"),
                      ("--queue", f"{plan.root}/queue_lm"), ("--model", plan.model_dir), ("--root", plan.root)]:
        assert a[a.index(flag) + 1] == val
    assert "--response-cache" in a
    assert s.env["OELLM_ENGRAM_TABLES"] == "uva" and s.env["OELLM_GRAPH_MAX_ROWS"] == "256"


def test_accelerate_backend_is_lighteval_only(tmp_path):
    with pytest.raises(SpecError):
        _spec(tmp_path, {"evalchemy": {"backend": "accelerate"}})


# ---------------------------------------------------------------- the launch files


def test_launch_files_plan_runtime_sweep_and_manifest(tmp_path):
    spec = _spec(tmp_path)
    export_job, plan, pools = launch.build(spec)
    d = tmp_path / "launch"
    cfg_path = launch.write_launch(spec, export_job, plan, pools, d)
    text = cfg_path.read_text()
    cfg = yaml.safe_load(text)
    assert cfg["defaults"][0] == "/experiments/korbi/production/evals/eval_job_mn5"
    assert cfg["hydra"]["searchpath"] == [f"file://{REPO / 'config'}"]
    points = cfg["sweep"]["groups"][0]["configs"]
    assert len(points) == len(plan.units) + 1
    assert points[0]["aux.runtime"] == "plain" and points[0]["aux.unit"] == "-"
    assert load(JobSpec, json.loads(Path(points[0]["aux.spec"]).read_text())) == export_job
    for pt, u in zip(points[1:], plan.units):
        assert (pt["aux.runtime"], pt["aux.spec"], pt["aux.unit"]) == ("worker", str(d / "plan.json"), u.id)
        assert pt["aux.job"] == pools[u.id]["job"] and pt["slurm.sbatch.time"].endswith(":00:00")
    assert "flores200" not in text and "mmlu" not in text
    assert read_plan(d / "plan.json") == plan
    rt = read_runtime(d / "runtime.json")
    assert rt.kind == "local" and rt.python == "${INFER_PY}"
    stage = spec.paths.resolved().oellm_stage
    assert rt.dirs == {"evalchemy": f"{stage}/evalchemy"} and rt.harness_pythonpath["evalchemy"][-1] == stage
    assert "lm-eval-harness" not in rt.harness_pythonpath, "the sessions keep the vLLM install's lm-eval"
    assert rt.cleanup_globs == ["/dev/shm/oellm_engram_*"] and "$DEST/typer" in rt.prune["lighteval"]
    assert rt.hash_files == [f"{stage}/{f}" for f in launch.STAGE_FILES]
    assert rt.require == {"lm-eval-harness": {"lm_eval": "0.4.13"}}, "the image's lm-eval is pinned"
    m = json.loads((d / "manifest.json").read_text())
    assert {u["kind"]: u["model"] for u in m["units"]} == {"lm_eval_session": plan.model_dir,
                                                           "lighteval": plan.model_dir, "evalchemy": plan.view_dir}
    assert m["plan"]["rows"] == len(plan.rows) and m["plan"]["task_groups"] == ["flag-evals-vllm", "flag-evals-lighteval"]
    assert {"plan.json", "runtime.json", "spec.json", "launch.yaml", "jobs/export.json"} <= set(m["files"])
    assert "sync_stamp" in m["code"]["caller"]


def test_apptainer_runtime_for_pools_on_other_clusters(tmp_path):
    images = {"lm-eval-harness": "/v.sif", "lighteval": "/l.sif", "evalchemy": "/v.sif"}
    spec = _spec(tmp_path, {"runtime": {"kind": "apptainer", "images": images, "apptainer_args": ["--nv"]},
                            "slurm": {"job_config": "experiments/korbi/production/evals/eval_job_jupiter"}})
    _, plan, _ = launch.build(spec)
    rt = launch.runtime(spec, plan)
    assert rt.kind == "apptainer" and rt.images == images and rt.dirs == {"evalchemy": "/opt/evalchemy"}
    with pytest.raises(SpecError, match="no image"):
        _spec(tmp_path, {"runtime": {"kind": "apptainer", "images": {"lighteval": "/l.sif"}}})
    jup = _spec(tmp_path, {"runtime": {"kind": "apptainer", "images": images,
                                       "require": {"lm-eval-harness": {"lm_eval": "0.4.12"}}}})
    assert launch.runtime(jup, plan).require == {"lm-eval-harness": {"lm_eval": "0.4.12"}}
    with pytest.raises(SpecError, match="runtime.require: unknown harness"):
        _spec(tmp_path, {"runtime": {"require": {"lm_eval": {"lm_eval": "0.4.13"}}}})


def test_job_spec_roundtrip(tmp_path):
    export_job, _, _ = launch.build(_spec(tmp_path))
    assert load(JobSpec, json.loads(json.dumps(to_dict(export_job)))) == export_job


def test_job_sh_parses():
    assert subprocess.run(["bash", "-n", str(REPO / "scripts/evals/job.sh")]).returncode == 0


@pytest.mark.skipif(not (REPO / "oellm_autoexp").is_dir(), reason="needs the autoexp package")
def test_launch_composes_in_a_run_autoexp_dry_run(tmp_path):
    spec = _spec(tmp_path, {"rows": {"names": ["vllm/mmlu/5", "vllm/AMC23/0", "lighteval/xcopa:it/0"]}})
    export_job, plan, pools = launch.build(spec)
    d = tmp_path / "launch"
    launch.write_launch(spec, export_job, plan, pools, d)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, "scripts/run_autoexp.py", "--dry-run", "--dry-run-root", str(tmp_path / "dry"),
                        "-C", str(d), "--config-name", "launch"], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    scripts = sorted((tmp_path / "dry").rglob("job.sbatch"))
    assert len(scripts) == len(plan.units) + 1
    sb = next(s for s in scripts if s.parent.name == export_job.name).read_text()
    assert f"bash scripts/evals/job.sh plain {d}/jobs/export.json -" in sb and "pylibs_hfx" in sb
    for u in plan.units:
        sb = next(s for s in scripts if s.parent.name == pools[u.id]["job"]).read_text()
        assert f"bash scripts/evals/job.sh worker {d}/plan.json {u.id}" in sb and "pylibs_hfx" not in sb


# ---------------------------------------------------------------- the export job


def test_export_job_steps(tmp_path):
    export_job, _, _ = launch.build(_spec(tmp_path))
    steps = run_job.export_steps(export_job, "PY")
    assert steps[0].argv == ["bash", "scripts/korbi/hf_export/export_main_run_v2.sh", "ambitious_v4_hd256", "13000",
                             f"{tmp_path}/exports"]
    assert steps[0].env == {"EXPORT_ONLY": "1", "EXPERT_DTYPE": "fp8", "HF_EXPORTER": "parallel",
                            "OELLM_EXPORT_MAX_POSITIONS": "100000000"}
    assert steps[-1].argv == ["touch", f"{export_job.export_dir}/.downstream_ready"]
    assert any("prepare_base_model_view.py" in " ".join(s.argv) for s in steps)


def test_export_of_any_run_dir(tmp_path):
    """export.run_dir (e.g. a ckda_branch post-fit arm): export_run.sh into the export dir itself."""
    spec = _spec(tmp_path, {"export": {"step": 23500, "name": "ckdab_bgqa_fp8_23500",
                                        "run_dir": "/runs/ckda_branch/bgqa"}})
    assert spec.export.export_name == "ckdab_bgqa_fp8_23500"
    export_job, _, _ = launch.build(spec)
    steps = run_job.export_steps(export_job, "PY")
    assert steps[0].argv == ["bash", "scripts/korbi/hf_export/export_run.sh", "/runs/ckda_branch/bgqa", "23500",
                             export_job.export_dir]
    assert export_job.export_dir.endswith("/ckdab_bgqa_fp8_23500")
    with pytest.raises(SpecError, match="run_dir needs"):
        _spec(tmp_path, {"export": {"step": 23500, "run_dir": "/runs/x"}}).export.export_name


def test_run_job_rejects_a_bad_spec(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "worker", "name": "x"}))
    assert run_job.main([str(bad)]) == 2
