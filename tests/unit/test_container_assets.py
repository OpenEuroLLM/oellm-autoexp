import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_build_container_user():
    spec = importlib.util.spec_from_file_location(
        "build_container_user", REPO_ROOT / "container" / "build_container_user.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_container_def_template_has_placeholders():
    template_path = REPO_ROOT / "container" / "megatron" / "MegatronTraining.def.in"
    assert template_path.exists()
    template = template_path.read_text()
    assert "${BASE_IMAGE}" in template
    assert "${REPO_ROOT}" in template
    assert "${REQUIREMENTS_PATH}" in template


def test_build_script_references_apptainer():
    script = (REPO_ROOT / "container" / "build_container.sh").read_text()
    assert "$CONTAINER_RUNTIME build" in script
    assert "MegatronTraining" in script


def test_requirements_file_exists():
    requirements = REPO_ROOT / "container" / "megatron" / "requirements_latest.txt"
    assert requirements.exists()
    contents = requirements.read_text().strip().splitlines()
    assert any(line.strip() for line in contents)


def test_write_runtime_sections_persists_def_metadata(tmp_path):
    module = _load_build_container_user()
    sections = {
        "environment": "export PYTHONPATH=/opt/megatron-bridge/src:${PYTHONPATH}\n",
        "runscript": 'cd /workspace/oellm-autoexp\nexec "$@"\n',
        "test": 'python -c "import modelopt"\n',
        "help": "Single image for training and conversion.\n",
        "labels": "Author OpenSci\nVersion 0.1.0\n",
    }
    module.write_runtime_sections(tmp_path, sections)

    env = (tmp_path / ".singularity.d" / "env" / "99-build-user-environment.sh").read_text()
    assert "export PYTHONPATH=/opt/megatron-bridge/src:${PYTHONPATH}" in env

    runscript = (tmp_path / ".singularity.d" / "runscript").read_text()
    assert runscript.startswith("#!/bin/sh\n")
    assert 'exec "$@"' in runscript

    assert (tmp_path / ".singularity.d" / "test").read_text().strip().endswith('import modelopt"')

    # %help is plain text (no shebang, not executable), written to both
    # locations Apptainer/Singularity read: runscript.help and help.
    for name in ("help", "runscript.help"):
        help_file = tmp_path / ".singularity.d" / name
        assert help_file.read_text() == "Single image for training and conversion.\n"
        assert not help_file.read_text().startswith("#!")
        assert not (help_file.stat().st_mode & 0o111)

    labels = json.loads((tmp_path / ".singularity.d" / "labels.json").read_text())
    assert labels == {"Author": "OpenSci", "Version": "0.1.0"}


def test_sandbox_test_cmd_uses_bash_e_without_writable(tmp_path):
    module = _load_build_container_user()
    cmd = module.build_sandbox_test_cmd("singularity", tmp_path, "python -c 'pass'")
    assert cmd[:2] == ["singularity", "exec"]
    assert "--writable" not in cmd
    i = cmd.index("/bin/bash")
    assert cmd[i + 1 : i + 3] == ["-e", "-c"]
    assert cmd[-1] == "python -c 'pass'\n"


def test_write_runtime_sections_merges_existing_labels(tmp_path):
    module = _load_build_container_user()
    dotd = tmp_path / ".singularity.d"
    dotd.mkdir(parents=True)
    (dotd / "labels.json").write_text(json.dumps({"Base": "laif-rocm"}))

    module.write_runtime_sections(tmp_path, {"labels": "Version 0.2.0\n"})

    labels = json.loads((dotd / "labels.json").read_text())
    assert labels == {"Base": "laif-rocm", "Version": "0.2.0"}


def test_lumi_bridge_recipe_keeps_runtime_sections_and_mandatory_modelopt():
    template = (
        REPO_ROOT / "container" / "megatron" / "MegatronTrainingLumi-bridge.def.in"
    ).read_text()
    # Runtime config must live in sections the build helper persists.
    assert "%environment" in template
    assert "/opt/megatron-bridge/src" in template
    # Patched Bridge installed outside the bind-mounted repo path.
    assert (
        "cp -a /workspace/oellm-autoexp/submodules/Megatron-Bridge /opt/megatron-bridge" in template
    )
    assert "/opt/megatron-bridge/src/megatron/bridge/models" in template
    # nvidia-modelopt is mandatory (failure aborts the build) and the %test
    # section smoke-tests the actual imports.
    assert '"nvidia-modelopt"' in template
    assert "exit 1" in template
    assert "%test" in template
    assert "from megatron.bridge import AutoBridge" in template


def test_lumi_chain_convert_stage_uses_bridge_image():
    import yaml

    chain = yaml.safe_load(
        (
            REPO_ROOT
            / "config"
            / "experiments"
            / "korbi"
            / "chain_qwen3_bridge_train_eval_lumi.yaml"
        ).read_text()
    )
    convert = next(c for c in chain["sweep"]["groups"][0]["configs"] if c.get("stage") == "convert")
    # container.image must be set actively (not commented out); otherwise the
    # convert stage runs in the training image, where /opt/megatron-bridge
    # does not exist.
    assert "MegatronTrainingLumi-bridge" in convert["container.image"]
    assert convert["backend.bridge_root"] == "/opt/megatron-bridge"
    assert "/opt/megatron-bridge/src" in convert["backend.env.PYTHONPATH"]
