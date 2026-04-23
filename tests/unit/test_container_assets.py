from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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
