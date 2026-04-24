"""Tests for SBATCH script generation helpers."""

from pathlib import Path

import pytest

from oellm_autoexp.slurm_gen.generator import (
    build_replacements,
    build_sbatch_directives,
    generate_script,
    merge_slurm_config,
)
from oellm_autoexp.slurm_gen.schema import SbatchConfig, SlurmConfig


def make_config(tmp_path: Path) -> SlurmConfig:
    return SlurmConfig(
        template_path=str(tmp_path / "template.sbatch"),
        script_dir=str(tmp_path / "scripts"),
        log_dir=str(tmp_path / "logs"),
    )


class TestBuildSbatchDirectives:
    """Tests for build_sbatch_directives."""

    def test_builds_directives_with_extras(self, tmp_path: Path):
        config = make_config(tmp_path)
        config.sbatch = SbatchConfig(
            account="acct",
            nodes=True,
            partition="gpu",
            time="1-00:00:00",
        )
        config.sbatch_extra_directives = [
            "--mail-type=END",
            "#SBATCH --exclusive",
        ]

        directives = build_sbatch_directives(config)

        assert "#SBATCH --account=acct" in directives
        assert "#SBATCH --nodes" in directives
        assert "#SBATCH --partition=gpu" in directives
        assert "#SBATCH --time=1-00:00:00" in directives
        assert "#SBATCH --mail-type=END" in directives
        assert "#SBATCH --exclusive" in directives


class TestBuildReplacements:
    """Tests for build_replacements."""

    def test_builds_expected_replacements(self, tmp_path: Path):
        config = make_config(tmp_path)
        config.env = {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "8"}
        config.launcher_cmd = "srun"
        config.srun_opts = "--cpu-bind=cores"
        config.launcher_env_passthrough = True

        replacements = build_replacements(
            config,
            job_name="job",
            log_path="/tmp/log.txt",
            command=["python", "train.py"],
            extra_args=["--epochs", "5"],
        )

        assert replacements["job_name"] == "job"
        assert replacements["log_path"] == "/tmp/log.txt"
        assert replacements["command"] == "python train.py --epochs 5"
        assert replacements["launcher_cmd"] == "srun"
        assert replacements["srun_opts"] == "--cpu-bind=cores"
        assert replacements["launcher_env_passthrough"] == "true"
        assert "export CUDA_VISIBLE_DEVICES=0" in replacements["env_exports"]
        assert "export OMP_NUM_THREADS=8" in replacements["env_exports"]


class TestGenerateScript:
    """Tests for generate_script."""

    def test_generates_script_in_script_dir(self, tmp_path: Path):
        config = make_config(tmp_path)
        template_path = Path(config.template_path)
        template_path.write_text("#!/bin/bash\n{sbatch_directives}\n{env_exports}\n{command}\n")

        script_path = generate_script(
            config,
            job_name="demo",
            log_path=str(tmp_path / "logs" / "demo.log"),
            command=["python", "train.py"],
        )

        assert Path(script_path).exists()
        contents = Path(script_path).read_text()
        assert "#SBATCH --time=0-01:00:00" in contents
        assert "python train.py" in contents

    def test_generates_script_in_custom_script_path(self, tmp_path: Path):
        config = make_config(tmp_path)
        template_path = Path(config.template_path)
        template_path.write_text("#!/bin/bash\n{command}\n")
        custom_path = str(tmp_path / "custom" / "my_script.sbatch")

        script_path = generate_script(
            config,
            job_name="demo",
            script_path=custom_path,
            log_path=str(tmp_path / "logs" / "demo.log"),
            command=["echo", "hello"],
        )

        assert Path(script_path).exists()
        assert "echo hello" in Path(script_path).read_text()

    def test_missing_template_path_raises(self, tmp_path: Path):
        config = make_config(tmp_path)
        config.template_path = ""

        with pytest.raises(ValueError, match="template_path is required"):
            generate_script(
                config,
                job_name="demo",
                log_path=str(tmp_path / "logs" / "demo.log"),
                command=["echo", "hello"],
            )


class TestMergeSlurmConfig:
    """Tests for merge_slurm_config."""

    def test_merge_none_inputs(self):
        assert merge_slurm_config(None, None) == {}
        assert merge_slurm_config(None, {"a": 1}) == {"a": 1}
        assert merge_slurm_config({"a": 1}, None) == {"a": 1}

    def test_merge_nested_dicts(self):
        base = {"sbatch": {"nodes": 2, "partition": "gpu"}, "array": True}
        override = {"sbatch": {"nodes": 4}, "array": False}
        merged = merge_slurm_config(base, override)
        assert merged["sbatch"]["nodes"] == 4
        assert merged["sbatch"]["partition"] == "gpu"
        assert merged["array"] is False
