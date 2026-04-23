"""Tests for template rendering functionality."""

from pathlib import Path

import pytest

from oellm_autoexp.slurm_gen import render_template, render_template_file, SbatchTemplateError


class TestRenderTemplate:
    """Tests for render_template function."""

    def test_basic_replacement(self):
        """Test basic variable replacement."""
        template = "Hello {name}!"
        result = render_template(template, {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_replacements(self):
        """Test multiple variable replacements."""
        template = "#SBATCH --job-name={job_name}\n#SBATCH --nodes={nodes}"
        result = render_template(template, {"job_name": "test", "nodes": "4"})
        assert result == "#SBATCH --job-name=test\n#SBATCH --nodes=4"

    def test_sbatch_template(self):
        """Test rendering a realistic SBATCH template."""
        template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --time={time}
#SBATCH --partition={partition}

module load python/3.10
python {script_path} --epochs {epochs}
"""
        result = render_template(
            template,
            {
                "job_name": "training",
                "nodes": "2",
                "time": "24:00:00",
                "partition": "gpu",
                "script_path": "train.py",
                "epochs": "100",
            },
        )
        assert "#SBATCH --job-name=training" in result
        assert "#SBATCH --nodes=2" in result
        assert "python train.py --epochs 100" in result

    def test_empty_replacements(self):
        """Test template with no placeholders."""
        template = "#!/bin/bash\necho 'hello'"
        result = render_template(template, {})
        assert result == template

    def test_repeated_placeholder(self):
        """Test template with repeated placeholder."""
        template = "{name} is {name}"
        result = render_template(template, {"name": "Claude"})
        assert result == "Claude is Claude"

    def test_missing_placeholder_raises(self):
        """Test that missing placeholder raises SbatchTemplateError."""
        template = "Hello {name}, you are {age} years old"
        with pytest.raises(SbatchTemplateError, match="Missing template variable: age"):
            render_template(template, {"name": "World"})


class TestRenderTemplateFile:
    """Tests for render_template_file function."""

    def test_render_to_file(self, tmp_path: Path):
        """Test rendering a template file to output."""
        template_path = tmp_path / "template.sbatch"
        template_path.write_text("#!/bin/bash\n#SBATCH --job-name={job_name}\necho {message}")

        output_path = tmp_path / "output" / "job.sbatch"
        result = render_template_file(
            template_path,
            output_path,
            {"job_name": "test_job", "message": "hello"},
        )

        assert output_path.exists()
        assert result == "#!/bin/bash\n#SBATCH --job-name=test_job\necho hello"
        assert output_path.read_text() == result

    def test_creates_parent_directories(self, tmp_path: Path):
        """Test that parent directories are created."""
        template_path = tmp_path / "template.sbatch"
        template_path.write_text("#!/bin/bash\necho {msg}")

        output_path = tmp_path / "deep" / "nested" / "path" / "job.sbatch"
        render_template_file(template_path, output_path, {"msg": "test"})

        assert output_path.exists()
        assert output_path.read_text() == "#!/bin/bash\necho test"

    def test_template_not_found(self, tmp_path: Path):
        """Test error when template file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            render_template_file(
                tmp_path / "nonexistent.sbatch",
                tmp_path / "output.sbatch",
                {},
            )

    def test_overwrites_existing_output(self, tmp_path: Path):
        """Test that existing output file is overwritten."""
        template_path = tmp_path / "template.sbatch"
        template_path.write_text("version {version}")

        output_path = tmp_path / "output.sbatch"
        output_path.write_text("old content")

        render_template_file(template_path, output_path, {"version": "2"})
        assert output_path.read_text() == "version 2"
