"""Tests for configuration schema."""

from oellm_autoexp.slurm_gen import SlurmConfig, SbatchConfig, SrunConfig


class TestSlurmConfig:
    """Tests for SlurmConfig dataclass."""

    def test_create_with_required_fields(self):
        """Test creating config with required fields."""
        config = SlurmConfig(
            template_path="templates/job.sbatch",
            script_dir="/tmp/scripts",
            log_dir="/tmp/logs",
        )
        assert config.template_path == "templates/job.sbatch"
        assert config.script_dir == "/tmp/scripts"
        assert config.log_dir == "/tmp/logs"

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SlurmConfig(
            template_path="templates/job.sbatch",
            script_dir="/tmp/scripts",
            log_dir="/tmp/logs",
        )
        assert config.array is False
        assert config.launcher_cmd == ""
        assert config.srun_opts == ""
        assert config.launcher_env_passthrough is False
        assert config.env == {}
        assert config.command == []
        assert config.sbatch_extra_directives == []
        assert config.test_only is False

    def test_nested_sbatch_config(self):
        """Test nested sbatch configuration."""
        config = SlurmConfig(
            template_path="templates/job.sbatch",
            script_dir="/tmp/scripts",
            log_dir="/tmp/logs",
            sbatch=SbatchConfig(
                account="myaccount",
                nodes=4,
                partition="gpu",
                time="24:00:00",
            ),
        )
        assert config.sbatch.account == "myaccount"
        assert config.sbatch.nodes == 4
        assert config.sbatch.partition == "gpu"

    def test_environment_variables(self):
        """Test setting environment variables."""
        config = SlurmConfig(
            template_path="templates/job.sbatch",
            script_dir="/tmp/scripts",
            log_dir="/tmp/logs",
            env={"CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "8"},
        )
        assert config.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert config.env["OMP_NUM_THREADS"] == "8"


class TestSbatchConfig:
    """Tests for SbatchConfig dataclass."""

    def test_default_time(self):
        """Test default time value."""
        config = SbatchConfig()
        assert config.time == "0-01:00:00"

    def test_all_fields(self):
        """Test setting all fields."""
        config = SbatchConfig(
            account="project123",
            nodes=8,
            partition="gpu-large",
            qos="high",
            time="7-00:00:00",
        )
        assert config.account == "project123"
        assert config.nodes == 8
        assert config.partition == "gpu-large"
        assert config.qos == "high"
        assert config.time == "7-00:00:00"


class TestSrunConfig:
    """Tests for SrunConfig dataclass."""

    def test_empty_config(self):
        """Test creating empty srun config."""
        config = SrunConfig()
        # Should work without any fields
        assert config is not None
