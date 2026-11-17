from pathlib import Path

from omegaconf import OmegaConf

from oellm_autoexp.config.resolvers import register_default_resolvers


def test_lumi_launcher_passes_backend_environment_to_container():
    """Bridge must import the same Megatron fork as the training stage."""
    root = Path(__file__).resolve().parents[2]
    register_default_resolvers(force=True)
    lumi = OmegaConf.load(root / "config/slurm/lumi.yaml")
    config = OmegaConf.merge(
        lumi,
        {
            "backend": {"env": {"PYTHONPATH": "/workspace/oellm-autoexp/submodules/Megatron-LM"}},
            "container": {"runtime": "singularity", "image": "training.sif", "bind": ["/scratch/"]},
        },
    )

    launcher = OmegaConf.to_container(config, resolve=True)["launcher_cmd"]
    assert "singularity exec" in launcher
    assert "--env PYTHONPATH=/workspace/oellm-autoexp/submodules/Megatron-LM" in launcher
    assert "--bind /scratch/" in launcher


def test_lumi_launcher_with_empty_backend_environment():
    root = Path(__file__).resolve().parents[2]
    register_default_resolvers(force=True)
    lumi = OmegaConf.load(root / "config/slurm/lumi.yaml")
    config = OmegaConf.merge(
        lumi,
        {
            "backend": {"env": {}},
            "container": {"runtime": "singularity", "image": "training.sif", "bind": []},
        },
    )

    launcher = OmegaConf.to_container(config, resolve=True)["launcher_cmd"]
    assert "singularity exec" in launcher
    assert "--env PYTHONPATH" not in launcher
