from pathlib import Path

from oellm_autoexp.config.evaluator import evaluate
from oellm_autoexp.config.loader import load_config


def test_evaluate_config(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "data" / "sample_config.yaml"
    cfg = load_config(config_path)
    runtime = evaluate(cfg)

    assert runtime.backend.config.class_name == "NullBackend"
    assert runtime.monitor.config.class_name == "NullMonitor"
