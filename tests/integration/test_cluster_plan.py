from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.cluster


@pytest.mark.cluster
@pytest.mark.skipif(
    os.getenv("AUTOEXP_CLUSTER_CONFIG") is None,
    reason="AUTOEXP_CLUSTER_CONFIG not set (skip cluster workflow test)",
)
def test_cluster_plan(tmp_path: Path) -> None:
    """Optional cluster test: render a plan manifest using a real SLURM configuration.

    Configure by setting:
      - AUTOEXP_CLUSTER_CONFIG: path or Hydra ref for the cluster config
      - AUTOEXP_MANIFEST_DIR (optional): directory to store the manifest
    """

    config_ref = os.environ["AUTOEXP_CLUSTER_CONFIG"]
    manifest_dir = Path(os.getenv("AUTOEXP_MANIFEST_DIR", tmp_path))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"plan_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"

    cmd = [
        "python",
        "scripts/plan_autoexp.py",
        "--config-ref",
        config_ref,
        "--manifest",
        str(manifest),
    ]
    subprocess.run(cmd, check=True)

    assert manifest.exists()
