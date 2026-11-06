"""scripts/korbi/vllm/sitecustomize.py, the eval runtime's startup hook in $DEST: hidden image
packages, FlashInfer's JIT cache in $DEST, the image's own sitecustomize chained, and NLTK's
downloader on a local index (offline compute nodes)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "scripts/korbi/vllm/sitecustomize.py"


def _run(tmp_path: Path, code: str, env: dict | None = None) -> str:
    dest, orig = tmp_path / "dest", tmp_path / "orig"
    dest.mkdir(exist_ok=True)
    orig.mkdir(exist_ok=True)
    shutil.copy(SRC, dest / "sitecustomize.py")
    (orig / "sitecustomize.py").write_text('import os\nos.environ["ORIG_RAN"] = "1"\n')
    (orig / "moonep").mkdir(exist_ok=True)
    (orig / "moonep" / "__init__.py").write_text("raise RuntimeError('must stay hidden')\n")
    extra: list[str] = []
    full = {k: v for k, v in os.environ.items() if not k.startswith(("OELLM_", "FLASHINFER_", "NLTK_"))}
    full.update(env or {})
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    full["PYTHONPATH"] = os.pathsep.join([str(dest), str(orig), *extra, *inherited])
    out = subprocess.run([sys.executable, "-c", code], env=full, capture_output=True, text=True,
                         cwd=tmp_path, check=True)
    return out.stdout.strip()


def test_hides_the_image_packages_and_chains_the_original(tmp_path):
    out = _run(tmp_path, "import importlib.util as u, os\n"
               "print(u.find_spec('moonep') is None, os.environ.get('ORIG_RAN'),"
               " os.environ['FLASHINFER_WORKSPACE_BASE'].endswith('dest'))")
    assert out == "True 1 True"
    # an empty list hides nothing; an explicit FlashInfer base is kept
    out = _run(tmp_path, "import os\ntry:\n import moonep\nexcept RuntimeError: print('real')\n"
               "print(os.environ['FLASHINFER_WORKSPACE_BASE'])",
               {"OELLM_VLLM_HIDE_MODULES": "", "FLASHINFER_WORKSPACE_BASE": "/x"})
    assert out.splitlines() == ["real", "/x"]


@pytest.mark.skipif(importlib.util.find_spec("nltk") is None, reason="nltk not installed")
def test_nltk_downloader_reads_a_local_index(tmp_path):
    data = tmp_path / "nltk_data"
    data.mkdir()
    (data / "index.xml").write_text("<nltk_data><packages/><collections/></nltk_data>")
    code = ("import nltk, nltk.downloader as d\n"
            "print(d.Downloader.DEFAULT_URL); print(d._downloader._url)")
    lines = _run(tmp_path, code, {"NLTK_DATA": str(data)}).splitlines()
    assert lines == [f"file://{data}/index.xml"] * 2
    # without an index.xml the downloader is left alone
    out = _run(tmp_path, "import nltk.downloader as d; print(d.Downloader.DEFAULT_URL)",
               {"NLTK_DATA": str(tmp_path / "none")})
    assert out.startswith("https://")
