"""HF resource templates for checkpoint conversion."""

import shutil
from importlib import resources as pkg_resources
from pathlib import Path


def prepare_resources(architecture: str, output_dir: str) -> None:
    """Copy architecture-specific HF template files (config, tokenizer, modeling)
    to *output_dir* so that ``AutoConfig.from_pretrained`` can load them.
    """
    resource_root = pkg_resources.files("oellm_autoexp.postprocess.resources") / architecture
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for item in resource_root.iterdir():
        if item.name in ("__init__.py", "__pycache__"):
            continue
        target = dest / item.name
        if item.is_file():
            with pkg_resources.as_file(item) as src_path:
                shutil.copy2(src_path, target)
        elif item.is_dir():
            with pkg_resources.as_file(item) as src_path:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(src_path, target)
