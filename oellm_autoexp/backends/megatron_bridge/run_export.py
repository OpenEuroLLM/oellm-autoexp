"""End-to-end Megatron → HuggingFace export, ported from ``Megatron-Bridge-
utils/megatron-to-hf.sh``.

Four steps, all in one Python process so the sbatch stays clean:

1. Build a dummy HF model from a local config + reference tokenizer.
2. Copy the Megatron checkpoint to a temp dir, drop a templated
   ``run_config.yaml`` next to it (vocab size of the target tokenizer
   substituted into the template).
3. Shell out to ``Megatron-Bridge/examples/conversion/convert_checkpoints.py``
   in ``export`` mode (this is the actual heavy lifting; needs Bridge's
   PYTHONPATH set up).
4. Patch the exported HF dir with the *target* tokenizer + matching
   ``config.json`` (vocab_size, special-token IDs).

Run as a CLI module:

    python -m oellm_autoexp.backends.megatron_bridge.run_export \\
        --megatron-path /run/torch_dist/iter_0001000 \\
        --hf-path      /run/hf/iter_0001000 \\
        --hf-model     Qwen/Qwen3-0.6B \\
        --tokenizer    Qwen/Qwen3-0.6B \\
        --bridge-root  ./submodules/Megatron-Bridge \\
        --resources    ./oellm_autoexp/postprocess/resources/megatron_bridge

The reference config + ``run_config.yaml`` template + reference
tokenizer are looked up under ``--resources`` by HF model id.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from oellm_autoexp.backends.megatron_bridge.create_dummy_model import build_dummy_model
from oellm_autoexp.backends.megatron_bridge.hf_config_gen import write_hf_config_dir
from oellm_autoexp.backends.megatron_bridge.patch_tokenizer import patch_config_and_tokenizer

LOGGER = logging.getLogger(__name__)


def _resolve_resource(resources: Path, kind: str, name: str) -> Path:
    """Look up <resources>/<kind>/<name>; <name> may contain a '/' (e.g.
    'Qwen/Qwen3-0.6B')."""
    path = resources / kind / name
    if not path.exists():
        raise FileNotFoundError(f"Megatron-Bridge resource not found: {path}")
    return path


def _vocab_size_of(tokenizer_path: str) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return len(tok)


def _stage_checkpoint(
    src: Path, dst_root: Path, run_config_template: Path, vocab_size: int
) -> Path:
    """Copy the source ckpt into dst_root and drop a templated run_config.yaml
    beside it.

    Returns the path of the staged checkpoint
    (dst_root/<basename(src)>).
    """
    dst_root.mkdir(parents=True, exist_ok=True)
    staged = dst_root / src.name
    LOGGER.info("Copying %s -> %s", src, staged)
    shutil.copytree(src, staged)
    run_config_text = run_config_template.read_text().replace("<<<VOCAB_SIZE>>>", str(vocab_size))
    (staged / "run_config.yaml").write_text(run_config_text)
    return staged


def _run_convert(
    bridge_root: Path,
    megatron_path: Path,
    hf_ref_model: Path,
    hf_out: Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Convert via Bridge's Python API directly.

    Older versions of this code shelled out to
    ``examples/conversion/convert_checkpoints.py``, but that script requires a
    ``run_config.yaml`` next to the Megatron checkpoint — Bridge's vendored
    template path references a class that no longer exists in upstream Bridge,
    and writing a fresh template is brittle to upstream API churn. The
    underlying ``AutoBridge.export_ckpt`` already supports legacy MLM
    checkpoints (it falls back to ``_load_args_from_checkpoint`` when there is
    no run_config.yaml), so we call it directly.

    The Bridge import lives behind a runtime check: this whole step only works
    inside a container with Megatron-Bridge installed (see
    ``container/megatron/MegatronTraining-JUPITER-bridge.def.in``).
    """
    # Some installed `nvidia_resiliency_ext` versions don't expose
    # `__version__`; Megatron-LM's `dist_checkpointing/strategies/nvrx.py`
    # reads it unconditionally during module import and crashes. Patch
    # before any megatron.core import.
    try:
        import nvidia_resiliency_ext as _nvrx

        if not hasattr(_nvrx, "__version__"):
            # Bridge's vendored Megatron-LM asserts >= 0.6.0; lie if needed
            # so the dist-ckpt strategies module imports without async support.
            _nvrx.__version__ = "0.6.0"
    except ImportError:
        pass

    # Bridge's src must come AFTER any source that provides `megatron.core`,
    # otherwise Python's namespace package lookup finds `megatron/bridge`
    # first and shadows `megatron.core`. If `megatron.core` is already
    # importable (typical: PYTHONPATH points at the OpenEuroLLM fork at
    # submodules/Megatron-LM, or the training container ships it via pip),
    # leave it alone — only fall back to Bridge's vendored 3rdparty copy
    # when nothing else provides it. That fallback can crash on
    # version-skewed `nvidia_resiliency_ext`, so the prefer-existing path
    # is important.
    try:
        import megatron.core  # noqa: F401
    except ImportError:
        thirdparty = bridge_root / "3rdparty" / "Megatron-LM"
        if str(thirdparty) not in sys.path:
            sys.path.insert(0, str(thirdparty))
    src_path = bridge_root / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

    # Older OpenEuroLLM Megatron-LM forks lack
    # `_clean_metadata_for_serialization` (Bridge added it upstream). Shim
    # it as a no-op so Bridge's `checkpointing` module can import.
    try:
        from megatron.core.dist_checkpointing import utils as _mc_dc_utils

        if not hasattr(_mc_dc_utils, "_clean_metadata_for_serialization"):
            _mc_dc_utils._clean_metadata_for_serialization = lambda *a, **kw: None
    except ImportError:
        pass

    # Bridge expects `megatron.core.optimizer.layer_wise_optimizer.LayerWiseDistributedOptimizer`,
    # which the OpenEuroLLM fork doesn't ship. Bridge only uses it for
    # `isinstance` checks against the 'torch' (legacy) checkpoint format —
    # our torch_dist export never triggers that branch — so registering a
    # placeholder type is sufficient.
    try:
        import importlib
        import types

        _opt_pkg = importlib.import_module("megatron.core.optimizer")
        try:
            importlib.import_module("megatron.core.optimizer.layer_wise_optimizer")
        except ImportError:
            _layer_wise_mod = types.ModuleType("megatron.core.optimizer.layer_wise_optimizer")
            _layer_wise_mod.LayerWiseDistributedOptimizer = type(
                "LayerWiseDistributedOptimizer", (), {}
            )
            sys.modules["megatron.core.optimizer.layer_wise_optimizer"] = _layer_wise_mod
            _opt_pkg.layer_wise_optimizer = _layer_wise_mod  # type: ignore[attr-defined]
    except ImportError:
        pass

    try:
        from megatron.bridge import AutoBridge
        from megatron.bridge.training.model_load_save import (
            load_megatron_model as _load_megatron_model,
            temporary_distributed_context,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Megatron-Bridge is not importable in this Python environment. "
            "Run inside the bridge-enabled apptainer image "
            "(container/megatron/MegatronTraining-JUPITER-bridge.def.in)."
        ) from exc

    LOGGER.info("Loading AutoBridge from %s", hf_ref_model)
    bridge = AutoBridge.from_hf_pretrained(str(hf_ref_model), trust_remote_code=True)

    # AutoBridge.export_ckpt() calls load_megatron_model without passing
    # model_type; for legacy MegatronLM checkpoints (no run_config.yaml) this
    # produces `model_type=None` and the build_and_load_model assertion
    # `model_type in ("gpt", "mamba")` fails. Inline the equivalent flow so we
    # can pass model_type="gpt" explicitly.
    LOGGER.info("Loading Megatron checkpoint: %s", megatron_path)
    with temporary_distributed_context(backend="gloo"):
        megatron_model = _load_megatron_model(
            str(megatron_path),
            model_type="gpt",
            use_cpu_init=True,
            skip_temp_dist_context=True,
        )
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Bridge's exporter reads `model_config.share_embeddings_and_output_weights`
        # without a default; legacy MegatronLM checkpoints produce a
        # TransformerConfig that doesn't have it. Derive from
        # `untie_embeddings_and_output_weights` (the negation) when missing.
        for _m in megatron_model:
            cfg = getattr(_m, "config", None)
            if cfg is not None and not hasattr(cfg, "share_embeddings_and_output_weights"):
                untie = getattr(cfg, "untie_embeddings_and_output_weights", False)
                cfg.share_embeddings_and_output_weights = not untie
                LOGGER.info(
                    "Patched model_config.share_embeddings_and_output_weights = %s",
                    cfg.share_embeddings_and_output_weights,
                )

        LOGGER.info("Saving HuggingFace export: %s", hf_out)
        bridge.save_hf_pretrained(megatron_model, str(hf_out))


def run_export(
    *,
    megatron_path: Path,
    hf_path: Path,
    hf_model: str,
    tokenizer: str,
    bridge_root: Path,
    resources: Path,
    keep_staging: bool = False,
    derive_hf_arch: str | None = None,
    megatron_config: Path | dict | None = None,
) -> None:
    """Run the full 4-step conversion pipeline.

    Set ``derive_hf_arch`` to e.g. ``"qwen3"`` and pass ``megatron_config``
    (path to YAML or pre-loaded dict) to synthesise the HF reference config
    on the fly from the Megatron training config — bypasses the vendored
    ``resources/configs/<hf_model>/`` snapshot.
    """
    if hf_path.exists():
        raise FileExistsError(f"hf-path {hf_path} already exists; refusing to clobber")
    hf_path.parent.mkdir(parents=True, exist_ok=True)

    if derive_hf_arch:
        config_dir = None  # set below after staging dir is created
    else:
        config_dir = _resolve_resource(resources, "configs", hf_model)
    _ = _resolve_resource(resources, "templates", hf_model) / "run_config.yaml"
    reference_tokenizer = _resolve_resource(resources, "tokenizers", hf_model)
    target_tokenizer = (
        _resolve_resource(resources, "tokenizers", tokenizer)
        if (resources / "tokenizers" / tokenizer).exists()
        else tokenizer  # fall back to HF Hub id
    )

    with tempfile.TemporaryDirectory(prefix="mb_export_") as tmp:
        tmp = Path(tmp)
        dummy_dir = tmp / "dummy_hf_model"

        if derive_hf_arch:
            megatron_dict = _load_megatron_config(megatron_config)
            tok_for_vocab = (
                str(target_tokenizer) if isinstance(target_tokenizer, Path) else target_tokenizer
            )
            vocab = _vocab_size_of(tok_for_vocab)
            generated_dir = tmp / "generated_hf_config"
            LOGGER.info("Step 1/4 (auto-gen): synthesise HF config for arch=%s", derive_hf_arch)
            config_dir = write_hf_config_dir(
                arch=derive_hf_arch,
                megatron=megatron_dict,
                vocab_size=vocab,
                tokenizer_src=reference_tokenizer,
                outdir=generated_dir,
            )
        LOGGER.info("Step 1/4: build dummy HF model from %s + %s", config_dir, reference_tokenizer)
        build_dummy_model(config_dir, reference_tokenizer, dummy_dir)

        LOGGER.info(
            "Step 2/4: skipped (we call bridge.export_ckpt directly, no run_config.yaml needed)"
        )

        LOGGER.info("Step 3/4: bridge.export_ckpt (in-process)")
        _run_convert(bridge_root, megatron_path, dummy_dir, hf_path)

        LOGGER.info("Step 4/4: patch HF export to use target tokenizer %s", target_tokenizer)
        patch_config_and_tokenizer(hf_path=hf_path, tokenizer_path=str(target_tokenizer))

        if keep_staging:
            persist = hf_path.parent / (hf_path.name + ".staging")
            LOGGER.info("Preserving staging dir at %s", persist)
            shutil.copytree(tmp, persist)


def _load_megatron_config(src: Path | dict | None) -> dict:
    if src is None:
        raise ValueError("--megatron-config is required when --derive-hf-arch is set")
    if isinstance(src, dict):
        return src
    src = Path(src)
    text = src.read_text()
    # Try yaml first (oellm-autoexp's rendered job config), fall back to json.
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    # The orchestrator's `config-<jobid>.yaml` / `current.yaml` wraps the
    # resolved root under a top-level `config:` key. Megatron is nested at
    # `[config.]backend.megatron`. Try both layouts.
    if isinstance(data, dict):
        root = data.get("config", data)
        if isinstance(root, dict):
            backend = root.get("backend")
            if isinstance(backend, dict) and isinstance(backend.get("megatron"), dict):
                return backend["megatron"]
    return data


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--megatron-path", required=True, type=Path)
    ap.add_argument("--hf-path", required=True, type=Path)
    ap.add_argument("--hf-model", required=True, help="HF model id; key under resources/configs/")
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer id or path; resolved under resources/tokenizers/ first, then treated as a HF id",
    )
    ap.add_argument(
        "--bridge-root",
        required=True,
        type=Path,
        help="Path to the Megatron-Bridge submodule (clone of NVIDIA-NeMo/Megatron-Bridge)",
    )
    ap.add_argument(
        "--resources",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "postprocess"
        / "resources"
        / "megatron_bridge",
        help="Root of the bridge resources tree (configs/, templates/, tokenizers/)",
    )
    ap.add_argument(
        "--derive-hf-arch",
        default=None,
        help="When set (e.g. 'qwen3'), generate the HF reference config on the fly from --megatron-config instead of using resources/configs/<hf-model>/config.json.",
    )
    ap.add_argument(
        "--megatron-config",
        type=Path,
        default=None,
        help="Path to the resolved Megatron training config YAML (typically <train_dir>/config-<jobid>.yaml). Required when --derive-hf-arch is set.",
    )
    ap.add_argument(
        "--keep-staging", action="store_true", help="Preserve temp staging dir for debugging"
    )
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse()
    run_export(
        megatron_path=args.megatron_path,
        hf_path=args.hf_path,
        hf_model=args.hf_model,
        tokenizer=args.tokenizer,
        bridge_root=args.bridge_root,
        resources=args.resources,
        keep_staging=args.keep_staging,
        derive_hf_arch=args.derive_hf_arch,
        megatron_config=args.megatron_config,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
