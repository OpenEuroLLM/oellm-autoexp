"""Oellm-autoexp's Megatron-Bridge export with two fixes for our 32B
checkpoints.

1. Vocab source: run_export redirects the checkpoint's SentencePiece-only tokenizer
   path to the Qwen3 reference tokenizer (151k vocab); use our 262144-token one.
2. Pipeline layout: training saved its PP4xVPP4 layout ("Et*5|t*4|...|t*3L"). Bridge
   resets pp/vpp to 1 but keeps the layout, so it builds only the first stage
   (embedding + 5 layers). Drop it from the checkpoint args.
"""

import sys
from pathlib import Path

import oellm_autoexp.backends.megatron_bridge.run_export as rx

tok = Path(sys.argv[sys.argv.index("--tokenizer") + 1])
_install = rx._install_tokenizer_fallback


def _prepare(_ref):  # called by run_export after the Bridge imports, before loading
    _install(tok)
    from megatron.bridge.training.mlm_compat import arguments

    load = arguments._load_args_from_checkpoint

    def _single_stage(path):
        args = load(path)
        args.pipeline_model_parallel_layout = None
        args.virtual_pipeline_model_parallel_size = None
        args.num_layers_per_virtual_pipeline_stage = None
        return args

    arguments._load_args_from_checkpoint = _single_stage


rx._install_tokenizer_fallback = _prepare
sys.exit(rx.main())
