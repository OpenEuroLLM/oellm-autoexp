# Megatron-Bridge resources

Vendored from `OpenEuroLLM/Megatron-Bridge-utils` to support offline Megatron → HuggingFace checkpoint conversion by way of `MegatronBridgeBackend`.

## Layout

```
configs/<HF_MODEL>/config.json          # HF model config (architecture reference)
templates/<HF_MODEL>/run_config.yaml    # Megatron-Bridge run_config with <<<VOCAB_SIZE>>> placeholder
tokenizers/<HF_MODEL>/                  # tokenizer files (small metadata in git; tokenizer.json downloaded)
```

Tokenizer data files (`tokenizer.json`, `tokenizer.model`, `merges.txt`, `vocab.json`) are not tracked in git — see `tokenizers/.gitignore`. Run `python scripts/download_tokenizers.py` to populate them from the HF Hub.

## Adding a new model

1. Drop the HF `config.json` under `configs/<HF_MODEL>/`.
2. Drop a Megatron-Bridge `run_config.yaml` under `templates/<HF_MODEL>/` (use `<<<VOCAB_SIZE>>>` as the placeholder for the target vocab size).
3. Add the model + tokenizer to `scripts/download_tokenizers.py`'s manifest, then run it to fetch the heavy tokenizer files.
