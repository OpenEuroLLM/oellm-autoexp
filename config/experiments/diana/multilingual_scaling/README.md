## Multilingual scaling experiments configurations

Configurations are currently set up for Leonardo and the Nemotron dataset. They do not yet include the GBSZ and LR grid; this is work in progress.

To launch a set of experiments:
```
python scripts/run_autoexp.py --config-name experiments/diana/mutlilingual_scaling/dense_qwen3_<B>_ne
```
where `<B>` denotes NE models size (e.g., `0.1B`, `0.2B`, `0.4B`, `0.9B`)