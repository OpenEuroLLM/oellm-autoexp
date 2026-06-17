## Multilingual scaling experiments configurations

To launch experiments:
```
PYTHONPATH=. python scripts/run_autoexp.py   --config-name experiments/multilingual_scaling/<B>_ne
```
where `<B>` denotes NE models size (for example, `0.1B`, `0.2B`, `0.4B`, `0.9B`)
