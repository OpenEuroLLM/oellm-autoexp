"""Shadow of the image's torchtitan for the native export / eval (scripts/korbi/recall_surgery/ladder17b_eval.sh):
lm_scaling.titan_ext registers its train specs at import time when torchtitan imports, which pulls in titan_oellm
and its dependencies (toml, tomli_w, ...). The exporter only needs titan_ext.hybrid_layers (a pure function), so
the registration is skipped by making `import torchtitan` fail."""
raise ImportError("torchtitan is disabled for the native ladder export / eval")
