"""Shim: oellm-eval's ``oellm.runner.lm_eval_parts`` (document parts of an lm-eval row)."""
from scripts.korbi.flag_eval import _oellm  # noqa: F401
from oellm.runner.lm_eval_parts import merge, parse, part_key, part_samples, raw_metrics, write_part  # noqa: F401,E402
