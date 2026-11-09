"""Shim: oellm-eval's ``oellm.runner.node_ep`` (node-local EP frontends for the lm-eval session)."""
from scripts.korbi.flag_eval import _oellm  # noqa: F401
from oellm.runner.node_ep import distributed_adapter, launch  # noqa: F401,E402
