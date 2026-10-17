"""Megatron argument values survive the shell launch command intact."""

import shutil
import subprocess

import pytest

from oellm_autoexp.argparse_schema.converter import build_cmdline_args, shell_token
from oellm_autoexp.backends.megatron.cli_metadata import (
    MEGATRON_ACTION_SPECS,
    MEGATRON_ARG_METADATA,
)

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _through_bash(token: str, env: dict | None = None) -> str:
    return subprocess.run(
        ["bash", "-c", f"printf '%s' {token}"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
        check=True,
    ).stdout


@pytest.mark.parametrize("plain", ["plain", "/p/a-th_1.2:x", "1e-4", "-5", "key=value,other"])
def test_plain_tokens_unchanged(plain):
    # Ordinary commands must render byte-identically to before.
    assert shell_token(plain) == plain


@needs_bash
@pytest.mark.parametrize(
    "value",
    ["L-L-*-|L-L-*-", "a b", "it's", 'say "hi"', "", "--*-", "back\\slash", "semi;colon", "g[0-9]"],
)
def test_special_values_round_trip(value):
    assert _through_bash(shell_token(value)) == value


@needs_bash
def test_dollar_values_still_expand():
    env = {"HOME": "/h", "SLURM_JOB_ID": "42"}
    assert _through_bash(shell_token("${HOME}/x y"), env) == "/h/x y"
    assert _through_bash(shell_token("$SLURM_JOB_ID|run"), env) == "42|run"


def test_hybrid_pattern_renders_as_one_shell_word():
    meta, specs = dict(MEGATRON_ARG_METADATA), dict(MEGATRON_ACTION_SPECS)
    args = {"hybrid_layer_pattern": "L-L-*-|L-L-*-"}
    assert build_cmdline_args(args, meta, specs, shell_quote=True) == [
        "--hybrid-layer-pattern",
        '"L-L-*-|L-L-*-"',
    ]
    # Unquoted for argparse/subprocess consumers.
    assert build_cmdline_args(args, meta, specs) == ["--hybrid-layer-pattern", "L-L-*-|L-L-*-"]


def test_value_starting_with_dash_rides_on_its_flag():
    meta, specs = dict(MEGATRON_ARG_METADATA), dict(MEGATRON_ACTION_SPECS)
    # argparse would read a separate "--*-" token as the next option.
    assert build_cmdline_args({"hybrid_layer_pattern": "--*-"}, meta, specs) == [
        "--hybrid-layer-pattern=--*-"
    ]
    assert build_cmdline_args({"hybrid_layer_pattern": "--*-"}, meta, specs, shell_quote=True) == [
        '"--hybrid-layer-pattern=--*-"'
    ]


@needs_bash
@pytest.mark.parametrize("value", ["L-L-*-|L-L-*-", "a b", "--*-", "g[0-9]", "semi;colon"])
def test_values_survive_an_outer_single_quoted_bash_c(value):
    # MareNostrum's launch line wraps the whole command: srun bash -c '<command>'
    inner = f"printf '%s' {shell_token(value)}".replace("'%s'", '%s')
    out = subprocess.run(["bash", "-c", f"bash -c '{inner}'"], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin"}, check=True).stdout
    assert out == value
