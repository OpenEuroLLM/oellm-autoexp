"""Tests for the segment-end rules added on 2026-09-06 to
``config/job/auto_restart_ckptreset_faultscan.yaml``: ``operator_cancel``,
``sigterm_rollover`` and ``time_limit_rollover``.

A segment that receives SLURM's ``--signal=TERM@<margin>`` saves a checkpoint and
prints ``exiting program after receiving SIGTERM.`` (training.py:3642). The same
line follows a deliberate ``scancel`` (SLURM sends SIGTERM first), which SLURM
announces in the log as ``*** JOB <id> ON <node> CANCELLED AT <ts> ***``; the
wall-clock limit adds ``DUE TO TIME LIMIT``. The monitor evaluates the rules in
list order and the first terminal effect wins, so the tests check the patterns
against real log lines and the order of the rules in the policy file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from oellm_autoexp.monitor.actions import LogEvent, LogEventConfig, RestartActionConfig

POLICY = (
    Path(__file__).resolve().parents[2] / "config" / "job" / "auto_restart_ckptreset_faultscan.yaml"
)

# Verbatim shapes from the production logs (slurm-1486215.log etc.) and Megatron. NB the delivery of
# `--signal=TERM@N` is itself logged as a STEP line (job 1691075, 2026-09-06), never as a JOB line.
SIGNAL_STEP_LINE = (
    "slurmstepd: error: *** STEP 1691075.0 ON jpbo-008-04 CANCELLED AT 2026-09-06T15:39:18 ***"
)
SCANCEL_JOB_LINE = "srun: forcing job termination\nslurmstepd: error: *** JOB 1486215 ON jpbo-002-01 CANCELLED AT 2026-08-25T08:21:49 ***"
SCANCEL_STEP_LINE = (
    "slurmstepd: error: *** STEP 1486215.0 ON jpbo-002-01 CANCELLED AT 2026-08-25T08:21:49 ***"
)
TIME_LIMIT_LINE = "slurmstepd: error: *** JOB 1537344 ON jpbo-002-01 CANCELLED AT 2026-08-30T22:59:07 DUE TO TIME LIMIT ***"
SIGTERM_EXIT_LINE = (
    "[default0]:[exiting program after receiving SIGTERM.] datetime: 2026-09-06 17:57:10 "
)
DURATION_EXIT_LINE = (
    "[default0]:[exiting program after 690.0159624854724 minutes] datetime: 2026-08-25 19:54:10 "
)


def load_rules():
    cfg = yaml.safe_load(POLICY.read_text())
    return [
        e
        for e in cfg["log_events"]
        if e.get("pattern") is not None or e.get("pattern_type") in ("inactivity", "progress")
    ]


def rule(name):
    return next(e for e in load_rules() if e["name"] == name)


def first_terminal(text):
    """Name of the first rule (in policy order) whose pattern matches ``text``
    and whose action is terminal (Cancel/Finish/Restart), i.e. what
    _process_log_events returns on this log excerpt."""
    for e in load_rules():
        if e.get("pattern_type") in ("inactivity", "progress"):
            continue
        cls = (e.get("action") or {}).get("class_name")
        if cls not in ("CancelAction", "FinishAction", "RestartAction"):
            continue
        cfg = LogEventConfig(
            name=e["name"],
            pattern=e["pattern"],
            pattern_type=e.get("pattern_type", "substring"),
            action=RestartActionConfig(reason="x"),
        )
        if LogEvent(cfg).check_triggers(text):
            return e["name"]
    return None


def test_rules_present_in_the_right_order():
    names = [e["name"] for e in load_rules()]
    i = {
        n: names.index(n)
        for n in (
            "operator_cancel",
            "sigterm_rollover",
            "time_limit_rollover",
            "exit_duration_rollover",
        )
    }
    assert (
        i["operator_cancel"]
        < i["sigterm_rollover"]
        < i["time_limit_rollover"]
        < i["exit_duration_rollover"]
    )
    assert rule("operator_cancel")["action"]["class_name"] == "CancelAction"
    assert rule("sigterm_rollover")["action"]["class_name"] == "RestartAction"
    assert rule("time_limit_rollover")["action"]["class_name"] == "RestartAction"
    assert rule("sigterm_rollover")["condition"]["max_fires"] >= 100


def test_operator_cancel_pattern():
    pat = rule("operator_cancel")["pattern"]
    assert re.search(pat, SCANCEL_JOB_LINE)
    assert re.search(pat, TIME_LIMIT_LINE) is None, (
        "the wall-limit line must not read as an operator cancel"
    )
    assert re.search(pat, SCANCEL_STEP_LINE) is None, (
        "only the JOB line is needed; STEP lines follow it"
    )


def test_time_limit_pattern():
    pat = rule("time_limit_rollover")["pattern"]
    assert re.search(pat, TIME_LIMIT_LINE)
    assert re.search(pat, SCANCEL_JOB_LINE) is None


def test_sigterm_pattern_is_exact():
    pat = rule("sigterm_rollover")["pattern"]
    assert pat in SIGTERM_EXIT_LINE
    assert pat not in DURATION_EXIT_LINE


@pytest.mark.parametrize(
    "text, expected",
    [
        # the wall-clock signal: SLURM's STEP line (its delivery) and Megatron's line -> rollover
        (SIGTERM_EXIT_LINE, "sigterm_rollover"),
        (SIGNAL_STEP_LINE + "\n" + SIGTERM_EXIT_LINE, "sigterm_rollover"),
        # scancel: SLURM's line and Megatron's line in the same poll, either order -> stays cancelled
        (SCANCEL_JOB_LINE + "\n" + SIGTERM_EXIT_LINE, "operator_cancel"),
        (SIGTERM_EXIT_LINE + "\n" + SCANCEL_JOB_LINE + "\n" + SCANCEL_STEP_LINE, "operator_cancel"),
        # the wall limit without an exit checkpoint -> rollover from the last complete checkpoint
        (TIME_LIMIT_LINE, "time_limit_rollover"),
        # the timed exit is untouched
        (DURATION_EXIT_LINE, "exit_duration_rollover"),
    ],
)
def test_first_terminal_effect(text, expected):
    assert first_terminal(text) == expected
