"""Stop the ambitious main run when it destabilizes (docs/ambitious_stabilization.md, section 6),
so the run it displaced (the conservative q35, requeued at a lower priority) takes the nodes back.

One check per call (``--once``; a bash loop on the login node runs it every few minutes, so no
process accumulates CPU time against the login node's limit). Reads the run's TensorBoard event
files with a stdlib-only reader (the login node has neither tensorboard nor protobuf) and, from
``--min-step`` on, trips on:
  1. experts: ``moe/routed_expert_output_rms_max`` above ``--expert-max`` at ``--expert-count``
     consecutive readings (pab held ~0.07 in the reproducer; the unbounded runaway was at 0.88 by
     step 500 and 6.7 at 900);
  2. grad norm: the median of the last 50 steps above max(``--gn-floor``, ``--gn-factor`` x the
     median of steps 100-300) (stable runs ~0.2-0.3; QB without pab went to 7.7 / 12.8);
  3. loss: NaN / inf, or the mean of the last 20 steps above the mean of the 100 before by
     ``--loss-jump``;
  4. router (from ``--router-min-step``, the medians of the last ``--router-window`` readings):
     near-dead (layer, expert) slots above ``--near-dead-max``, ghost slots above ``--ghost-max``,
     or the worst layer's load entropy below ``--entropy-min`` (v1 healthy: < 10 near-dead,
     entropy > 0.99; the broken v2 starts: thousands of near-dead / hundreds of ghost slots);
  5. no progress: a job is RUNNING in a pool but the last logged step has not advanced for
     ``--stall-min`` minutes (crash / restart loops; state in <run-dir>/watchdog_state.json);
  6. extra rules (``--rule TAG:OP:LIMIT:MIN_STEP:WINDOW``, repeatable): the median of TAG's last
     WINDOW readings from MIN_STEP on is beyond LIMIT (OP ``>`` or ``<``); a TAG ending in ``*``
     is the max (``>``) / min (``<``) over every tag with that prefix, per step, one ending in
     ``*mean`` their mean. For the LDM run
     the v3 precursors: front-layer expert overload and the first site's write ratio moved
     300-600 steps before the grad norm (docs/hybrid_moe_3b_campaign.md).
On a trip it writes ``<run-dir>/watchdog_tripped.json`` and cancels ``--job`` in every ``--pool``
(unless ``--dry-run``); later calls see the marker and do nothing.

    while true; do python main_run_watchdog.py --once --run-dir <dir> --job ambitious_v3_128n_p1 \\
        [--job ambitious_v3_128n_p2 ...] --pool <pool> [--pool ...]; sleep 600; done
"""

import argparse
import glob
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

TAGS = ("moe/routed_expert_output_rms_max", "grad-norm", "lm loss",
        "moe/selected_near_dead_expert_layer_slots", "moe/ghost_expert_layer_slots",
        "moe/dispatched_expert_load_entropy_min")


def _varint(buf, i):
    out = shift = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7


def _fields(buf):
    """(field number, wire type, value) of a protobuf message; value: int / bytes / raw 4-8 bytes."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i : i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i : i + ln], i + ln
        elif wt == 5:
            v, i = buf[i : i + 4], i + 4
        else:
            return
        yield fn, wt, v


def read_scalars(path, tags, prefixes=()):
    """{tag: {step: value}} of the simple_value scalars in one tfevents file (Event.step = 2,
    Event.summary = 5 -> Summary.value = 1 -> Value.tag = 1, Value.simple_value = 2); ``tags``
    exactly, plus every tag starting with one of ``prefixes``."""
    out = {t: {} for t in tags}
    data = Path(path).read_bytes()
    i = 0
    while i + 12 <= len(data):
        (length,) = struct.unpack("<Q", data[i : i + 8])
        start = i + 12
        if start + length + 4 > len(data):
            break  # a record still being written
        event = data[start : start + length]
        i = start + length + 4
        step, summary = None, None
        for fn, wt, v in _fields(event):
            if fn == 2 and wt == 0:
                step = v
            elif fn == 5 and wt == 2:
                summary = v
        if step is None or summary is None:
            continue
        for fn, wt, v in _fields(summary):
            if fn != 1 or wt != 2:
                continue
            tag, value = None, None
            for f2, w2, v2 in _fields(v):
                if f2 == 1 and w2 == 2:
                    tag = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 5:
                    (value,) = struct.unpack("<f", v2)
            if value is None or tag is None:
                continue
            if tag in out:
                out[tag][step] = value
            elif prefixes and tag.startswith(prefixes):
                out.setdefault(tag, {})[step] = value
    return out


def parse_rule(text):
    tag, op, limit, min_step, window = text.rsplit(":", 4)
    if op not in (">", "<"):
        raise ValueError(f"rule {text!r}: OP must be > or <")
    return tag, op, float(limit), int(min_step), int(window)


def _prefix(tag):
    """(prefix, aggregate) of a prefix rule tag (``...*`` max / min, ``...*mean``), else None."""
    if tag.endswith("*mean"):
        return tag[: -len("*mean")], "mean"
    if tag.endswith("*"):
        return tag[:-1], "extreme"
    return None


def rule_tags(rules):
    """(exact tags, prefixes) the rules read."""
    exact = tuple(r[0] for r in rules if _prefix(r[0]) is None)
    return exact, tuple(_prefix(r[0])[0] for r in rules if _prefix(r[0]) is not None)


def load(run_dir, rules=()):
    exact, prefixes = rule_tags(rules)
    tags = TAGS + tuple(t for t in exact if t not in TAGS)
    merged = {t: {} for t in tags}
    for path in sorted(glob.glob(os.path.join(run_dir, "tb", "events.out.tfevents.*"))):
        for tag, series in read_scalars(path, tags, prefixes).items():
            merged.setdefault(tag, {}).update(series)
    return merged


def rule_series(series, tag, op):
    """{step: value} of a rule's tag; a prefix tag is the per-step max / min (``*``) or mean
    (``*mean``) over its matches."""
    pre = _prefix(tag)
    if pre is None:
        return series.get(tag, {})
    prefix, how = pre
    per_step = {}
    for t, s in series.items():
        if t.startswith(prefix):
            for step, v in s.items():
                per_step.setdefault(step, []).append(v)
    if how == "mean":
        return {step: sum(vs) / len(vs) for step, vs in per_step.items()}
    pick = max if op == ">" else min
    return {step: pick(vs) for step, vs in per_step.items()}


def check_rules(series, rules):
    """None, or the first rule the run is beyond."""
    for tag, op, limit, min_step, window in rules:
        s = rule_series(series, tag, op)
        pts = [v for st, v in sorted(s.items()) if st >= min_step and math.isfinite(v)]
        if len(pts) >= window:
            m = median(pts[-window:])
            if (m > limit) if op == ">" else (m < limit):
                return (f"rule {tag} {op} {limit}: median {m:.4g} over the last {window} readings "
                        f"(step {max(s)})")
    return None


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def check(series, a):
    """None, or the reason the run should stop."""
    exp, gn, loss = (series[t] for t in TAGS[:3])
    last = max(list(gn) + list(loss) + [0])
    if last < a.min_step:
        return None
    for step in sorted(loss)[-50:]:
        if not math.isfinite(loss[step]):
            return f"loss not finite at step {step}"
    pts = [(s, v) for s, v in sorted(exp.items()) if s >= a.min_step]
    run = 0
    for s, v in pts:
        run = run + 1 if v > a.expert_max else 0
        if run >= a.expert_count:
            return f"routed_expert_output_rms_max {v:.3g} > {a.expert_max} at {a.expert_count} readings (step {s})"
    steps = sorted(gn)
    base = [gn[s] for s in steps if 100 <= s <= 300]
    recent = [gn[s] for s in steps[-50:] if s >= a.min_step]
    if len(base) >= 20 and len(recent) >= 20:
        limit = max(a.gn_floor, a.gn_factor * median(base))
        if median(recent) > limit:
            return f"grad-norm median {median(recent):.3g} over the last {len(recent)} steps > {limit:.3g} (step {steps[-1]})"
    ls = sorted(loss)
    if len(ls) >= 120:
        now = sum(loss[s] for s in ls[-20:]) / 20
        before = sum(loss[s] for s in ls[-120:-20]) / 100
        if now > before + a.loss_jump:
            return f"loss mean {now:.4f} over the last 20 steps > {before:.4f} + {a.loss_jump} (step {ls[-1]})"
    router = (("moe/selected_near_dead_expert_layer_slots", "near-dead slots", a.near_dead_max, 1),
              ("moe/ghost_expert_layer_slots", "ghost slots", a.ghost_max, 1),
              ("moe/dispatched_expert_load_entropy_min", "load entropy min", a.entropy_min, -1))
    for tag, label, limit, sign in router:
        pts = [v for st, v in sorted(series[tag].items()) if st >= a.router_min_step]
        if len(pts) >= a.router_window:
            m = median(pts[-a.router_window:])
            if sign * (m - limit) > 0:
                return f"{label}: median {m:.4g} over the last {a.router_window} readings beyond {limit} (step {last})"
    return None


def _is_job(name, job):
    """``name`` is ``job`` or ``job`` + a sweep-index suffix (autoexp names list-sweep points
    ``<name>_<index>``: ambitious_v3_128n_p1_0, not ambitious_v3_128n_p1)."""
    return name == job or (name.startswith(job + "_") and name[len(job) + 1:].isdigit())


# the sweep-index suffixes a job name may carry (see _is_job); pool_ctl matches names exactly
_SUFFIXES = ("",) + tuple(f"_{i}" for i in range(8))


def running_jobs(a):
    """Names of a.job RUNNING in any of a.pool (pool_ctl status)."""
    found = set()
    for pool in a.pool:
        r = subprocess.run(a.pool_ctl.split() + [pool, "status"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "RUNNING" and any(_is_job(parts[2], j) for j in a.job):
                found.add(parts[2])
    return found


def check_stall(series, a, now):
    """None, or a reason: a job runs but the last step has not moved for --stall-min minutes."""
    if a.stall_min <= 0:
        return None
    state_path = Path(a.run_dir) / "watchdog_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    last = max([max(s) for s in series.values() if s] + [-1])
    running = running_jobs(a)
    if not running or last != state.get("last_step"):
        state = {"last_step": last, "since": now}  # waiting in a queue, or progress: reset
        state_path.write_text(json.dumps(state))
        return None
    if now - state["since"] > 60 * a.stall_min:
        return f"no progress for {(now - state['since']) / 60:.0f} min at step {last} while {sorted(running)} RUNNING"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the run's base dir (with tb/)")
    ap.add_argument("--job", action="append", required=True,
                    help="job name to cancel (pool_ctl cancel --all); repeat for several (curriculum phases)")
    ap.add_argument("--pool", action="append", default=[], help="pool dirs to cancel it in")
    ap.add_argument("--pool-ctl", default="scripts/korbi/mn5_autoexp.sh scripts/pool_ctl.py")
    ap.add_argument("--min-step", type=int, default=200)
    ap.add_argument("--expert-max", type=float, default=0.5)
    ap.add_argument("--expert-count", type=int, default=2)
    ap.add_argument("--gn-floor", type=float, default=1.0)
    ap.add_argument("--gn-factor", type=float, default=4.0)
    ap.add_argument("--loss-jump", type=float, default=0.2)
    ap.add_argument("--router-min-step", type=int, default=300)
    ap.add_argument("--router-window", type=int, default=50)
    ap.add_argument("--near-dead-max", type=float, default=500)
    ap.add_argument("--ghost-max", type=float, default=300)
    ap.add_argument("--entropy-min", type=float, default=0.85)
    ap.add_argument("--stall-min", type=float, default=0, help="minutes without a new step while RUNNING (0: off)")
    ap.add_argument("--rule", action="append", default=[], type=parse_rule,
                    help="TAG:OP:LIMIT:MIN_STEP:WINDOW (OP > or <; TAG* = max / min over a prefix); repeatable")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true", help="one check (the default; kept for the loop)")
    a = ap.parse_args()

    marker = Path(a.run_dir) / "watchdog_tripped.json"
    if marker.exists():
        return 0
    series = load(a.run_dir, a.rule)
    reason = check(series, a) or check_rules(series, a.rule) or check_stall(series, a, time.time())
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    last = {t: (max(s), s[max(s)]) if s else None for t, s in series.items() if t in TAGS}
    for tag, op, *_ in a.rule:
        s = rule_series(series, tag, op)
        last[tag] = (max(s), s[max(s)]) if s else None
    if reason is None:
        print(f"[{stamp}] ok: {json.dumps(last)}", flush=True)
        return 0
    print(f"[{stamp}] TRIPPED: {reason}", flush=True)
    report = {"time": stamp, "reason": reason, "last": last, "job": a.job, "pools": a.pool,
              "dry_run": a.dry_run, "cancel": []}
    if not a.dry_run:
        for pool in a.pool:
            for job in a.job:
                # every sweep-index variant: the 2026-11-01 trip cancelled only ..._p2 and missed
                # the running ..._p1_0 (it trained on, unstable, for 2.3 h)
                for name in (job + sfx for sfx in _SUFFIXES):
                    cmd = a.pool_ctl.split() + [pool, "cancel", name, "--all"]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode == 0:
                        report["cancel"].append({"pool": pool, "job": name, "rc": r.returncode,
                                                 "out": r.stdout[-500:] + r.stderr[-500:]})
                        print(f"  cancel {name} in {pool}: rc 0", flush=True)
        marker.write_text(json.dumps(report, indent=1))
    else:
        print(json.dumps(report, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
