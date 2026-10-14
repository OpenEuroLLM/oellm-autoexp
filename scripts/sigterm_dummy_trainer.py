#!/usr/bin/env python3
"""Dummy trainer for the segment-end test of the monitor policy (1 node, no GPU, stdlib only).

Mimics what matters of Megatron's behaviour at the end of a SLURM segment:
  * the log lines the monitor policy reads (`iteration N/M | ...`, `saving checkpoint at iteration N`,
    `successfully saved checkpoint from iteration N`, `[after training is done]`);
  * checkpoint bookkeeping: `<ckpt-dir>/iter_<N>/state.json` plus `latest_checkpointed_iteration.txt`,
    updated only after a complete save, and loading from the tracker on start;
  * the semantics of megatron/training/dist_signal_handler.py: SIGTERM only sets a flag, the flag is
    checked at the iteration boundary, then a checkpoint is saved, `exiting program after receiving
    SIGTERM.` is printed and the process exits 0 (training.py:3642).

Run under SLURM with `--signal=TERM@<margin>` and a wall time shorter than the schedule, so that every
segment ends by SIGTERM; the monitor's `sigterm_rollover` rule must resubmit, and the next segment must
continue from the signal checkpoint. With `--iters` reached the run prints `[after training is done]`.

  sigterm_dummy_trainer.py --ckpt-dir DIR --iters 600 --tick 1 --save-every 30 --save-seconds 3
"""
import argparse
import datetime
import json
import os
import signal
import sys
import time

_received = False


def _handler(signum, frame):  # noqa: ARG001
    global _received
    _received = True


def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def say(msg):
    print(f"[default0]: [{ts()}] {msg}", flush=True)


def save(ckpt_dir, it, seconds, tracker=True):
    say(f"saving checkpoint at iteration {it:8d} to {ckpt_dir}")
    d = os.path.join(ckpt_dir, f"iter_{it:07d}")
    os.makedirs(d, exist_ok=True)
    time.sleep(seconds)  # simulated write time
    with open(os.path.join(d, "state.json"), "w") as fh:
        json.dump({"iteration": it, "saved_at": ts(), "job": os.environ.get("SLURM_JOB_ID")}, fh)
    if tracker:
        with open(os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt"), "w") as fh:
            fh.write(str(it))
    say(f"successfully saved checkpoint from iteration {it:8d} to {ckpt_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--tick", type=float, default=1.0, help="seconds per iteration")
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--save-seconds", type=float, default=3.0)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _handler)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    start = 0
    tracker = os.path.join(a.ckpt_dir, "latest_checkpointed_iteration.txt")
    if os.path.exists(tracker):
        with open(tracker) as fh:
            start = int(fh.read().strip())
        say(f"loading checkpoint from {a.ckpt_dir} at iteration {start}")
        with open(os.path.join(a.ckpt_dir, f"iter_{start:07d}", "state.json")) as fh:
            st = json.load(fh)
        say(f"successfully loaded checkpoint from {a.ckpt_dir} [ t 0, p 0 ] at iteration {start} (saved by job {st.get('job')} at {st.get('saved_at')})")
    else:
        say(f"no checkpoint under {a.ckpt_dir}: starting at iteration 0")
    say(f"job {os.environ.get('SLURM_JOB_ID')} on {os.uname().nodename}: iterations {start + 1}..{a.iters}, {a.tick}s each, persistent save every {a.save_every}")
    it = start
    while it < a.iters:
        it += 1
        time.sleep(a.tick)
        say(f"iteration {it:8d}/{a.iters:8d} | consumed samples: {it * 4096:12d} | elapsed time per iteration (ms): {a.tick * 1000:.1f} | lm loss: 1.500000E+00 | grad norm: 0.500")
        if _received:
            say("SIGTERM received at the iteration boundary: saving before exit")
            save(a.ckpt_dir, it, a.save_seconds)
            say("exiting program after receiving SIGTERM.")
            sys.exit(0)
        if it % a.save_every == 0:
            save(a.ckpt_dir, it, a.save_seconds)
    say("[after training is done] schedule complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
