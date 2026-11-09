"""Shim: the shared-filesystem row queue is oellm-eval's ``oellm.runner.queue`` (and
``oellm.runner.plan.split_items``). The command line of the old runners keeps working:

    python scripts/korbi/flag_eval/flag_queue.py init --queue <dir> --done <out>/<export>/done \\
        --names vllm/hellaswag/0 vllm/mmlu/5 ... [--split-s 300 --max-parts 8]
    python scripts/korbi/flag_eval/flag_queue.py status --queue <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.korbi.flag_eval import _oellm  # noqa: E402,F401
from oellm.runner.layout import Layout  # noqa: E402
from oellm.runner.plan import split_items  # noqa: E402
from oellm.runner.queue import Queue as _Queue  # noqa: E402
from oellm.runner.queue import lm_item_done  # noqa: E402

ENGINE_TIMES = Path(__file__).resolve().parents[3] / "config/eval_tasks/flag_evals_engine_times.json"


def plan(names, times, split_s, max_parts):
    return [tuple(i) for i in split_items(names, times, split_s, max_parts)]


def is_done(item: str, done_dir: Path) -> bool:
    return lm_item_done(item, Layout(Path(done_dir).parent))


class Queue(_Queue):
    def init(self, items, done_dir=None):
        return super().init(items, None if done_dir is None else (lambda item: is_done(item, done_dir)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("init")
    pi.add_argument("--queue", type=Path, required=True)
    pi.add_argument("--done", type=Path, help="the runner's done dir: done rows are not queued")
    pi.add_argument("--names", nargs="+", required=True)
    pi.add_argument("--times", type=Path, default=ENGINE_TIMES)
    pi.add_argument("--split-s", type=float, default=300.0, help="split rows longer than this (0: never)")
    pi.add_argument("--max-parts", type=int, default=8)
    ps = sub.add_parser("status")
    ps.add_argument("--queue", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "init":
        times = json.loads(a.times.read_text()) if a.times.exists() else {}
        items = plan(a.names, times, a.split_s, a.max_parts)
        q = Queue(a.queue)
        added = q.init(items, a.done)
        print(f"queued {added} of {len(items)} items ({len(a.names)} rows); {q.status()}")
    else:
        print(json.dumps(Queue(a.queue).status()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
