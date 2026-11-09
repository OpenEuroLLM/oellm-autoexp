"""scripts/korbi/flag_eval/flag_queue.py (a shim over oellm-eval's oellm.runner.queue): the old interface of the lm-eval row queue."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

from scripts.korbi.flag_eval.flag_queue import Queue, is_done, plan


def test_plan_splits_long_rows_longest_first():
    items = plan(["vllm/a/0", "vllm/b/5", "vllm/c/0"], {"vllm/a/0": 1000, "vllm/b/5": 100, "vllm/c/0": 50}, 300, 8)
    names = [n for n, _ in items]
    assert names[:4] == [f"vllm/a/0@{i}of4" for i in range(4)]  # 1000 s -> 4 parts of 250 s
    assert names[4:] == ["vllm/b/5", "vllm/c/0"]
    # a row without a time gets the median (here 550 s: split in 2)
    assert [n for n, _ in plan(["vllm/d/0"], {"vllm/a/0": 1000, "vllm/b/5": 100}, 300, 8)] == [
        "vllm/d/0@0of2", "vllm/d/0@1of2"]
    assert plan(["vllm/a/0"], {"vllm/a/0": 10**6}, 300, 8)[0][0] == "vllm/a/0@0of8"  # max parts
    assert plan(["vllm/a/0"], {"vllm/a/0": 10**6}, 0, 8) == [("vllm/a/0", 10**6)]  # never split


def test_init_skips_done_and_known(tmp_path):
    done = tmp_path / "done"
    done.mkdir()
    (done / "a_0.ok").write_text("{}")  # the whole row: its parts are done too
    (done / "b_5@1of2.ok").write_text("{}")
    assert is_done("vllm/a/0@3of4", done) and is_done("vllm/b/5@1of2", done)
    assert not is_done("vllm/b/5@0of2", done)
    q = Queue(tmp_path / "q")
    items = [("vllm/a/0", 9), ("vllm/b/5@0of2", 5), ("vllm/b/5@1of2", 5)]
    assert q.init(items, done) == 1
    assert q.init(items, done) == 0, "idempotent"
    assert q.status() == {"pending": 1, "running": 0, "failed": 0}


def _claim_all(root: str, out) -> None:
    q = Queue(root)
    got = []
    while (c := q.claim()) is not None:
        got.append(c[1]["item"])
        q.complete(c[0])
    out.put(got)


def test_concurrent_workers_claim_each_item_once(tmp_path):
    q = Queue(tmp_path / "q")
    q.init([(f"vllm/t{i}/0", 100 - i) for i in range(60)])
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_claim_all, args=(str(tmp_path / "q"), out)) for _ in range(4)]
    for p in procs:
        p.start()
    claimed = [x for _ in procs for x in out.get(timeout=60)]
    for p in procs:
        p.join(timeout=60)
    assert sorted(claimed) == sorted(f"vllm/t{i}/0" for i in range(60))
    assert q.status() == {"pending": 0, "running": 0, "failed": 0}


def test_longest_first_and_a_dead_workers_claim_returns(tmp_path):
    q = Queue(tmp_path / "q", lease_s=5)
    q.init([("vllm/short/0", 1), ("vllm/long/0", 100)])
    path, data = q.claim()
    assert data["item"] == "vllm/long/0"
    old = time.time() - 10
    os.utime(path, (old, old))  # its worker stopped touching it
    path2, data2 = q.claim()
    assert data2["item"] == "vllm/long/0", "the stale claim is taken back first"
    q.touch(path2)
    path3, data3 = q.claim()
    assert data3["item"] == "vllm/short/0"


def test_fail_retries_then_gives_up(tmp_path):
    q = Queue(tmp_path / "q", max_attempts=2)
    q.init([("vllm/x/0", 1)])
    p, d = q.claim()
    q.fail(p, d)
    assert q.status()["pending"] == 1
    p, d = q.claim()
    assert d["attempts"] == 1
    q.fail(p, d)
    assert q.status() == {"pending": 0, "running": 0, "failed": 1}
    assert json.loads(next((tmp_path / "q" / "failed").iterdir()).read_text())["attempts"] == 2
