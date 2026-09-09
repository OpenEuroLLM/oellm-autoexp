#!/usr/bin/env python3
"""Ingest oellm-eval results into an eval-browser-compatible sqlite, and diff
against another.

jitsev1's eval workspace keeps every result in
``/e/fscratch/e-sta-openeurollm/jj1/eval/eval_browser/eval_browser.sqlite``. This builds a
database with the SAME `aggregates` schema from our own eval output, so our runs can be read
with the same tooling and joined against theirs task-for-task.

    # build our DB from an oellm-eval output tree
    python3 scripts/korbi/eval_results_db.py ingest \\
        --root /e/fscratch/e-sta-openeurollm/poeppel1/eval/softcap_downstream/results \\
        --db   /e/fscratch/e-sta-openeurollm/poeppel1/eval/softcap_downstream/eval_results.sqlite

    # compare one of our exports against one of theirs, metric for metric
    python3 scripts/korbi/eval_results_db.py compare \\
        --db /e/.../eval_results.sqlite --ours cont9_68000 \\
        --ref-db /e/fscratch/e-sta-openeurollm/jj1/eval/eval_browser/eval_browser.sqlite \\
        --theirs production_68000

INPUT LAYOUT, as lm-eval and oellm-eval write it and as their `sources` table records it::

    <root>/<export>/<task_group>/<run_ts>/results/<hash>_<timestamp>.json

with a `results` mapping of task -> {"acc,none": 0.80, "acc_stderr,none": 0.013, ...}, i.e.
metric keys of the form ``<metric>,<filter>``, plus a top-level ``n-shot`` mapping.

ON `is_primary`: their database contains ONLY primary rows -- one designated metric per task
(acc_norm for belebele, acc for global-mmlu-eu, mixed for open-sci-0.01) -- and no
is_primary=0 rows at all. We cannot reconstruct that per-task choice from the JSON, so this
ingests EVERY metric and flags a primary by the documented heuristic below. That is
deliberate: `compare` joins on (task_group, task, metric, filter), so the comparison never depends on
whether our primary rule matches theirs. task_group is part of that key on purpose: the same
task can appear in two groups with different values -- hellaswag scores 0.58215 under
open-sci-0.01 and 0.56772 under dclm-core-rest for the same export.
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS aggregates (
  id INTEGER PRIMARY KEY, export TEXT NOT NULL, arm TEXT, iteration INTEGER,
  root TEXT NOT NULL, task_group TEXT, run_ts TEXT, task TEXT NOT NULL, parent TEXT,
  backend TEXT, metric TEXT NOT NULL, filter TEXT, value REAL, stderr REAL, n INTEGER,
  n_shot INTEGER, repeats INTEGER, is_primary INTEGER, date_id TEXT, source_file TEXT);
CREATE INDEX IF NOT EXISTS ix_agg_task ON aggregates (task, export, metric);
CREATE INDEX IF NOT EXISTS ix_agg_src  ON aggregates (root, source_file);
CREATE TABLE IF NOT EXISTS roots (label TEXT PRIMARY KEY, path TEXT, scanned_at TEXT);
CREATE TABLE IF NOT EXISTS sources (
  root TEXT NOT NULL, source_file TEXT NOT NULL, kind TEXT, mtime REAL, size INTEGER,
  n_rows INTEGER, note TEXT, ingested_at TEXT, PRIMARY KEY (root, source_file));
CREATE VIEW IF NOT EXISTS primary_aggregates_v AS SELECT * FROM aggregates WHERE is_primary = 1;
CREATE VIEW IF NOT EXISTS exports_v AS
  SELECT export, arm, iteration, COUNT(*) AS n_rows FROM aggregates GROUP BY export, arm, iteration;
"""

# Order of preference when flagging one metric per task as primary. Matches what their
# database actually contains for the groups we run: acc_norm for belebele, acc for
# global-mmlu-eu. Only ever a labelling convenience -- `compare` does not rely on it.
PRIMARY_PREFERENCE = [
    ("acc_norm", "none"),
    ("acc", "none"),
    ("exact_match", "strict-match"),
    ("pass_at_1", "none"),
    ("accuracy_avg", ""),
]

# Everything a stderr column, not a metric of its own.
_STDERR = re.compile(r"_stderr$")
_SKIP_KEYS = {"name", "alias", "sample_len", "samples"}


def split_export(export: str):
    """`cont9_68000` -> ("cont9", 68000).

    Trailing digits are the iteration.
    """
    m = re.match(r"^(.*?)_(\d+)$", export)
    return (m.group(1), int(m.group(2))) if m else (export, None)


def parse_results_json(path: Path):
    """Yield (task, metric, filter, value, stderr, n, n_shot) from one lm-eval
    results file."""
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:  # a partial write mid-run should not abort the whole ingest
        print(f"  ! skipping unreadable {path.name}: {exc}", file=sys.stderr)
        return
    results = doc.get("results") or {}
    n_shot_map = doc.get("n-shot") or {}
    for task, payload in results.items():
        if not isinstance(payload, dict):
            continue
        n = payload.get("sample_len")
        n_shot = n_shot_map.get(task)
        stderrs, values = {}, {}
        for key, val in payload.items():
            if key in _SKIP_KEYS or not isinstance(val, (int, float)):
                continue
            metric, _, filt = key.partition(",")
            if _STDERR.search(metric):
                stderrs[(_STDERR.sub("", metric), filt)] = val
            else:
                values[(metric, filt)] = val
        for (metric, filt), value in values.items():
            yield task, metric, filt, value, stderrs.get((metric, filt)), n, n_shot


def pick_primary(rows):
    """Flag one metric per (export, task) as primary, by PRIMARY_PREFERENCE."""
    by_task = {}
    for i, r in enumerate(rows):
        by_task.setdefault((r["export"], r["task"]), []).append(i)
    for idxs in by_task.values():
        chosen = None
        for pref in PRIMARY_PREFERENCE:
            for i in idxs:
                if (rows[i]["metric"], rows[i]["filter"]) == pref:
                    chosen = i
                    break
            if chosen is not None:
                break
        if chosen is None:
            chosen = idxs[0]
        rows[chosen]["is_primary"] = 1


def cmd_ingest(args) -> int:
    root = Path(args.root)
    if not root.is_dir():
        print(f"FATAL: no such results root: {root}", file=sys.stderr)
        return 1

    files = sorted(root.glob("*/*/*/results/*.json"))
    files = [f for f in files if not f.name.startswith("samples_")]
    if not files:
        print(f"FATAL: no results JSON under {root}/<export>/<task_group>/<run_ts>/results/")
        return 1

    rows, sources = [], []
    for path in files:
        run_ts = path.parent.parent.name
        task_group = path.parent.parent.parent.name
        export = path.parent.parent.parent.parent.name
        arm, iteration = split_export(export)
        rel = str(path.relative_to(root))
        n_before = len(rows)
        for task, metric, filt, value, stderr, n, n_shot in parse_results_json(path):
            rows.append(
                dict(
                    export=export,
                    arm=arm,
                    iteration=iteration,
                    root=args.label,
                    task_group=task_group,
                    run_ts=run_ts,
                    task=task,
                    parent=None,
                    backend="lm_eval",
                    metric=metric,
                    filter=filt,
                    value=value,
                    stderr=stderr,
                    n=n,
                    n_shot=n_shot,
                    repeats=None,
                    is_primary=0,
                    date_id=run_ts,
                    source_file=rel,
                )
            )
        st = path.stat()
        sources.append(
            (
                args.label,
                rel,
                "lm_aggregate",
                st.st_mtime,
                st.st_size,
                len(rows) - n_before,
                None,
                datetime.now(timezone.utc).isoformat(),
            )
        )

    pick_primary(rows)

    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    if args.replace:
        con.execute("DELETE FROM aggregates WHERE root = ?", (args.label,))
        con.execute("DELETE FROM sources WHERE root = ?", (args.label,))
    cols = list(rows[0].keys())
    con.executemany(
        f"INSERT INTO aggregates ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    con.executemany("INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?)", sources)
    con.execute(
        "INSERT OR REPLACE INTO roots VALUES (?,?,?)",
        (args.label, str(root), datetime.now(timezone.utc).isoformat()),
    )
    con.commit()

    print(f"[eval-db] {db}")
    print(f"[eval-db]   {len(files)} result files -> {len(rows)} rows")
    for export, arm, iteration, n in con.execute(
        "SELECT export, arm, iteration, COUNT(*) FROM aggregates GROUP BY export ORDER BY export"
    ):
        print(f"[eval-db]   {export:<26} arm={arm:<12} iter={iteration}  rows={n}")
    con.close()
    return 0


def cmd_compare(args) -> int:
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.execute("ATTACH DATABASE ? AS ref", (f"file:{args.ref_db}?mode=ro&immutable=1",))
    # Join on task+metric+filter, so a differing is_primary convention cannot affect this.
    sql = """
      SELECT a.task_group,
             COUNT(*)                                   AS tasks,
             ROUND(AVG(b.value), 4)                     AS theirs,
             ROUND(AVG(a.value), 4)                     AS ours,
             ROUND(AVG(a.value) - AVG(b.value), 4)      AS delta
      FROM aggregates a
      JOIN ref.aggregates b
        ON a.task_group = b.task_group AND a.task = b.task AND a.metric = b.metric
       AND IFNULL(a.filter,'') = IFNULL(b.filter,'')
      WHERE a.export = ? AND b.export = ?
      GROUP BY a.task_group ORDER BY tasks DESC
    """
    rows = list(con.execute(sql, (args.ours, args.theirs)))
    if not rows:
        print(f"no overlapping (task, metric, filter) between {args.ours} and {args.theirs}")
        return 1
    print(f"{args.ours}  vs  {args.theirs}")
    print(f"{'task_group':<22}{'tasks':>7}{'theirs':>10}{'ours':>10}{'delta':>10}")
    for tg, n, theirs, ours, delta in rows:
        print(f"{tg or '?':<22}{n:>7}{theirs:>10}{ours:>10}{delta:>+10}")
    con.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="build a DB from an oellm-eval results tree")
    ing.add_argument("--root", required=True, help="directory holding <export>/<task_group>/...")
    ing.add_argument("--db", required=True, help="sqlite file to create or extend")
    ing.add_argument("--label", default="poeppel1", help="value for the `root` column")
    ing.add_argument("--replace", action="store_true", help="drop this label's rows first")
    ing.set_defaults(func=cmd_ingest)

    cmp_ = sub.add_parser("compare", help="diff one export against one in another DB")
    cmp_.add_argument("--db", required=True)
    cmp_.add_argument("--ours", required=True, help="export name in --db")
    cmp_.add_argument("--ref-db", required=True)
    cmp_.add_argument("--theirs", required=True, help="export name in --ref-db")
    cmp_.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
