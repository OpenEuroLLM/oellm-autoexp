#!/usr/bin/env python3
"""Per-item contamination split for the standard evals, streaming the v6 references.

No numpy / no items.jsonl.gz: seg_stats.npz is read with the stdlib, the reference texts are
streamed once, and only the segments whose normalized text occurs in an eval sample are kept.

    python3 stream_split.py --work /gpfs/scratch/.../flag_decon_v6 \
        --stats /gpfs/scratch/.../contamination_split/v6/seg_stats.npz \
        --samples /gpfs/scratch/.../contamination_split/samples \
        --ref b1consTW_300k --ours main_24000 --extra b1cons_90k oldprod_60k \
        --out /gpfs/scratch/.../contamination_split/v6/split.json
"""
from __future__ import annotations
import argparse, array, ast, gzip, json, os, random, re, statistics, struct, sys, time, zipfile

TASK_SRC = {
    "squadv2": r"^(lighteval|rajpurkar)/squad_v2$", "coqa": r"^EleutherAI/coqa$", "boolq": r"^aps/super_glue$",
    "winogrande": r"^allenai/winogrande$", "wsc273": r"^winograd_wsc$", "belebele": r"^facebook/belebele$",
    "hellaswag": r"^Rowan/hellaswag$", "piqa": r"^baber/piqa$", "social_iqa": r"^allenai/social_i_qa$",
    "openbookqa": r"^allenai/openbookqa$",
}
TASKS = list(TASK_SRC)


def tnorm(s) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(s).lower()).split())


def load_npz_1d(path: str, member: str, typecode: str):
    """Read one 1-D little-endian .npy member of an .npz with the stdlib."""
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        magic = f.read(6)
        assert magic == b"\x93NUMPY", magic
        major = f.read(1)[0]
        f.read(1)  # minor
        if major == 1:
            hl = struct.unpack("<H", f.read(2))[0]
        else:
            hl = struct.unpack("<I", f.read(4))[0]
        hdr = ast.literal_eval(f.read(hl).decode("latin1").strip())
        data = f.read()
    assert hdr["fortran_order"] is False and len(hdr["shape"]) == 1, hdr
    a = array.array(typecode)
    a.frombytes(data)
    if sys.byteorder != "little" and hdr["descr"][0] == "<":
        a.byteswap()
    assert len(a) == hdr["shape"][0], (member, len(a), hdr["shape"])
    return a


def load_stats(path: str):
    print(f"loading {path}", flush=True)
    n = load_npz_1d(path, "n.npy", "b")
    n_tokens = load_npz_1d(path, "n_tokens.npy", "q")
    maxrun = load_npz_1d(path, "maxrun.npy", "i")
    nflag = load_npz_1d(path, "nflag.npy", "q")
    print(f"  {len(n)} segments; {sum(1 for x in n if x > 0)} matchable; "
          f"{sum(1 for x in nflag if x > 0)} flagged", flush=True)
    return n, n_tokens, maxrun, nflag


def load_samples(samples_root: str, models, tasks):
    records = {t: [] for t in tasks}
    texts = {t: set() for t in tasks}
    for model in models:
        for task in tasks:
            path = os.path.join(samples_root, model, f"{task}.jsonl.gz")
            if not os.path.exists(path):
                print(f"  missing {path}", flush=True)
                continue
            k = 0
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    rec = {"model": model, "task": task, "key": r["key"], "score": r.get("score"),
                           "f1": r.get("f1"), "align": r.get("align") or {}}
                    records[task].append(rec)
                    for txt in rec["align"].values():
                        if txt:
                            texts[task].add(txt)
                    k += 1
            print(f"  {model}/{task}: {k} items", flush=True)
    return records, texts


def build_seg_task(work: str, tasks):
    rx = {t: re.compile(TASK_SRC[t]) for t in tasks}
    seg_task = {}
    path = os.path.join(work, "refs", "refs_v6.attrib.items.jsonl.gz")
    print(f"reading {path}", flush=True)
    t0 = time.time()
    n_items = n_sel = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            n_items += 1
            r = json.loads(line)
            for task, x in rx.items():
                if x.match(r["src"]):
                    for seg, _field in r["segs"]:
                        seg_task[seg] = task
                    n_sel += 1
                    break
            if n_items % 1_000_000 == 0:
                print(f"  {n_items} items, {len(seg_task)} segments, {n_sel} selected, "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"  done: {n_items} items, {len(seg_task)} segments, {n_sel} selected, {time.time()-t0:.0f}s",
          flush=True)
    return seg_task


def label(rec, stats):
    found, flagged, cover = [], False, 0.0
    for _field, txt in rec["align"].items():
        s = stats.get((rec["task"], txt))
        if s is None:
            continue
        matchable, ntok, mrun, nfl = s
        if matchable:
            found.append(_field)
            flagged |= nfl > 0
            if ntok:
                cover = max(cover, mrun / ntok)
    if flagged:
        return "seen", cover
    return ("clean" if found else "unknown"), cover


def boot(diffs, n=2000, seed=0):
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    rnd = random.Random(seed)
    m = statistics.fmean(diffs)
    k = len(diffs)
    bs = sorted(statistics.fmean(rnd.choices(diffs, k=k)) for _ in range(n))
    return m, bs[int(0.025*n)], bs[int(0.975*n)-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--ref", default="b1consTW_300k")
    ap.add_argument("--ours", default="main_24000")
    ap.add_argument("--extra", nargs="*", default=["b1cons_90k", "oldprod_60k", "main_23500"])
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--metric", choices=["score", "f1", "both"], default="both")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-attrib", type=int, default=0)
    ap.add_argument("--max-ref", type=int, default=0)
    a = ap.parse_args()

    models = [a.ref, a.ours] + [e for e in a.extra if e not in (a.ref, a.ours)]
    records, eval_texts = load_samples(a.samples, models, a.tasks)

    n, n_tokens, maxrun, nflag = load_stats(a.stats)
    seg_task = build_seg_task(a.work, a.tasks)
    if a.max_attrib:
        # rebuild a truncated mapping for a quick test
        rx = {t: re.compile(TASK_SRC[t]) for t in a.tasks}
        seg_task = {}
        with gzip.open(os.path.join(a.work, "refs", "refs_v6.attrib.items.jsonl.gz"), "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= a.max_attrib:
                    break
                r = json.loads(line)
                for task, x in rx.items():
                    if x.match(r["src"]):
                        for seg, _field in r["segs"]:
                            seg_task[seg] = task
                        break
    print(f"seg_task: {len(seg_task)} segments", flush=True)

    # Stream the texts, but keep the segment ids that hit an eval text.
    hit_ids = {t: {} for t in a.tasks}  # task -> normalized text -> [segment ids]
    path = os.path.join(a.work, "refs", "refs_v6.jsonl.gz")
    print(f"streaming {path}", flush=True)
    t0 = time.time()
    n_hit = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if a.max_ref and i >= a.max_ref:
                break
            task = seg_task.get(i)
            if task is None:
                continue
            want = eval_texts.get(task)
            if not want:
                continue
            txt = json.loads(line)["text"]
            k = tnorm(txt)
            if k in want:
                hit_ids[task].setdefault(k, []).append(i)
                n_hit += 1
            if i and i % 2_000_000 == 0:
                print(f"  {i} lines, {n_hit} eval-text hits, {time.time()-t0:.0f}s", flush=True)
    print(f"  done: {i+1 if 'i' in locals() else 0} lines, {n_hit} hits, {time.time()-t0:.0f}s", flush=True)

    # Resolve the last segment id per (task, text), as items.jsonl.gz did.
    stats = {}
    for task, by_text in hit_ids.items():
        for k, ids in by_text.items():
            seg = ids[-1]
            stats[(task, k)] = (n[seg] > 0, n_tokens[seg], maxrun[seg], nflag[seg])
    print(f"resolved {len(stats)} eval-text reference segments", flush=True)

    # Per task/metric paired analysis.
    out = {}
    for task in a.tasks:
        recs = records[task]
        if not recs:
            continue
        by_model = {}
        for r in recs:
            by_model.setdefault(r["model"], {})[r["key"]] = r
        if a.ours not in by_model or a.ref not in by_model:
            print(f"\n== {task}: missing model samples", flush=True)
            continue
        metrics = []
        if a.metric in ("score", "both"):
            metrics.append("score")
        if a.metric in ("f1", "both") and task in ("coqa", "squadv2"):
            metrics.append("f1")
        out[task] = {}
        for metric in metrics:
            keys = sorted(set(by_model[a.ours]) & set(by_model[a.ref]))
            groups = {"all": [], "seen": [], "clean": [], "unknown": []}
            labels = {}
            for k in keys:
                lab, cov = label(by_model[a.ours][k], stats)
                labels[k] = (lab, cov)
                groups["all"].append(k)
                groups[lab].append(k)
            res = {}
            for g, ks in groups.items():
                if not ks:
                    continue
                row = {"n": len(ks)}
                for m in [a.ref, a.ours] + [e for e in a.extra if e in by_model]:
                    vals = [by_model[m][k][metric] for k in ks if k in by_model[m] and by_model[m][k][metric] is not None]
                    row[m] = 100 * statistics.fmean(vals) if vals else float("nan")
                diffs = [by_model[a.ref][k][metric] - by_model[a.ours][k][metric] for k in ks
                         if k in by_model[a.ref] and k in by_model[a.ours]
                         and by_model[a.ref][k][metric] is not None and by_model[a.ours][k][metric] is not None]
                m, lo, hi = boot(diffs)
                row.update(diff=100*m, lo=100*lo, hi=100*hi)
                for e in a.extra:
                    if e in by_model:
                        ee = [by_model[e][k][metric] - by_model[a.ours][k][metric] for k in ks
                              if k in by_model[e] and k in by_model[a.ours]
                              and by_model[e][k][metric] is not None and by_model[a.ours][k][metric] is not None]
                        if ee:
                            row[f"{e}-ours"] = 100*statistics.fmean(ee)
                res[g] = row
            out[task][metric] = res
            print(f"\n== {task} ({metric}); ref={a.ref} ours={a.ours}; n={len(keys)}", flush=True)
            for g, row in res.items():
                ex = "  ".join(f"{k} {v:+.2f}" for k, v in row.items() if k.endswith("-ours"))
                print(f"  {g:8s} n={row['n']:5d}  ours {row[a.ours]:6.2f}  {a.ref} {row[a.ref]:6.2f}  "
                      f"{a.ref}-ours {row['diff']:+6.2f} [{row['lo']:+.2f},{row['hi']:+.2f}]  {ex}", flush=True)
            # contamination-explained fraction of the overall difference
            allrow = res.get("all"); seen = res.get("seen"); clean = res.get("clean")
            if allrow and seen and clean and allrow["diff"] != 0:
                w = seen["n"] / allrow["n"]
                contrib = w * seen["diff"] / allrow["diff"]
                extra = seen["diff"] - clean["diff"]
                print(f"  -> seen share {w:.1%}; seen-group contribution to the gap {contrib:.1%}; "
                      f"extra gap on seen items {extra:+.2f} pp", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
