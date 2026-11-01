"""Export the v8 + multilingual data (datamix_v8ml + stage files + fixed validation sets) from MN5 to JSC
(/e/data1/datasets/playground/mmlaion/oellm_testing) without duplicating what JSC already holds.
Stdlib only (MN5 / JUPITER login nodes). Modes:

  manifest (MN5)  one line per inode: inode, class, size, origin path, all names (relative to the MN5 data root).
                  Classes by the inode's origin tree: A = flag collection, B = mixture-vitae subsets (both copies of
                  JSC oellm/ files), C = GigaToken output on JUPITER scratch, D = LUMI vtok, E = new on MN5.
  hash   (both)   SHA-256 of the full .idx and of 64 x 4 MiB ranges of each .bin, classes A / B / C, on the
                  MN5 file or its JSC counterpart ("inode<TAB>hash"); --part i/n for parallel processes.
  stage  (MN5)    the files to send: classes D / E, one name per inode. Large .bin -> "SRC DST" pairs for
                  parallel_transfer.sh; the rest hard-linked into <stage>/b<k>/ buckets (one session each), plus the
                  tokenizer, the validation sets' info / selection files and the datamix files with JSC paths.
  link   (JSC)    A / B: a relative symlink per name into ../oellm (only where size and hash matched: --hashes-ok);
                  C: copied from JUPITER scratch; C / D / E: further names hard-linked to the first. Nothing is
                  written outside the target; existing entries are never replaced.
  verify (JSC)    every name exists with the manifest size (symlinks resolved).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

MN5_DATA = "/gpfs/projects/ehpc533/users/korbip/data"
MN5_MIX = "/gpfs/projects/ehpc533/users/korbip/hybrid_v019/data"
MIXES = ["datamix_v8ml_mn5.txt", "datamix_v8ml_stage1_mn5.txt", "datamix_v8ml_stage2_mn5.txt",
         "datamix_v8ml_stage3_mn5.txt"]
JSC_T = "/e/data1/datasets/playground/mmlaion/oellm_testing"
MN5_ORIGIN = [("A", "/gpfs/projects/ehpc533/training-data-collection/flag"),
              ("B", MN5_DATA + "/v8_sources/jsc_mv1_subsets"),
              ("C", MN5_DATA + "/v8_sources/jsc_tok"),
              ("D", MN5_DATA + "/v8_sources/lumi_vtok")]
JSC_ORIGIN = {"A": "/e/data1/datasets/playground/mmlaion/oellm/training/collection/flag",
              "B": "/e/data1/datasets/playground/mmlaion/oellm/container/data/training/collection/flag/"
                   "mixture-vitae-1.0-subsets-reasoning-instruct",
              "C": "/e/scratch/e-ext-2025e02-108/poeppel1/v8/tok"}
BIG = 1 << 30                                  # .bin files above this go as their own transfer session


def read_manifest(path):
    rows = []
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        rows.append({"ino": f[0], "cls": f[1], "size": int(f[2]), "origin": f[3], "names": f[4:]})
    return rows


def needed_prefixes():
    out = set()
    for m in MIXES:
        for line in open(os.path.join(MN5_MIX, m)):
            if line.strip() and not line.startswith("#"):
                out.add(line.split()[1])
    for line in open(os.path.join(MN5_DATA, "v8ml-valid", "valid_sets.txt")):
        out.add(line.split()[2])
    return out


def cmd_manifest(a):
    need = {}
    for p in needed_prefixes():
        for ext in (".bin", ".idx"):
            st = os.stat(p + ext)
            need.setdefault((st.st_dev, st.st_ino), []).append(p + ext)
    origin = {}
    for cls, root in MN5_ORIGIN:
        for dp, _, fn in os.walk(root):
            for f in fn:
                if f.endswith((".bin", ".idx")):
                    q = os.path.join(dp, f)
                    try:
                        st = os.lstat(q)
                    except OSError:
                        continue
                    k = (st.st_dev, st.st_ino)
                    if k in need and k not in origin:
                        origin[k] = (cls, os.path.relpath(q, root))
    with open(a.out, "w") as fo:
        for k, ps in sorted(need.items(), key=lambda kv: sorted(kv[1])[0]):
            cls, rel = origin.get(k, ("E", ""))
            names = sorted(os.path.relpath(p, MN5_DATA) for p in ps)
            fo.write("\t".join([str(k[1]), cls, str(os.stat(ps[0]).st_size), rel] + names) + "\n")
    print(f"{len(need)} inodes -> {a.out}")


def file_hash(path, size, full):
    h, ch = hashlib.sha256(), 4 << 20
    with open(path, "rb") as fi:
        if full:
            for b in iter(lambda: fi.read(16 << 20), b""):
                h.update(b)
        else:
            for off in sorted({min(i * size // 64, max(0, size - ch)) for i in range(64)} | {max(0, size - ch)}):
                fi.seek(off)
                h.update(fi.read(ch))
    return h.hexdigest()


def cmd_hash(a):
    i, n = (int(x) for x in a.part.split("/"))
    rows = [r for r in read_manifest(a.manifest) if r["cls"] in "ABC"]
    for j, r in enumerate(rows):
        if j % n != i:
            continue
        p = (os.path.join(MN5_DATA, r["names"][0]) if a.side == "mn5"
             else os.path.join(JSC_ORIGIN[r["cls"]], r["origin"]))
        print(r["ino"], file_hash(p, r["size"], p.endswith(".idx")), sep="\t", flush=True)


def jsc_path(p):
    return p.replace(MN5_DATA, JSC_T)


def cmd_stage(a):
    rows = [r for r in read_manifest(a.manifest) if r["cls"] in "DE"]
    pairs, small = [], []
    for r in rows:
        first = r["names"][0]
        if first.endswith(".bin") and r["size"] > BIG:
            pairs.append(f"marenostrum_new:{MN5_DATA}/{first} jupiter:{JSC_T}/{first}")
        else:
            small.append((r["size"], first))
    small.sort(reverse=True)
    load = [0] * a.buckets
    for size, first in small:                   # largest first into the emptiest bucket
        k = load.index(min(load))
        load[k] += size
        dst = os.path.join(a.stage, f"b{k}", first)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            os.link(os.path.join(MN5_DATA, first), dst)
    extra = os.path.join(a.stage, "b0")
    for sub in ("tokenizers/tokenizer-256k-agent",):
        for dp, _, fn in os.walk(os.path.join(MN5_DATA, sub)):
            for f in fn:
                src = os.path.join(dp, f)
                dst = os.path.join(extra, os.path.relpath(src, MN5_DATA))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    os.link(src, dst)
    vdir = os.path.join(MN5_DATA, "v8ml-valid")
    for f in os.listdir(vdir):
        if f.endswith((".info.json", ".selection.npz")):
            dst = os.path.join(extra, "v8ml-valid", f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                os.link(os.path.join(vdir, f), dst)
    rewrite = [(os.path.join(vdir, "valid_sets.txt"), "v8ml-valid/valid_sets.txt")]
    rewrite += [(os.path.join(MN5_MIX, m), "datamix/" + m.replace("_mn5", "_jsc")) for m in MIXES]
    rewrite += [(os.path.join(MN5_MIX, "datamix_v8ml_mn5_report.tsv"), "datamix/datamix_v8ml_jsc_report.tsv")]
    for src, rel in rewrite:
        dst = os.path.join(extra, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src) as fi, open(dst, "w") as fo:
            fo.write(jsc_path(fi.read()))
    with open(a.pairs, "w") as fo:
        fo.write("\n".join(pairs) + "\n")
        for k in range(a.buckets):
            fo.write(f"marenostrum_new:{a.stage}/b{k}/ jupiter:{JSC_T}/\n")
    print(f"{len(pairs)} large files ({sum(r['size'] for r in rows if r['names'][0].endswith('.bin') and r['size'] > BIG) / 1e12:.2f} TB), "
          f"{len(small)} small files in {a.buckets} buckets ({sum(s for s, _ in small) / 1e12:.3f} TB) -> {a.pairs}")


def under_target(p):
    assert os.path.abspath(p).startswith(JSC_T + "/"), p
    return p


def cmd_link(a):
    rows = read_manifest(a.manifest)
    ok = set()
    if a.hashes_ok:
        ok = {line.split()[0] for line in open(a.hashes_ok) if line.strip()}
    copies, n_sym, n_hl, skipped = [], 0, 0, []
    for r in rows:
        if r["cls"] in "AB":
            tgt = os.path.join(JSC_ORIGIN[r["cls"]], r["origin"])
            if r["ino"] not in ok or os.stat(tgt).st_size != r["size"]:
                skipped.append(r["names"][0])
                continue
            for name in r["names"]:
                link = under_target(os.path.join(JSC_T, name))
                os.makedirs(os.path.dirname(link), exist_ok=True)
                rel = os.path.relpath(tgt, os.path.dirname(link))
                if os.path.lexists(link):
                    assert os.path.islink(link) and os.readlink(link) == rel, f"exists, differs: {link}"
                    continue
                os.symlink(rel, link)
                n_sym += 1
        elif r["cls"] == "C":
            copies.append(r)
    def copy(r):
        dst = under_target(os.path.join(JSC_T, r["names"][0]))
        if os.path.exists(dst) and os.stat(dst).st_size == r["size"]:
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["cp", "--preserve=timestamps", os.path.join(JSC_ORIGIN["C"], r["origin"]), dst + ".part"],
                       check=True)
        os.replace(dst + ".part", dst)
        print(f"copied {r['names'][0]} {r['size'] / 1e9:.1f} GB", flush=True)
    with ThreadPoolExecutor(a.copy_jobs) as ex:
        list(ex.map(copy, copies))
    missing = []
    for r in rows:
        if r["cls"] in "AB":
            continue
        first = under_target(os.path.join(JSC_T, r["names"][0]))
        if not os.path.exists(first) or os.stat(first).st_size != r["size"]:
            missing.append(r["names"][0])
            continue
        for name in r["names"][1:]:
            p = under_target(os.path.join(JSC_T, name))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if os.path.lexists(p):
                assert os.path.samefile(p, first), f"exists, differs: {p}"
                continue
            os.link(first, p)
            n_hl += 1
    print(f"symlinks {n_sym}, hardlinks {n_hl}, copied {len(copies)} (class C); "
          f"A/B without a hash match {len(skipped)}, C/D/E not (yet) present {len(missing)}")
    for p in (skipped + missing)[:10]:
        print("  ", p)


def cmd_verify(a):
    bad = 0
    for r in read_manifest(a.manifest):
        for name in r["names"]:
            p = os.path.join(JSC_T, name)
            try:
                s = os.stat(p).st_size
            except OSError:
                s = None
            if s != r["size"]:
                bad += 1
                if bad <= 10:
                    print("BAD", name, s, r["size"])
    print("verify:", "OK" if not bad else f"{bad} bad names")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["manifest", "hash", "stage", "link", "verify"])
    ap.add_argument("--manifest", default="v8_manifest.tsv")
    ap.add_argument("--out", default="v8_manifest.tsv")
    ap.add_argument("--side", choices=["mn5", "jsc"], default="mn5")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--stage", default=MN5_DATA + "/_jsc_export_stage")
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--pairs", default="jsc_export_pairs.txt")
    ap.add_argument("--hashes-ok", default="", help="inodes whose MN5 and JSC hashes matched (one per line)")
    ap.add_argument("--copy-jobs", type=int, default=8)
    a = ap.parse_args()
    sys.exit({"manifest": cmd_manifest, "hash": cmd_hash, "stage": cmd_stage, "link": cmd_link,
              "verify": cmd_verify}[a.mode](a) or 0)


if __name__ == "__main__":
    main()
