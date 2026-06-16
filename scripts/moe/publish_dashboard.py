"""Stage, compress, and build the publishable static dashboard.

Output layout (default): _publish/
  index.html, style.css, app.js, manifest.json
  <root-name>/bsz<X>/nexp_<Y>/lr<Z>/120BT/...assets...

  e.g.
    moe_analysis/bsz256/nexp_8/lr0.001/120BT/...
    moe_analysis_global_batch_aux/bsz256/nexp_8/lr0.001/120BT/...
    moe_analysis_deepseek_bias/bsz256/nexp_8/lr0.001/120BT/...

Usage (run with the miniforge python that has Pillow):
  $MINIFORGE/bin/python scripts/moe/publish_dashboard.py [options]

After this:
  cd _publish && git init -b main && git add . && git commit -m 'init' \\
    && git remote add origin git@github.com:<user>/moe-analysis-dashboard.git \\
    && git push -u origin main

Then enable Pages on that repo (Settings -> Pages, branch=main, /).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

# Imported lazily so --help works even without Pillow.
def _pil():
    from PIL import Image  # noqa
    return Image


REFERENCED_TOP = [
    "expert_activation_norms.png",
    "expert_activation_max_over_median.png",
    "router_saturation_vs_final.png",
    "expert_routing.gif",
]
REFERENCED_GLOBS = [
    "coactivation_layer_*.png",
]
REFERENCED_SUBDIR_GLOBS = {
    "activation_norms_per_layer": "activation_norms_layer_*.png",
    "expert_routing_per_layer": "expert_routing_layer_*.gif",
}
JSON_FILES = [
    "saturation.json",
    "coactivation.json",
    "token_counts.json",
    "expert_activation_norms.json",
]


def _up_to_date(src: Path, dst: Path) -> bool:
    """True if dst already reflects src (same-or-newer mtime). For images, the
    staged copy may have been compressed in place (PNG keeps name) or a GIF may
    have been re-encoded to .webp and the .gif deleted; treat the .webp sibling
    as the staged artifact for gifs."""
    candidates = [dst]
    if dst.suffix == ".gif":
        candidates.append(dst.with_suffix(".webp"))
    src_mtime = src.stat().st_mtime
    for c in candidates:
        if c.exists() and c.stat().st_mtime >= src_mtime:
            return True
    return False


def stage_bucket(src: Path, dst: Path, include_json: bool, incremental: bool = False) -> list[Path]:
    """Copy referenced assets from src->dst. Returns the list of paths actually
    written this call (so the caller can compress only new files)."""
    written: list[Path] = []

    def _copy(s: Path, d: Path) -> None:
        if incremental and _up_to_date(s, d):
            return
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        written.append(d)

    dst.mkdir(parents=True, exist_ok=True)
    for name in REFERENCED_TOP:
        s = src / name
        if s.exists():
            _copy(s, dst / name)
    for pat in REFERENCED_GLOBS:
        for s in src.glob(pat):
            _copy(s, dst / s.name)
    for sub, pat in REFERENCED_SUBDIR_GLOBS.items():
        sub_src = src / sub
        if not sub_src.is_dir():
            continue
        sub_dst = dst / sub
        sub_dst.mkdir(exist_ok=True)
        for s in sub_src.glob(pat):
            _copy(s, sub_dst / s.name)
    if include_json:
        for name in JSON_FILES:
            s = src / name
            if s.exists():
                _copy(s, dst / name)
    return written


def compress_png(p_str: str) -> tuple[int, int]:
    Image = _pil()
    p = Path(p_str)
    before = p.stat().st_size
    img = Image.open(p)
    img.load()
    img.save(p, format="PNG", optimize=True)
    return before, p.stat().st_size


def gif_to_webp(g_str: str, quality: int = 70, method: int = 2) -> tuple[int, int]:
    """Re-encode an animated GIF as animated WebP. Removes the .gif on success.
    Uses Pillow's streaming save (no all-frames-in-RAM) and method=2 by default
    (30x faster than method=6 with comparable size)."""
    Image = _pil()
    g = Path(g_str)
    before = g.stat().st_size
    out = g.with_suffix(".webp")
    img = Image.open(g)
    img.save(out, format="WEBP", save_all=True, quality=quality, method=method, loop=0)
    after = out.stat().st_size
    g.unlink()
    return before, after


def parallel_apply(label: str, paths: list[Path], fn, workers: int, executor: str = "thread") -> tuple[int, int]:
    total_before = total_after = 0
    if not paths:
        print(f"  {label}: 0 files")
        return 0, 0
    Executor = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    str_paths = [str(p) for p in paths]
    with Executor(max_workers=workers) as ex:
        futures = {ex.submit(fn, p): p for p in str_paths}
        done = 0
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if isinstance(result, tuple) and len(result) >= 2:
                    total_before += result[0]
                    total_after += result[1]
            except Exception as e:
                print(f"  ! {futures[fut]}: {e}", file=sys.stderr)
            done += 1
            if done % 100 == 0 or done == len(paths):
                print(f"  {label}: {done}/{len(paths)}", end="\r", flush=True)
    print()
    saved = total_before - total_after
    pct = (saved / total_before * 100) if total_before else 0
    print(f"  {label}: {total_before/1048576:.1f} MB -> {total_after/1048576:.1f} MB  (-{pct:.0f}%)")
    return total_before, total_after


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", default=None,
                        help="Results root to publish as a tab (repeatable). "
                             "Defaults to the three known roots: moe_analysis, "
                             "moe_analysis_global_batch_aux, moe_analysis_deepseek_bias.")
    parser.add_argument("--stage", type=str, action="append", default=None,
                        help="Stage subdir per --root (repeatable; positional with --root). "
                             "Defaults to 120BT for every root. e.g. --stage stable")
    parser.add_argument("--publish-dir", type=Path, default=repo_root / "_publish")
    parser.add_argument("--quality", type=int, default=70, help="WebP quality (lower = smaller).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-json", action="store_true",
                        help="Also stage saturation/coactivation/token_counts/activation_norms JSONs (~127 MB).")
    parser.add_argument("--no-routing-per-layer", action="store_true",
                        help="Skip per-layer routing animations (drops ~760 MB pre-compress).")
    parser.add_argument("--build-script", type=Path, default=repo_root / "scripts" / "moe" / "build_dashboard.py")
    parser.add_argument("--build-python", default="python3.11",
                        help="Interpreter for build_dashboard.py (stdlib only).")
    parser.add_argument("--clean", action="store_true",
                        help="Remove the publish dir before staging (forces a full rebuild).")
    parser.add_argument("--full", action="store_true",
                        help="Re-stage and re-compress every bucket. Default is incremental: "
                             "skip assets already staged & up-to-date, compress only new files.")
    args = parser.parse_args()

    # Incremental unless a full/clean rebuild was requested.
    incremental = not (args.full or args.clean)

    if args.root:
        roots = [r.resolve() for r in args.root]
    else:
        roots = [
            (repo_root / "results" / "moe_analysis").resolve(),
            (repo_root / "results" / "moe_analysis_global_batch_aux").resolve(),
            (repo_root / "results" / "moe_analysis_deepseek_bias").resolve(),
        ]
        roots = [r for r in roots if r.is_dir()]
        if not roots:
            sys.exit("no default roots exist; pass --root explicitly")
    out: Path = args.publish_dir.resolve()
    for r in roots:
        if not r.is_dir():
            sys.exit(f"results root not found: {r}")

    stages = args.stage or []
    if stages and len(stages) != len(roots):
        sys.exit(
            f"--stage given {len(stages)} times but {len(roots)} roots resolved; "
            f"counts must match"
        )
    stages = stages + ["120BT"] * (len(roots) - len(stages))

    if args.clean and out.exists():
        print(f"Removing {out} ...")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Stage each root under _publish/<root-name>/...
    print(f"Mode: {'incremental' if incremental else 'full rebuild'}")
    newly_staged: list[Path] = []
    for root, stage in zip(roots, stages):
        root_out = out / root.name
        print(f"Staging buckets from {root} (stage={stage}) -> {root_out}")
        buckets = sorted([b for b in root.glob(f"bsz*/nexp_*/lr*/{stage}") if b.is_dir()])
        print(f"  found {len(buckets)} buckets")
        for src in buckets:
            rel = src.relative_to(root)
            dst = root_out / rel
            newly_staged += stage_bucket(
                src, dst, include_json=args.include_json, incremental=incremental
            )
            if args.no_routing_per_layer:
                sub = dst / "expert_routing_per_layer"
                if sub.is_dir():
                    shutil.rmtree(sub)
    if incremental:
        print(f"  {len(newly_staged)} new/changed files to compress")

    # 2. Compress. Incremental: only files staged this run; full: everything.
    if incremental:
        pngs = [p for p in newly_staged if p.suffix == ".png"]
        gifs = [p for p in newly_staged if p.suffix == ".gif"]
    else:
        pngs = list(out.rglob("*.png"))
        gifs = list(out.rglob("*.gif"))

    print("Compressing PNGs ...")
    parallel_apply("PNG optimize", pngs, compress_png, args.workers)

    print("Re-encoding GIFs as animated WebP (process pool, method=2) ...")
    parallel_apply(
        "GIF -> WebP",
        gifs,
        partial(gif_to_webp, quality=args.quality, method=2),
        max(1, args.workers // 2),
        executor="process",
    )

    # 3. Build (writes index.html + manifest at the publish root)
    print("Building dashboard at root ...")
    cmd = [
        args.build_python,
        str(args.build_script),
        "--publish-dir", str(out),
        "--gif-ext", "webp",
    ]
    for r, stage in zip(roots, stages):
        cmd += ["--root", str(r), "--stage", stage]
    if not args.include_json:
        cmd.append("--no-raw-json")
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 4. Report
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    n_files = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"\nPublish dir: {out}")
    print(f"  {n_files} files, {total/1048576:.1f} MB")
    if total > 1024 * 1024 * 1024:
        print("  WARNING: > 1 GB - GitHub Pages will warn. Consider --quality 60 or --no-routing-per-layer.")
    elif total > 900 * 1024 * 1024:
        print("  Note: close to the 1 GB GitHub Pages soft cap.")
    else:
        print("  Within GitHub Pages 1 GB soft cap.")


if __name__ == "__main__":
    main()
