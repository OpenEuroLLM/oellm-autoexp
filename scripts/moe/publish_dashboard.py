"""Stage, compress, and build the publishable static dashboard.

Output layout (default): _publish/
  index.html, style.css, app.js, manifest.json
  bsz<X>/nexp_<Y>/lr<Z>/120BT/...assets...

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


def stage_bucket(src: Path, dst: Path, include_json: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in REFERENCED_TOP:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    for pat in REFERENCED_GLOBS:
        for s in src.glob(pat):
            shutil.copy2(s, dst / s.name)
    for sub, pat in REFERENCED_SUBDIR_GLOBS.items():
        sub_src = src / sub
        if not sub_src.is_dir():
            continue
        sub_dst = dst / sub
        sub_dst.mkdir(exist_ok=True)
        for s in sub_src.glob(pat):
            shutil.copy2(s, sub_dst / s.name)
    if include_json:
        for name in JSON_FILES:
            s = src / name
            if s.exists():
                shutil.copy2(s, dst / name)


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
    parser.add_argument("--root", type=Path, default=repo_root / "results" / "moe_analysis")
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
                        help="Remove the publish dir before staging.")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    out: Path = args.publish_dir.resolve()
    if not root.is_dir():
        sys.exit(f"results root not found: {root}")

    if args.clean and out.exists():
        print(f"Removing {out} ...")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Stage
    print(f"Staging buckets from {root} -> {out}")
    buckets = sorted([b for b in root.glob("bsz*/nexp_*/lr*/120BT") if b.is_dir()])
    print(f"  found {len(buckets)} buckets")
    for src in buckets:
        rel = src.relative_to(root)
        dst = out / rel
        stage_bucket(src, dst, include_json=args.include_json)
        if args.no_routing_per_layer:
            sub = dst / "expert_routing_per_layer"
            if sub.is_dir():
                shutil.rmtree(sub)

    # 2. Compress
    print("Compressing PNGs ...")
    pngs = list(out.rglob("*.png"))
    parallel_apply("PNG optimize", pngs, compress_png, args.workers)

    print("Re-encoding GIFs as animated WebP (process pool, method=2) ...")
    gifs = list(out.rglob("*.gif"))
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
        "--root", str(root),
        "--publish-dir", str(out),
        "--gif-ext", "webp",
    ]
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
