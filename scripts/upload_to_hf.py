#!/usr/bin/env python3
"""Upload converted HF checkpoints to a Hub repo, one branch per training
iteration.

Each local <output-dir>/<iter>/ directory (as produced by
mass_convert_checkpoints.py) is pushed to a branch named after that
iteration (e.g. branch "iter_0002400"), with files at the branch root, so
`AutoModelForCausalLM.from_pretrained(repo_id, revision="iter_0002400")`
just works.

Needs internet access -- run on Leonardo's lrd_all_serial partition (login
nodes exposed as a schedulable 4h queue), not a compute node.

Safe to re-run: a branch is only considered done if it actually has the
model weight files (not just if the branch ref exists), so a crash or
network failure mid-upload gets retried automatically rather than silently
treated as complete. Failures are logged to
<output-dir>/manifests/upload_failures.json and retried on the next pass.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, type=Path,
                     help="Local dir containing iter_* checkpoint subdirs (mass_convert_checkpoints.py output)")
    ap.add_argument("--repo-id", required=True, help="Target HF Hub repo id, e.g. openeurollm/prelude")
    ap.add_argument("--token-file", default=str(Path.home() / ".cache" / "huggingface" / "token"))
    ap.add_argument("--iters", nargs="*", default=None, help="Only these iter names; default: all found")
    ap.add_argument("--force", action="store_true", help="Re-upload even if the branch already looks complete")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watch", action="store_true",
                     help="Keep polling for newly-completed checkpoints and upload them, instead of a single pass")
    ap.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls in --watch mode")
    ap.add_argument("--shard-count", type=int, default=1, help="Split discovered checkpoints across N parallel workers")
    ap.add_argument("--shard-index", type=int, default=0, help="This worker shard, 0-indexed, less than shard-count")
    ap.add_argument("--upload-workers", type=int, default=6, help="Parallel workers for upload_large_folder (a checkpoint has ~6 files, so more rarely helps)")
    args = ap.parse_args()

    token = Path(args.token_file).read_text().strip()
    api = HfApi(token=token)

    failures_path = args.output_dir / "manifests" / "upload_failures.json"
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}

    def write_failures():
        failures_path.write_text(json.dumps(failures, indent=2))

    def discover():
        if args.iters:
            found = [args.output_dir / it for it in args.iters]
        else:
            found = sorted(
                d for d in args.output_dir.iterdir()
                if d.is_dir() and d.name.startswith("iter_") and (d / "validation.json").exists()
            )
        if args.shard_count > 1:
            found = [d for i, d in enumerate(found) if i % args.shard_count == args.shard_index]
        return found

    def branch_is_complete(branch: str) -> bool:
        try:
            files = set(api.list_repo_files(args.repo_id, revision=branch))
        except Exception:
            return False
        has_weights = any(
            f == "model.safetensors" or f == "model.safetensors.index.json" for f in files
        )
        return has_weights and "config.json" in files and "validation.json" in files

    def upload_one(d: Path) -> None:
        branch = d.name
        if branch not in {b.name for b in api.list_repo_refs(args.repo_id).branches}:
            api.create_branch(args.repo_id, branch=branch, exist_ok=True)
        # upload_large_folder: resumable (state cached under d/.cache/.huggingface/),
        # retries transient errors internally, parallel workers per-file. Safe to run
        # concurrently across DIFFERENT (folder, revision) pairs (our shards each own a
        # disjoint set of iters); just don't run two processes on the *same* one.
        api.upload_large_folder(
            repo_id=args.repo_id,
            folder_path=str(d),
            repo_type="model",
            revision=branch,
            num_workers=args.upload_workers,
            print_report=False,
        )

    def upload_pass(complete_branches: set[str]) -> tuple[int, int, int]:
        iter_dirs = discover()
        uploaded, skipped, failed = 0, 0, 0
        for d in iter_dirs:
            branch = d.name
            if branch in complete_branches and not args.force:
                skipped += 1
                continue
            if args.dry_run:
                print(f"[dry-run] would create branch {branch} and upload {d}")
                continue
            try:
                upload_one(d)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {branch}: {exc}", flush=True)
                failures[branch] = str(exc)
                write_failures()
                failed += 1
                continue
            print(f"uploaded {branch}", flush=True)
            complete_branches.add(branch)
            failures.pop(branch, None)
            write_failures()
            uploaded += 1
        return uploaded, skipped, failed

    print("checking existing branches for completeness...")
    all_branches = {b.name for b in api.list_repo_refs(args.repo_id).branches}
    complete_branches = {b for b in all_branches if branch_is_complete(b)}
    incomplete = sorted(all_branches - complete_branches - {"main"})
    if incomplete:
        print(f"{len(incomplete)} existing branch(es) look incomplete, will (re)upload: {incomplete}")
    print(f"{len(discover())} checkpoint dir(s) currently discoverable for this shard")

    if not args.watch:
        uploaded, skipped, failed = upload_pass(complete_branches)
        print(f"done: {uploaded} uploaded, {skipped} already complete, {failed} failed")
        if failures:
            print(f"failures logged to {failures_path}: {sorted(failures)}")
        return 0

    print(f"--watch mode: polling every {args.poll_interval}s (stop with scancel or let the job time out)")
    while True:
        uploaded, skipped, failed = upload_pass(complete_branches)
        if uploaded or failed:
            print(f"pass done: {uploaded} uploaded, {skipped} already complete, {failed} failed", flush=True)
        if failures:
            print(f"current failures: {sorted(failures)}", flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
