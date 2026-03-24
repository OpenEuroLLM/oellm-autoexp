#!/usr/bin/env python3
"""
Sync wandb offline runs to the cloud.

Usage:
    # Sync all runs in a folder
    python sync_runs.py --folder moe_200MA50M_10BT
    
    # Sync a specific run
    python sync_runs.py --rundir moe_200MA50M_10BT/moe_200MA50M_10BT_schedWSD_lr0.001_gbsz512_beta20.95
    
    # Dry run (just print commands)
    python sync_runs.py --folder moe_200MA50M_10BT --dry-run
    
    # Specify results directory
    python sync_runs.py --folder moe_200MA50M_10BT --results-dir /path/to/results
"""

import argparse
import subprocess
import sys
from pathlib import Path


def is_run_synced(offline_run: Path) -> bool:
    """Check if a wandb offline run has already been synced."""
    return bool(list(offline_run.glob("*.synced")))


def find_offline_runs(base_path: Path, include_synced: bool = False) -> tuple[list[Path], int]:
    """Find all wandb offline run directories under a path.
    
    Returns:
        Tuple of (runs_to_sync, skipped_count)
    """
    offline_runs = []
    skipped = 0
    
    # Look for wandb/offline-run-* patterns
    for wandb_dir in base_path.rglob("wandb"):
        if wandb_dir.is_dir():
            for offline_run in wandb_dir.glob("offline-run-*"):
                if offline_run.is_dir():
                    if not include_synced and is_run_synced(offline_run):
                        skipped += 1
                        continue
                    offline_runs.append(offline_run)
    
    return sorted(offline_runs), skipped


def sync_run(offline_run_path: Path, dry_run: bool = False) -> bool:
    """Sync a single wandb offline run."""
    # Remove trailing slash to avoid wandb CLI bug
    path_str = str(offline_run_path).rstrip("/")
    
    cmd = ["wandb", "sync", path_str]
    
    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True
    
    print(f"\n{'='*60}")
    print(f"Syncing: {path_str}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Warning: wandb sync returned non-zero exit code: {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"Error syncing {path_str}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sync wandb offline runs to the cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--folder", "-f",
        help="Folder name under results directory (e.g., moe_200MA50M_10BT)"
    )
    group.add_argument(
        "--rundir", "-r",
        help="Specific run directory path (e.g., moe_200MA50M_10BT/moe_200MA50M_10BT_schedWSD_lr0.001_gbsz512_beta20.95)"
    )
    
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base results directory (default: results)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--continue-on-error", "-c",
        action="store_true",
        help="Continue syncing other runs if one fails"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-sync runs even if they were already synced"
    )
    
    args = parser.parse_args()
    
    # Determine the base path to search
    results_dir = Path(args.results_dir)
    
    if args.folder:
        base_path = results_dir / args.folder
    else:
        # rundir can be relative to results or absolute
        rundir_path = Path(args.rundir)
        if rundir_path.is_absolute():
            base_path = rundir_path
        elif (results_dir / args.rundir).exists():
            base_path = results_dir / args.rundir
        else:
            base_path = Path(args.rundir)
    
    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        sys.exit(1)
    
    print(f"Searching for offline runs in: {base_path}")
    
    # Find all offline runs
    offline_runs, skipped = find_offline_runs(base_path, include_synced=args.force)
    
    if not offline_runs and skipped == 0:
        print(f"No offline runs found in {base_path}")
        sys.exit(0)
    
    if skipped > 0:
        print(f"\nSkipped {skipped} already-synced run(s) (use --force to re-sync)")
    
    if not offline_runs:
        print("All runs already synced!")
        sys.exit(0)
    
    print(f"\nFound {len(offline_runs)} offline run(s) to sync:")
    for run in offline_runs:
        print(f"  - {run}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No actual syncing will occur]\n")
    
    # Sync each run
    success_count = 0
    fail_count = 0
    
    for offline_run in offline_runs:
        success = sync_run(offline_run, dry_run=args.dry_run)
        if success:
            success_count += 1
        else:
            fail_count += 1
            if not args.continue_on_error and not args.dry_run:
                print(f"\nStopping due to error. Use --continue-on-error to continue.")
                break
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {success_count} succeeded, {fail_count} failed out of {len(offline_runs)} total")
    print(f"{'='*60}")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
