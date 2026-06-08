"""
Fix stuck/cancelled monitor session entries for the 0.9B cross session (1780477781).

Two classes of fixes:
  1. Decay jobs with final_state="cancelled" → reset so monitor resubmits them
     when their stable checkpoints become available.
  2. lr0.001_gbsz512_stable300BT → stuck with runtime_job_id=44131285 (failed
     in 4 s due to srun error, monitor can't detect it). Reset runtime so it
     gets resubmitted on next poll.

Run this while the monitor is paused (Ctrl-Z in the tmux session) then
resume it (fg) to let it pick up the changes.
"""

import json
import shutil
from pathlib import Path

SESSION_DIR = Path("monitor_state/1780477781")

# --- Decay jobs that were incorrectly marked final_state="cancelled" ---
CANCELLED_DECAYS = [
    "qwen3_dense_0.9B_ne_lr0.001_gbsz512_decay80BT_166_8c29c8.job.json",
    "qwen3_dense_0.9B_ne_lr0.001_gbsz512_decay200BT_168_a09738.job.json",
    "qwen3_dense_0.9B_ne_lr0.001_gbsz512_decay300BT_169_956401.job.json",
    "qwen3_dense_0.9B_ne_lr0.002_gbsz256_decay80BT_136_78ed15.job.json",
    "qwen3_dense_0.9B_ne_lr0.002_gbsz256_decay120BT_137_5e1a67.job.json",
    "qwen3_dense_0.9B_ne_lr0.002_gbsz256_decay200BT_138_3231a6.job.json",
    "qwen3_dense_0.9B_ne_lr0.002_gbsz256_decay300BT_139_7e2cb1.job.json",
    "qwen3_dense_0.9B_ne_lr0.002_gbsz1024_decay200BT_208_7c340a.job.json",
    "qwen3_dense_0.9B_ne_lr0.002_gbsz1024_decay300BT_209_438b67.job.json",
]

# --- Stable job stuck with a dead job ID ---
STUCK_STABLE = "qwen3_dense_0.9B_ne_lr0.001_gbsz512_stable300BT_160_2a14c6.job.json"


def patch_file(path: Path, dry_run: bool = False) -> None:
    data = json.loads(path.read_text())
    rt = data["runtime"]
    before = {
        "final_state": rt.get("final_state"),
        "submitted": rt.get("submitted"),
        "runtime_job_id": rt.get("runtime_job_id"),
        "attempts": rt.get("attempts"),
    }

    if path.name in CANCELLED_DECAYS:
        # Clear final_state so monitor treats it as active again.
        # Keep attempts so we don't lose restart history.
        rt["final_state"] = None
        rt["submitted"] = False
        rt["runtime_job_id"] = None
        rt["log_cursor"] = 0
        rt["action_state"] = {}   # clear so events can re-fire
    elif path.name == STUCK_STABLE:
        # Reset runtime — monitor will detect no job and resubmit.
        rt["final_state"] = None
        rt["submitted"] = False
        rt["runtime_job_id"] = None
        rt["log_cursor"] = 0
        rt["last_status"] = None
        rt["action_state"] = {}   # let error/failed events fire fresh
        # keep attempts counter intact

    after = {
        "final_state": rt.get("final_state"),
        "submitted": rt.get("submitted"),
        "runtime_job_id": rt.get("runtime_job_id"),
        "attempts": rt.get("attempts"),
    }

    print(f"{'[DRY RUN] ' if dry_run else ''}Patching {path.name}")
    print(f"  before: {before}")
    print(f"  after:  {after}")

    if not dry_run:
        backup = path.with_suffix(".bak.json")
        shutil.copy2(path, backup)
        path.write_text(json.dumps(data, indent=2))
        print(f"  backed up to {backup.name}")


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv

    for name in CANCELLED_DECAYS:
        p = SESSION_DIR / name
        if not p.exists():
            print(f"WARNING: not found: {name}")
            continue
        patch_file(p, dry_run=dry_run)

    p = SESSION_DIR / STUCK_STABLE
    if p.exists():
        patch_file(p, dry_run=dry_run)
    else:
        print(f"WARNING: not found: {STUCK_STABLE}")

    print("\nDone." if not dry_run else "\nDry-run complete — no files changed.")
