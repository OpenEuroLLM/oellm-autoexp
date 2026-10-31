"""Apply labels.json (display name, tags, notes per W&B run id) to the synced runs; a run not
uploaded yet is skipped. Called by sync_loop.sh after every pass.

    python apply_labels.py [labels.json] [entity/project]
"""

import json
import sys
from pathlib import Path

import wandb


def main() -> None:
    labels_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("labels.json")
    project = sys.argv[2] if len(sys.argv) > 2 else "openeurollm-project/flag_hybrid"
    labels = {k: v for k, v in json.loads(labels_path.read_text()).items() if not k.startswith("_")}
    api = wandb.Api(timeout=60)
    for rid, want in labels.items():
        try:
            run = api.run(f"{project}/{rid}")
        except Exception:
            print(f"labels {rid}: not uploaded yet")
            continue
        changed = False
        for key in ("name", "tags", "notes"):
            if key in want and getattr(run, key) != want[key]:
                setattr(run, key, want[key])
                changed = True
        if changed:
            run.update()
            print(f"labels {rid}: set ({want.get('name')})")


if __name__ == "__main__":
    main()
