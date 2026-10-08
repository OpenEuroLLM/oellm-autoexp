"""One CSV per export with the 438 suite and the cot half: <name>.flag-evals-471.tasks.csv.

    python3 flag_combine.py <suite csv> <cot csv> <out csv> <suite task list> <cot task list>

The task lists are config/eval_tasks/flag_evals.yaml and flag_cot.yaml. A row is kept when its
n_shot is one its suite item runs with: an export's results directory can hold an extra run
of some tasks at another n_shot (v2anneal_120k: arc_challenge_mt_* at 0 shots next to the
suite's 25), which the collected CSV keeps and summarize-evals then rejects as a second result.
A row's suite item is the longest item task that equals the row's task or prefixes it
(global_mmlu_full_de -> global_mmlu_full_de_philosophy); a row without an item is kept.
"""

import csv
import re
import sys


def n_shots(task_list):
    """Task -> the n_shots it runs with (some run twice: arc_challenge at 10
    and 25)."""
    shots = {}
    for t, n in re.findall(r'task: "([^"]+)", n_shot: (\d+)', open(task_list).read()):
        shots.setdefault(t, set()).add(int(n))
    return shots


def item(task, items):
    best = None
    for t in items:
        if (task == t or (task.startswith(t) and task[len(t)] in "_:")) and (
            best is None or len(t) > len(best)
        ):
            best = t
    return best


def main():
    suite_csv, cot_csv, out_csv, suite_list, cot_list = sys.argv[1:6]
    shots = n_shots(suite_list)
    for t, n in n_shots(cot_list).items():
        shots.setdefault(t, set()).update(n)
    readers = [csv.DictReader(open(suite_csv)), csv.DictReader(open(cot_csv))]
    if readers[0].fieldnames != readers[1].fieldnames:
        sys.exit(f"header mismatch: {suite_csv} vs {cot_csv}")
    kept = dropped = 0
    with open(out_csv, "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=readers[0].fieldnames)
        out.writeheader()
        for reader in readers:
            for row in reader:
                t = item(row["task"], shots)
                if t is not None and int(row["n_shot"]) not in shots[t]:
                    dropped += 1
                    continue
                out.writerow(row)
                kept += 1
    print(f"{out_csv}: {kept} rows ({dropped} at an n_shot outside the suite left out)")


if __name__ == "__main__":
    main()
