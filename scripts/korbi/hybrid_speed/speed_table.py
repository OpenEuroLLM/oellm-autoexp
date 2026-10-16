"""Speed table of a hybrid speed-suite run (config/experiments/korbi/hybrid_speed).

    python scripts/korbi/hybrid_speed/speed_table.py $OUTPUT_DIR/hybrid_speed/v1_dense130M \
        [--reference dense] [--skip 30] [--csv table.csv] [--json table.json]

A suite directory holds one sub-directory per arm and the array group's directory with
the task scripts (task_<i>.sh) and the Slurm logs (slurm-<array>_<i>.log; the newest
attempt of every task wins). Per arm, from Megatron's training log:

  step ms     median elapsed time per iteration over the iterations after --skip
              (CUDA-graph capture, autotuning and compilation happen before)
  TFLOP/s     median of Megatron's "throughput per GPU" (its analytic FLOP count
              per architecture; compare tokens/s across arms whose FLOP formulas differ)
  tok/s/GPU   global batch x sequence length / step time / GPUs
  TF/step     the model FLOPs of one step that Megatron's count implies
  mem GB      max allocated (rank 0)
  params      Megatron's parameter count (rank 0)
  vs ref      TFLOP/s and tok/s relative to the reference arm

Arms whose log has no measured iteration are listed with the last error line.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ITER = re.compile(
    r"iteration\s+(\d+)/\s*(\d+) \|.*?elapsed time per iteration \(ms\): ([\d.]+)"
    r"(?:.*?throughput per GPU \(TFLOP/s/GPU\): ([\d.]+))?"
)
GBS = re.compile(r"global batch size:\s+(\d+)")
PARAMS = re.compile(r"number of parameters on \([\w, ]+\) model parallel rank \([0, ]+\): (\d+)")
MEM = re.compile(r"max allocated: ([\d.]+)")
SEQ = re.compile(r"--seq-length (\d+)")
NPROC = re.compile(r"--nproc-per-node (\d+)")
NNODES = re.compile(r"--nnodes (\d+)")
ARM = re.compile(r"/([^/\s]+)/wandb")
ERR = re.compile(r"(\w*(Error|Exception)\b[^\n]{0,200}|CUDA out of memory[^\n]{0,120}|Segmentation fault)")


def _tasks(suite: Path):
    """(arm, task script, newest log) per task of the suite's array group(s)."""
    for script in sorted(suite.glob("*/task_*.sh"), key=lambda p: int(p.stem.split("_")[1])):
        index = script.stem.split("_")[1]
        text = script.read_text()
        arm = ARM.search(text)
        logs = sorted(script.parent.glob(f"slurm-*_{index}.log"), key=lambda p: p.stat().st_mtime)
        yield (arm.group(1) if arm else f"task{index}"), text, (logs[-1] if logs else None)


def measure(text: str, log: Path | None, skip: int) -> dict:
    seq = int(SEQ.search(text).group(1)) if SEQ.search(text) else 0
    gpus = int(NPROC.search(text).group(1)) * int(NNODES.search(text).group(1)) if NPROC.search(text) else 0
    row = {"status": "no log", "log": str(log) if log else None}
    if log is None:
        return row
    content = log.read_text(errors="replace")
    steps, tflops = [], []
    for m in ITER.finditer(content):
        if int(m.group(1)) > skip:
            steps.append(float(m.group(3)))
            if m.group(4):
                tflops.append(float(m.group(4)))
    params = PARAMS.search(content)
    mem = [float(x) for x in MEM.findall(content)]
    gbs = GBS.search(content)
    row.update(params=int(params.group(1)) if params else None,
               mem_gb=max(mem) / 1024 if mem else None)
    if not steps:
        errors = [m.group(1) for m in ERR.finditer(content)]
        row["status"] = "FAIL: " + (errors[-1] if errors else "no measured iteration")
        return row
    step_ms = statistics.median(steps)
    row.update(status="ok", step_ms=step_ms, n_steps=len(steps),
               step_spread=(max(steps) - min(steps)) / step_ms)
    if tflops:
        row["tflops"] = statistics.median(tflops)
        row["tf_per_step"] = row["tflops"] * gpus * step_ms / 1000
    if gbs and seq and gpus:
        row["tok_s_gpu"] = int(gbs.group(1)) * seq / (step_ms / 1000) / gpus
    return row


def table(suite: Path, reference: str, skip: int) -> list[dict]:
    rows = []
    for arm, text, log in _tasks(suite):
        rows.append({"arm": arm, **measure(text, log, skip)})
    ref = next((r for r in rows if r["arm"] == reference and r["status"] == "ok"), None)
    for r in rows:
        if ref and r["status"] == "ok":
            if r.get("tflops") and ref.get("tflops"):
                r["tflops_rel"] = r["tflops"] / ref["tflops"]
            if r.get("tok_s_gpu") and ref.get("tok_s_gpu"):
                r["tok_rel"] = r["tok_s_gpu"] / ref["tok_s_gpu"]
    return rows


def markdown(rows: list[dict], reference: str, threshold: float) -> str:
    def f(v, fmt):
        return format(v, fmt) if isinstance(v, (int, float)) else "-"

    out = [
        f"reference: `{reference}`; flagged (!) below {threshold:.0%} of its TFLOP/s\n",
        "| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        flag = " (!)" if r.get("tflops_rel", 1.0) < threshold else ""
        params = f"{r['params'] / 1e6:.1f}M" if r.get("params") else "-"
        out.append(
            f"| {r['arm']}{flag} | {params} | {f(r.get('step_ms'), '.0f')} | {f(r.get('tflops'), '.1f')} "
            f"| {f(r.get('tflops_rel'), '.2f')} | {f(r.get('tok_s_gpu'), ',.0f')} | {f(r.get('tok_rel'), '.2f')} "
            f"| {f(r.get('tf_per_step'), '.0f')} | {f(r.get('mem_gb'), '.1f')} | {r['status'][:90]} |"
        )
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", type=Path, help="the suite's output directory (one per family)")
    p.add_argument("--reference", default=None, help="reference arm (default: dense, or moe)")
    p.add_argument("--skip", type=int, default=30, help="iterations excluded as warm-up")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--csv", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    rows = table(args.suite, args.reference or "dense", args.skip)
    reference = args.reference or ("dense" if any(r["arm"] == "dense" for r in rows) else "moe")
    if reference != (args.reference or "dense"):
        rows = table(args.suite, reference, args.skip)
    print(markdown(rows, reference, args.threshold))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    if args.csv:
        keys = sorted({k for r in rows for k in r})
        lines = [",".join(keys)] + [",".join(str(r.get(k, "")) for k in keys) for r in rows]
        args.csv.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
