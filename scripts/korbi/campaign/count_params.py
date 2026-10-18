"""Parameter counts of a sweep's arms, from Megatron itself, and d_ffn matching.

    python scripts/korbi/campaign/count_params.py experiments/korbi/campaign_3b/screen_s1_mn5 \
        [--arms attn h_ckda ...] [--match attn] [--write s1]

Renders the sweep (autoexp dry run), takes each arm's pretrain_hybrid.py command, replaces data /
parallelism / schedule by a 1-GPU, 1-step mock setup (gloo instead of NCCL, no CUDA graphs) and
runs it inside the hybrid-ep image on the local GPU until Megatron prints the parameter count.
Non-embedding = total - padded_vocab x hidden (tied embeddings). With --match BASE, d_ffn of every
other arm is solved so its non-embedding count matches BASE's (SwiGLU: 3 x hidden x d_ffn per MLP
layer, rounded to 64) and verified by a second count; --write SIZE stores the result in
config/experiments/korbi/campaign_3b/param_match.json under SIZE.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SIF = Path.home() / ".cache/containers/MegatronTraining-hybrid-ep_x86_64.sif"
PM = REPO / "config/experiments/korbi/campaign_3b/param_match.json"
# the extras the image lacks (liger_kernel, mlstm_kernels): a local copy of MN5's $HYBRID_ENV/pylibs_ep
PYLIBS = os.environ.get("PYLIBS_EP", str(Path.home() / ".cache/pylibs_ep"))
DROP = {  # flags (with their value count) replaced for the local count
    "--data-args-path": 1, "--data-path": None, "--split": 1, "--tokenizer-type": 1,
    "--tokenizer-model": 1, "--data-cache-path": 1, "--micro-batch-size": 1,
    "--global-batch-size": 1, "--train-iters": 1, "--cuda-graph-impl": 1,
    "--tensor-model-parallel-size": 1, "--pipeline-model-parallel-size": 1,
    "--expert-model-parallel-size": 1, "--context-parallel-size": 1, "--tensorboard-dir": 1,
    "--save": 1, "--load": 1, "--eval-interval": 1, "--eval-iters": 1, "--cuda-graph-modules": None,
    "--wandb-project": 1, "--wandb-exp-name": 1, "--vocab-size": 1, "--padded-vocab-size": 1,
    "--profile": 0, "--use-pytorch-profiler": 0, "--num-workers": 1, "--lr-warmup-iters": 1,
    "--lr-wsd-decay-iters": 1, "--lr-wsd-decay-style": 1, "--lr-decay-style": 1,
}


def render(config: str, out: Path) -> dict[str, list[str]]:
    env = dict(os.environ, OUTPUT_DIR=str(out))
    subprocess.run(["uv", "run", "python", "scripts/run_autoexp.py", "--config-name", config,
                    "--dry-run"], cwd=REPO, env=env, check=True, capture_output=True)
    cmds = {}
    for task in out.rglob("task_*.sh"):
        text = task.read_text().replace("\\\n", " ")
        m = re.search(r"pretrain_hybrid\.py(.*?)(?:\n|$)", text)
        arm = re.search(r"/([A-Za-z0-9_]+)/tb\b", text) or re.search(r"--tensorboard-dir \S*/([^/\s]+)/", text)
        args = shlex.split(m.group(1))
        cmds[arm.group(1) if arm else task.stem] = args
    return cmds


def strip(args: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in DROP:
            n = DROP[a]
            i += 1
            if n is None:  # variable count: skip until the next flag
                while i < len(args) and not args[i].startswith("--"):
                    i += 1
            else:
                i += n
            continue
        out.append(a)
        i += 1
    return out


def count(args: list[str]) -> tuple[int, int]:
    """(total params, hidden) from a 1-GPU, 1-step mock run."""
    local = strip(args) + [
        "--mock-data", "--tokenizer-type", "NullTokenizer", "--vocab-size", "262143",
        "--micro-batch-size", "1", "--global-batch-size", "1", "--train-iters", "1",
        "--cuda-graph-impl", "none", "--distributed-backend", "gloo", "--eval-iters", "0",
        "--eval-interval", "1000000", "--num-workers", "0", "--lr-warmup-iters", "0",
        "--lr-decay-style", "constant"]
    hidden = int(local[local.index("--hidden-size") + 1])
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        site = Path(td) / "site"
        site.mkdir()
        cmd = ["apptainer", "exec", "--nv", "--cleanenv", "--no-mount", "/etc/resolv.conf",
               "--bind", "/tmp", str(SIF), "env",
               f"PYTHONPATH={REPO}:{REPO}/submodules/Megatron-LM:{REPO}/flash-linear-attention:"
               f"{REPO}/submodules/cute_kernels:{PYLIBS}", f"TRITON_CACHE_DIR={td}/triton",
               f"TORCHINDUCTOR_CACHE_DIR={td}/inductor", f"HOME={td}", "CUDA_DEVICE_MAX_CONNECTIONS=1",
               "python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "1",
               str(REPO / "submodules/Megatron-LM/pretrain_hybrid.py"), *local]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                cwd=REPO)
        total, tail = None, []
        for line in proc.stdout:
            tail = (tail + [line])[-200:]
            m = re.search(r"number of parameters on \([a-z_, ]+\) model parallel rank "
                          r"\([0-9, ]+\): (\d+)", line)
            if m:
                total = int(m.group(1))
                proc.kill()
                break
        proc.wait()
    if total is None:
        raise RuntimeError("no parameter count:\n" + "".join(tail))
    return total, hidden


def n_mlp(args: list[str]) -> int:
    pattern = args[args.index("--hybrid-layer-pattern") + 1]
    return pattern.count("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--match")
    ap.add_argument("--target-non-emb", type=float,
                    help="match every arm to this non-embedding count (instead of an arm of the suite, "
                         "e.g. a reference counted in another suite)")
    ap.add_argument("--write")
    ap.add_argument("--key-prefix", default="", help="prefix of the arm keys written to param_match.json")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        cmds = render(a.config, Path(td))
    arms = a.arms or sorted(cmds)
    res = {}
    for arm in arms:
        args = cmds[arm]
        total, hidden = count(args)
        ffn = int(args[args.index("--ffn-hidden-size") + 1])
        res[arm] = dict(total=total, non_emb=total - 262144 * hidden, ffn=ffn, hidden=hidden,
                        n_mlp=n_mlp(args), args=args)
        print(f"{arm:16s} total {total / 1e6:8.2f}M  non-emb {res[arm]['non_emb'] / 1e6:8.2f}M  d_ffn {ffn}")
    if a.match or a.target_non_emb:
        target = res[a.match]["non_emb"] if a.match else a.target_non_emb
        for arm, r in res.items():
            if arm == a.match:
                continue
            for _ in range(3):
                delta = (target - r["non_emb"]) / (3 * r["hidden"] * r["n_mlp"])
                ffn = int(round((r["ffn"] + delta) / 64) * 64)
                if ffn == r["ffn"]:
                    break
                args = list(r["args"])
                args[args.index("--ffn-hidden-size") + 1] = str(ffn)
                total, hidden = count(args)
                r.update(total=total, non_emb=total - 262144 * hidden, ffn=ffn, args=args)
            print(f"matched {arm:16s} d_ffn {r['ffn']:5d}  non-emb {r['non_emb'] / 1e6:8.2f}M "
                  f"({100 * (r['non_emb'] / target - 1):+.2f}% vs {a.match or f'{target / 1e6:.2f}M'})")
    if a.write:
        pm = json.loads(PM.read_text()) if PM.exists() else {}
        # merge: counting a subset of arms keeps the other arms' matches
        pm.setdefault(a.write, {}).update(
            {a.key_prefix + arm: {k: r[k] for k in ("total", "non_emb", "ffn")} for arm, r in res.items()})
        PM.write_text(json.dumps(pm, indent=1) + "\n")
        print(f"wrote {PM} [{a.write}]")


if __name__ == "__main__":
    main()
