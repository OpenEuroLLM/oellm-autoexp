"""Scaling efficiency analysis from Megatron-LM SLURM log files.

Fill in GPUS_PER_NODE, VRAM_GB, and EXPERIMENT_SERIES below, then run:
python plot_scaling_mn5.py
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


GPUS_PER_NODE = 4

# Set to a number (e.g. 64) to add a footnote about VRAM constraint; None to omit.
VRAM_GB = 64

# Each entry has a "label" and a "files" dict mapping GPU count → SLURM log path.
# Add or remove series freely; GPU counts can differ across series.
EXPERIMENT_SERIES = [
    {
        "label": "TP=4, MBS=4, GBS=1024",
        "files": {
            4: "output/qwen3-8b/qwen3_8b_1_ep1_pp1_tp4_mbs4_gbs1024_vpNone_none/slurm-442608.log",
            8: "output/qwen3-8b/qwen3_8b_2_ep1_pp1_tp4_mbs4_gbs1024_vpNone_none/slurm-442710.log",
            16: "output/qwen3-8b/qwen3_8b_4_ep1_pp1_tp4_mbs4_gbs1024_vpNone_none/slurm-442712.log",
            32: "output/qwen3-8b/qwen3_8b_8_ep1_pp1_tp4_mbs4_gbs1024_vpNone_none/slurm-442715.log",
            64: "output/qwen3-8b/qwen3_8b_16_ep1_pp1_tp4_mbs4_gbs1024_vpNone_none/slurm-442716.log",
            128: "output/qwen3-8b/qwen3_8b_32_ep1_pp1_tp4_mbs4_gbs1024_vpNone_none/slurm-442901.log",
            256: "output/qwen3-8b/qwen3_8b_64_ep1_pp1_tp4_mbs4_gbs1024_vpNone_none/slurm-442789.log",
            512: "output/qwen3-8b/qwen3_8b_128_ep1_pp1_tp4_mbs4_gbs1024_vpNone_none/slurm-442645.log",
            1024: "output/qwen3-8b/qwen3_8b_256_ep1_pp1_tp4_mbs4_gbs1024_vpNone_none/slurm-442690.log",
        },
    },
    {
        "label": "TP=2, MBS=1, GBS=1024",
        "files": {
            4: "output/qwen3-8b/qwen3_8b_1_ep1_pp1_tp2_mbs1_gbs1024_vpNone_none/slurm-442544.log",
            # 16:   "path/to/slurm-G.log",
            # 128:   "path/to/slurm-H.log",
            512: "output/qwen3-8b/qwen3_8b_128_ep1_pp1_tp2_mbs1_gbs1024_vpNone_none/slurm-442659.log",
            1024: "output/qwen3-8b/qwen3_8b_256_ep1_pp1_tp2_mbs1_gbs1024_vpNone_none/slurm-442673.log",
        },
    },
    {
        "label": "TP=4, MBS=4, GBS=2048",
        "files": {
            # 4:    "output/qwen3-8b/qwen3_8b_1_ep1_pp1_tp4_mbs4_gbs2048_vpNone_none/slurm-443445.log",
            # 16:   "path/to/slurm-L.log",
            # 128:   "path/to/slurm-M.log",
            256: "output/qwen3-8b/qwen3_8b_64_ep1_pp1_tp4_mbs4_gbs2048_vpNone_none/slurm-442779.log",
            # 512: "path/to/slurm-O.log",
        },
    },
    {
        "label": "TP=2, MBS=1, GBS=2048",
        "files": {
            # 4:    "path/to/slurm-K.log",
            # 16:   "path/to/slurm-L.log",
            # 128:   "path/to/slurm-M.log",
            256: "output/qwen3-8b/qwen3_8b_64_ep1_pp1_tp2_mbs1_gbs2048_vpNone_none/slurm-443037.log",
            # 512: "path/to/slurm-O.log",
        },
    },
]

OUTPUT_FILE = "qwen3_8b_jupiter_scaling.png"
PLOT_TITLE = "Strong scaling efficiency Qwen 3 8B Jupiter (max 64GB VRAM) "


_TFLOPS_RE = re.compile(r"wandb:\s+TFLOPS\s+([\d.]+)")
_TOK_GPU_RE = re.compile(r"wandb:\s+Tokens per second per GPU\s+([\d.]+)")
_BS_RE = re.compile(r"wandb:\s+batch-size\s+(\d+)")
_ITER_TIME_RE = re.compile(r"wandb:\s+iteration-time\s+([\d.]+)")
_WORLD_SIZE_RE = re.compile(r"using world size:\s*(\d+)")


def parse_log(path: str) -> dict:
    """Return final metrics from the wandb Run summary in a Megatron-LM SLURM
    log."""
    text = Path(path).read_text(errors="replace")

    def _require(pattern, name):
        m = pattern.search(text)
        if not m:
            raise ValueError(f"Could not find '{name}' in {path}")
        return m.group(1)

    tflops_per_gpu = float(_require(_TFLOPS_RE, "TFLOPS"))
    tok_per_s_per_gpu = float(_require(_TOK_GPU_RE, "Tokens per second per GPU"))
    global_bs = int(_require(_BS_RE, "batch-size"))
    s_per_step = float(_require(_ITER_TIME_RE, "iteration-time"))

    world_size_m = _WORLD_SIZE_RE.search(text)
    world_size = int(world_size_m.group(1)) if world_size_m else None

    tokens_per_step = tok_per_s_per_gpu * s_per_step * world_size if world_size else None

    return {
        "world_size": world_size,
        "global_bs": global_bs,
        "tokens_per_step": tokens_per_step,
        "s_per_step": s_per_step,
        "tflops_per_gpu": tflops_per_gpu,
        "tok_per_s_per_gpu": tok_per_s_per_gpu,
        "tok_per_s": tok_per_s_per_gpu * world_size if world_size else None,
    }


def main():
    # Union of all GPU counts across series → shared x-axis positions.
    all_gpu_counts = sorted({n_gpus for series in EXPERIMENT_SERIES for n_gpus in series["files"]})
    x_map = {n: i for i, n in enumerate(all_gpu_counts)}

    colors = ["#2874A6", "#E67E22", "#27AE60", "#8E44AD", "#C0392B"]
    markers = ["o", "s", "^", "D", "v"]

    parsed_series = []
    for s_idx, series in enumerate(EXPERIMENT_SERIES):
        gpu_counts = sorted(series["files"].keys())
        records = {}
        for n_gpus in gpu_counts:
            log_path = series["files"][n_gpus]
            print(f"[{series['label']}] Parsing {log_path} ({n_gpus} GPUs) …")
            records[n_gpus] = parse_log(log_path)

        header = (
            f"  {'Nodes':>6}  {'GPUs':>5}  {'Global BS':>10}  "
            f"{'s/step':>7}  {'TFLOPs/s/GPU':>13}  {'Tok/s/GPU':>11}"
        )
        sep = "-" * len(header)
        print(f"\n{series['label']}")
        print(sep)
        print(header)
        print(sep)

        rows = []
        for n_gpus in gpu_counts:
            r = records[n_gpus]
            nodes = n_gpus // GPUS_PER_NODE
            rows.append(
                {
                    "n_gpus": n_gpus,
                    "nodes": nodes,
                    "x_pos": x_map[n_gpus],
                    **r,
                }
            )
            print(
                f"  {nodes:>6}  {n_gpus:>5}  {r['global_bs']:>10}  "
                f"{r['s_per_step']:>7.3f}  {r['tflops_per_gpu']:>13.1f}  "
                f"{r['tok_per_s_per_gpu']:>11.1f}"
            )
        print(sep)

        parsed_series.append(
            {
                "label": series["label"],
                "color": colors[s_idx % len(colors)],
                "marker": markers[s_idx % len(markers)],
                "rows": rows,
            }
        )

    # ---- Plot ---------------------------------------------------------------
    n_series = len(parsed_series)
    bar_width = 0.6 / n_series
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * bar_width

    fig, ax = plt.subplots(figsize=(14, 6))
    axr = ax.twinx()

    all_tok_gpu = []
    all_tflops = []

    for i, ps in enumerate(parsed_series):
        rows = ps["rows"]
        x_base = np.array([r["x_pos"] for r in rows])
        tok_gpu = np.array([r["tok_per_s_per_gpu"] for r in rows])
        tfl = np.array([r["tflops_per_gpu"] for r in rows])

        all_tok_gpu.extend(tok_gpu.tolist())
        all_tflops.extend(tfl.tolist())

        bars = ax.bar(
            x_base + offsets[i],
            tfl,
            width=bar_width * 0.92,
            color=ps["color"],
            alpha=0.55,
            edgecolor=ps["color"],
            linewidth=0.8,
            label=ps["label"],
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7, color=ps["color"])

        axr.plot(
            x_base,
            tok_gpu,
            marker=ps["marker"],
            color=ps["color"],
            linewidth=2,
        )

    x_ticks = list(range(len(all_gpu_counts)))
    x_labels = [str(n) for n in all_gpu_counts]

    ax.set_xlabel("Number of GPUs", fontsize=11)
    ax.set_ylabel("TFLOPs/s / GPU", fontsize=11)
    ax.set_title(PLOT_TITLE, fontsize=11, fontweight="bold")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(all_tflops) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)

    axr.set_ylabel("Tokens / s / GPU", fontsize=11)
    axr.set_ylim(0, max(all_tok_gpu) * 1.3)
    axr.spines[["top", "left"]].set_visible(False)

    if VRAM_GB is not None:
        fig.text(
            0.5,
            -0.03,
            f"* VRAM manually constrained to {VRAM_GB} GB per GPU",
            ha="center",
            fontsize=9,
            color="gray",
            style="italic",
        )

    if OUTPUT_FILE:
        fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {OUTPUT_FILE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
