"""
Scaling efficiency analysis from Megatron-LM SLURM log files.

Fill in GPUS_PER_NODE and EXPERIMENTS below, then run:
    python plot_scaling.py
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from pathlib import Path


GPUS_PER_NODE = 4

# Map number of GPUs to the corresponding SLURM log file path.
# EXPERIMENTS = {
#     256: "output/megatron_qwen3_moe_235BA22B_jupiter/meg_moe_30BA3B_n64_ep16_pp8_tp2_mbs1_gbs4096_vp4_hybridep_fp8_overlap/slurm-357815.log",
#     512: "output/megatron_qwen3_moe_235BA22B_jupiter/meg_moe_30BA3B_n128_ep16_pp8_tp2_mbs1_gbs8192_vp4_hybridep_fp8_overlap/slurm-357817.log",
#     1024: "output/megatron_qwen3_moe_235BA22B_jupiter/meg_moe_30BA3B_n256_ep16_pp8_tp2_mbs1_gbs16384_vp4_hybridep_fp8_overlap/slurm-357780.log",
#     2048: "output/megatron_qwen3_moe_235BA22B_jupiter/meg_moe_30BA3B_n512_ep16_pp8_tp2_mbs1_gbs32768_vp4_hybridep_fp8_overlap/slurm-357928.log",
# }
# GAS=128
EXPERIMENTS = {
    256: "../output/megatron_qwen3_moe_235BA22B_jupiter/meg_moe_30BA3B_n64_ep16_pp8_tp2_mbs1_gbs2048_vp4_hybridep_fp8_overlap/slurm-357972.log",
    512: "../output/megatron_qwen3_moe_235BA22B_jupiter/meg_moe_30BA3B_n128_ep16_pp8_tp2_mbs1_gbs4096_vp4_hybridep_fp8_overlap/slurm-358028.log",
    1024: "../output/megatron_qwen3_moe_235BA22B_jupiter/meg_moe_30BA3B_n256_ep16_pp8_tp2_mbs1_gbs8192_vp4_hybridep_fp8_overlap/slurm-358035.log",
    2048: "../output/megatron_qwen3_moe_235BA22B_jupiter/meg_moe_30BA3B_n512_ep16_pp8_tp2_mbs1_gbs16384_vp4_hybridep_fp8_overlap/slurm-357992.log",
}

# Output filename for the combined figure (None = show interactively).
OUTPUT_FILE = "qwen3_235B-A22B_jupiter_scaling.png"

# Title for the top (token throughput) bar chart.
PLOT_TITLE = "Token Throughput Megatron Qwen 3 235B A22B (TP 2, PP 8, EP 16, VP 4, GAS 128, MBS 1)"

_TFLOPS_RE    = re.compile(r"wandb:\s+TFLOPS\s+([\d.]+)")
_TOK_GPU_RE   = re.compile(r"wandb:\s+Tokens per second per GPU\s+([\d.]+)")
_BS_RE        = re.compile(r"wandb:\s+batch-size\s+(\d+)")
_ITER_TIME_RE = re.compile(r"wandb:\s+iteration-time\s+([\d.]+)")

_WORLD_SIZE_RE = re.compile(r"using world size:\s*(\d+)")


def parse_log(path: str) -> dict:
    """Return final metrics from the wandb Run summary in a Megatron-LM SLURM log."""
    text = Path(path).read_text(errors="replace")

    def _require(pattern, name):
        m = pattern.search(text)
        if not m:
            raise ValueError(f"Could not find '{name}' in {path}")
        return m.group(1)

    tflops_per_gpu   = float(_require(_TFLOPS_RE,    "TFLOPS"))
    tok_per_s_per_gpu = float(_require(_TOK_GPU_RE,  "Tokens per second per GPU"))
    global_bs        = int(_require(_BS_RE,           "batch-size"))
    s_per_step       = float(_require(_ITER_TIME_RE, "iteration-time"))

    world_size_m = _WORLD_SIZE_RE.search(text)
    world_size   = int(world_size_m.group(1)) if world_size_m else None

    # tokens/step = tok/s/GPU * iter_time_s * world_size
    tokens_per_step = (
        tok_per_s_per_gpu * s_per_step * world_size if world_size else None
    )

    return {
        "world_size":        world_size,
        "global_bs":         global_bs,
        "tokens_per_step":   tokens_per_step,
        "s_per_step":        s_per_step,
        "tflops_per_gpu":    tflops_per_gpu,
        "tok_per_s_per_gpu": tok_per_s_per_gpu,
        "tok_per_s":         tok_per_s_per_gpu * world_size if world_size else None,
    }

def main():
    gpu_counts = sorted(EXPERIMENTS.keys())
    records = {}
    for n_gpus in gpu_counts:
        log_path = EXPERIMENTS[n_gpus]
        print(f"Parsing {log_path} ({n_gpus} GPUs) …")
        rec = parse_log(log_path)
        records[n_gpus] = rec


    # Baseline for efficiency: smallest GPU count
    baseline_gpus  = gpu_counts[0]
    baseline_tok_s = records[baseline_gpus]["tok_per_s"]

    header = (
        f"{'Nodes':>6}  {'GPUs':>5}  {'Global BS':>10}  "
        f"{'Tokens/step':>10}  {'s/step':>7}  "
        f"{'TFLOPs/s/GPU':>13}  {'Tokens/s':>12}  {'Efficiency':>11}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    table_rows = []
    for n_gpus in gpu_counts:
        r   = records[n_gpus]
        nodes = n_gpus // GPUS_PER_NODE
        optimal_tok_s = baseline_tok_s * (n_gpus / baseline_gpus)
        efficiency    = (r["tok_per_s"] / optimal_tok_s * 100.0) if optimal_tok_s else float("nan")

        table_rows.append({
            "n_gpus": n_gpus,
            "nodes": nodes,
            "optimal_tok_s": optimal_tok_s,
            "efficiency": efficiency,
            **r,
        })

        tok_step_str = f"{r['tokens_per_step']:>10.0f}" if r['tokens_per_step'] else f"{'N/A':>10}"
        print(
            f"{nodes:>6}  {n_gpus:>5}  {r['global_bs']:>10}  "
            f"{tok_step_str}  {r['s_per_step']:>7.3f}  "
            f"{r['tflops_per_gpu']:>13.1f}  {r['tok_per_s']:>12.0f}  "
            f"{efficiency:>10.1f}%"
        )
    print(sep)

    n_gpus_arr     = np.array([tr["n_gpus"]         for tr in table_rows])
    tflops_arr     = np.array([tr["tflops_per_gpu"]  for tr in table_rows])
    tok_s_arr      = np.array([tr["tok_per_s"]       for tr in table_rows])
    optimal_arr    = np.array([tr["optimal_tok_s"]   for tr in table_rows])
    efficiency_arr = np.array([tr["efficiency"]      for tr in table_rows])

    x_labels = n_gpus_arr.astype(str)
    x_pos    = np.arange(len(n_gpus_arr))

    mil_fmt = FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    bars1 = ax1.bar(x_pos, tok_s_arr, color="#AED6F1", edgecolor="white", width=0.6,
                    label="_nolegend_")
    ax1.bar_label(bars1, labels=[f"{v/1e6:.2f}M" for v in tok_s_arr],
                  padding=3, fontsize=9, color="black")
    ax1.plot(x_pos, tok_s_arr,   marker="o", color="#2874A6", linewidth=2,
             linestyle="-", label="Measured Tok/s")
    ax1.plot(x_pos, optimal_arr, marker="o", color="#E67E22", linewidth=2,
             linestyle=":", label="Optimal scaling")
    ax1.set_ylabel("Tokens / second", fontsize=11, color="black")
    ax1.set_title(PLOT_TITLE, fontsize=11, fontweight="bold", color="black")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(0, max(optimal_arr) * 1.25)
    ax1.yaxis.set_major_formatter(mil_fmt)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(colors="black")

    ax1r = ax1.twinx()
    ax1r.plot(x_pos, efficiency_arr, marker="s", color="#C0392B", linewidth=2,
              linestyle=":", label="Efficiency (%)")
    ax1r.set_ylabel("Efficiency (%)", fontsize=11, color="black")
    ax1r.tick_params(axis="y", labelcolor="black")
    ax1r.set_ylim(0, 115)
    ax1r.axhline(100, color="#C0392B", linewidth=0.8, linestyle="--", alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1r, labels1r = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines1r, labels1 + labels1r, loc="upper left", fontsize=9,
               bbox_to_anchor=(0.0, 0.88))

    # Bar plot
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(x_pos, tflops_arr, color="#228B22", edgecolor="white", width=0.6)
    ax2.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9, color="black")
    ax2.set_xlabel("Number of GPUs", fontsize=11, color="black")
    ax2.set_ylabel("TFLOPs/s / GPU", fontsize=11, color="black")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylim(0, max(tflops_arr) * 1.2)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(colors="black")

    if OUTPUT_FILE:
        fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {OUTPUT_FILE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
