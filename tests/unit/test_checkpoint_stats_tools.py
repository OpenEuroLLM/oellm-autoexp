from __future__ import annotations

import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scanner = load_script("scan_checkpoint_stats")
comparison = load_script("compare_checkpoint_stats")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_row(iteration: int, rms: float) -> dict[str, object]:
    return {
        "run": "ignored",
        "checkpoint": f"iter_{iteration:07d}",
        "iteration": iteration,
        "kind": "model",
        "state": "weight",
        "tensor": "decoder.output_layer.weight",
        "storage_key": "decoder.output_layer.weight",
        "layer": -1,
        "shape": "4x2",
        "dtype": "torch.float32",
        "numel": 8,
        "sample_numel": 8,
        "comparison_sample_numel": 8,
        "finite_frac": 1.0,
        "zero_frac": 0.0,
        "negative_frac": 0.5,
        "min": -rms,
        "max": rms,
        "mean": 0.0,
        "std": rms,
        "rms": rms,
        "abs_mean": rms,
        "abs_max": rms * 2,
        "abs_q001": rms,
        "abs_q01": rms,
        "abs_q05": rms,
        "abs_q50": rms,
        "abs_q95": rms,
        "abs_q99": rms,
        "abs_q999": rms * 1.5,
        "rms_vs_baseline": rms,
        "abs_max_vs_baseline": rms,
        "sample_delta_rms": max(rms - 1, 0),
        "sample_relative_delta_rms": max(rms - 1, 0),
        "sample_cosine": 1.0,
        "sample_sign_flip_frac": 0.0,
    }


def channel_row(iteration: int, median: float) -> dict[str, object]:
    return {
        "run": "ignored",
        "checkpoint": f"iter_{iteration:07d}",
        "iteration": iteration,
        "kind": "model",
        "state": "weight",
        "tensor": "decoder.output_layer.weight",
        "storage_key": "decoder.output_layer.weight",
        "layer": -1,
        "axis": "output_row",
        "metric": "rms",
        "channels": 4,
        "values_per_channel": 2,
        "finite_channel_frac": 1.0,
        "zero_channel_frac": 0.0,
        "min": median / 2,
        "q001": median / 2,
        "q01": median / 2,
        "q05": median,
        "median": median,
        "q95": median,
        "q99": median * 2,
        "q999": median * 2,
        "max": median * 2,
        "min_to_median": 0.5,
        "frac_below_dead_ratio": 0.0,
        "dead_ratio": 0.01,
    }


class CheckpointStatsToolsTest(unittest.TestCase):
    def test_sampled_elementwise_change_tracks_direction(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        baseline = torch.tensor([1.0, -2.0, 0.0, 4.0])
        current = torch.tensor([2.0, 2.0, 0.0, 4.0])

        initial, stored = scanner.sampled_elementwise_change(baseline, None, 100, torch)
        changed, _ = scanner.sampled_elementwise_change(current, stored, 100, torch)

        self.assertEqual(initial["sample_relative_delta_rms"], 0.0)
        self.assertEqual(changed["comparison_sample_numel"], 4)
        self.assertAlmostEqual(changed["sample_delta_rms"], math.sqrt(17 / 4))
        self.assertAlmostEqual(
            changed["sample_relative_delta_rms"], math.sqrt(17 / 21)
        )
        self.assertAlmostEqual(changed["sample_sign_flip_frac"], 1 / 3, places=6)
        self.assertLess(changed["sample_cosine"], 1.0)

    def test_comparison_builds_paired_effect_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run, final_rms in (("reference", 2.0), ("candidate", 3.0)):
                scan_dir = root / run
                write_rows(
                    scan_dir / "tensor_stats.csv",
                    [tensor_row(10, 1.0), tensor_row(20, final_rms)],
                )
                write_rows(
                    scan_dir / "channel_stats.csv",
                    [channel_row(10, 1.0), channel_row(20, final_rms)],
                )
                write_rows(
                    scan_dir / "skipped.csv",
                    [
                        {
                            "checkpoint": "",
                            "iteration": "",
                            "storage_key": "",
                            "reason": "",
                            "detail": "",
                        }
                    ],
                )

            output = root / "comparison"
            result = comparison.main(
                [
                    "--scan",
                    f"reference={root / 'reference'}",
                    "--scan",
                    f"candidate={root / 'candidate'}",
                    "--pair",
                    "candidate:reference",
                    "--baseline-iteration",
                    "10",
                    "--output-dir",
                    str(output),
                    "--no-plot",
                ]
            )

            self.assertEqual(result, 0)
            with (output / "paired_tensor_effects.csv").open(newline="") as handle:
                effects = list(csv.DictReader(handle))
            rms = next(row for row in effects if row["metric"] == "rms")
            self.assertAlmostEqual(float(rms["reference_change"]), 2.0)
            self.assertAlmostEqual(float(rms["candidate_change"]), 3.0)
            self.assertAlmostEqual(float(rms["effect_ratio"]), 1.5)

            with (output / "coverage.csv").open(newline="") as handle:
                coverage = list(csv.DictReader(handle))
            self.assertEqual(len(coverage), 4)
            self.assertTrue(all(row["missing_vs_run_baseline"] == "0" for row in coverage))
            self.assertTrue((output / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
