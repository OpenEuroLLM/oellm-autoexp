import csv
import importlib.util
import json
import os
import sqlite3
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from oellm_autoexp.profiling.adapters.nsys import NsysAdapter
from oellm_autoexp.profiling.adapters.rocprofv3 import RocprofV3Adapter
from oellm_autoexp.profiling.analysis import (
    IntervalUnion,
    analyze,
    interval_intersection_ns,
    parse_iteration_log,
    select_steady_window,
)
from oellm_autoexp.profiling.capture import wrap_launch_command
from oellm_autoexp.profiling.config import ProfilingConfig
from oellm_autoexp.profiling.models import ProfilingError
from oellm_autoexp.profiling.reporting import render_canvas, write_reports
from oellm_autoexp.profiling.taxonomy import classify_kernel


KERNEL_FIELDS = [
    "Kind",
    "Agent_Id",
    "Queue_Id",
    "Thread_Id",
    "Dispatch_Id",
    "Kernel_Id",
    "Kernel_Name",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
]


def _write_kernel_trace(path: Path, rows: list[tuple[str, int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KERNEL_FIELDS)
        writer.writeheader()
        for index, (name, start, end) in enumerate(rows):
            writer.writerow(
                {
                    "Kind": "KERNEL_DISPATCH",
                    "Agent_Id": 1,
                    "Queue_Id": 2,
                    "Thread_Id": 3,
                    "Dispatch_Id": index,
                    "Kernel_Id": index,
                    "Kernel_Name": name,
                    "Correlation_Id": index,
                    "Start_Timestamp": start,
                    "End_Timestamp": end,
                }
            )


def _iteration_line(iteration: int, timestamp: datetime, elapsed: float, tflops: float) -> str:
    return (
        f"0: [{timestamp.isoformat(sep=' ')}] iteration {iteration:8d}/      20 | "
        f"elapsed time per iteration (ms): {elapsed:.1f} | "
        f"throughput per GPU (TFLOP/s/GPU): {tflops:.1f} | "
        f"Tokens per second per GPU (Tok/s/GPU): {tflops * 50:.1f} |\n"
    )


def _load_run_profile():
    path = Path(__file__).parents[3] / "scripts" / "profiling" / "run_profile.py"
    spec = importlib.util.spec_from_file_location("run_profile_test_module", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_autoexp():
    path = Path(__file__).parents[3] / "scripts" / "run_autoexp.py"
    spec = importlib.util.spec_from_file_location("run_autoexp_test_module", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TaxonomyAndIntervalTests(unittest.TestCase):
    def test_taxonomy(self) -> None:
        cases = {
            "ncclDevKernel_Generic_4": "communication",
            "_sort_chunks_by_idxs_kernel": "routing_permutation",
            "chunk_gated_delta_rule_fwd_kernel": "gated_delta_net",
            "void ck_tile::FmhaBwdKernel": "attention",
            "Cijk_Ailk_Bljk": "gemm_ambiguous",
            "vectorized_elementwise_kernel": "elementwise",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(classify_kernel(name), expected)

    def test_interval_union_and_intersection(self) -> None:
        left = IntervalUnion()
        right = IntervalUnion()
        left.add(10, 20)
        left.add(0, 15)  # Exercise out-of-order fallback.
        right.add(5, 12)
        right.add(30, 40)
        self.assertEqual(left.duration_ns(), 20)
        self.assertEqual(right.duration_ns(), 17)
        self.assertEqual(interval_intersection_ns(left, right), 7)


class AdapterTests(unittest.TestCase):
    def test_rocprof_selects_largest_rank_zero_and_normalizes(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            _write_kernel_trace(directory / "rank0_pid1_kernel_trace.csv", [("small", 0, 1)])
            _write_kernel_trace(
                directory / "rank0_pid2_kernel_trace.csv",
                [("Cijk_kernel", 0, 10), ("nccl_kernel", 5, 15)],
            )
            _write_kernel_trace(
                directory / "rank1_pid3_kernel_trace.csv", [("worker", 0, 100)] * 3
            )
            adapter = RocprofV3Adapter()
            artifacts = adapter.discover(directory)
            events = list(adapter.iter_kernels(artifacts))

        self.assertEqual(artifacts.stem, "rank0_pid2")
        self.assertEqual(events[0].duration_ns, 10)
        self.assertEqual(events[0].stream_id, "2")

    def test_nsys_sqlite_string_join(self) -> None:
        with TemporaryDirectory() as raw:
            database = Path(raw) / "rank0.sqlite"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
                "(start INTEGER, end INTEGER, demangledName INTEGER, "
                "deviceId INTEGER, streamId INTEGER, correlationId INTEGER)"
            )
            connection.execute("INSERT INTO StringIds VALUES (1, 'ncclKernel')")
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
                "VALUES (100, 150, 1, 0, 7, 9)"
            )
            connection.commit()
            connection.close()

            adapter = NsysAdapter()
            artifacts = adapter.discover(database)
            events = list(adapter.iter_kernels(artifacts))
            summary = analyze(adapter, artifacts)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "ncclKernel")
        self.assertEqual(events[0].duration_ns, 50)
        self.assertEqual(summary["provider"], "nsys")
        self.assertEqual(summary["categories"][0]["category"], "communication")

    def test_nsys_missing_kernel_table_is_clear_error(self) -> None:
        with TemporaryDirectory() as raw:
            database = Path(raw) / "empty.sqlite"
            sqlite3.connect(database).close()
            adapter = NsysAdapter()
            with self.assertRaisesRegex(ProfilingError, "no recognized CUDA kernel table"):
                list(adapter.iter_kernels(adapter.discover(database)))


class AnalysisAndReportingTests(unittest.TestCase):
    def test_analysis_log_baseline_and_reports(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            trace = directory / "rank0_pid42_kernel_trace.csv"
            _write_kernel_trace(
                trace,
                [
                    ("ncclDevKernel_Generic_4", 0, 10_000_000),
                    ("Cijk_Ailk_Bljk", 5_000_000, 15_000_000),
                    ("_permute_kernel", 15_000_000, 17_000_000),
                ],
            )
            start = datetime(2026, 9, 25, 10, 0, 0)
            profile_log = directory / "profile.log"
            baseline_log = directory / "baseline.log"
            profile_log.write_text(
                "".join(
                    _iteration_line(i, start + timedelta(milliseconds=i * 5), 5, 10)
                    for i in range(1, 5)
                ),
                encoding="utf-8",
            )
            baseline_log.write_text(
                "".join(
                    _iteration_line(i, start + timedelta(milliseconds=i * 4), 4, 12)
                    for i in range(1, 5)
                ),
                encoding="utf-8",
            )
            adapter = RocprofV3Adapter()
            artifacts = adapter.discover(trace)
            profile_records = parse_iteration_log(profile_log)
            window = select_steady_window(profile_records, 2, 4)
            summary = analyze(
                adapter,
                artifacts,
                steady_window=window,
                baseline_records=parse_iteration_log(baseline_log),
            )
            canvas = render_canvas(summary)
            output = directory / "out"
            write_reports(summary, output, directory / "report.canvas.tsx")
            payload = json.loads((output / "summary.json").read_text())

        categories = {row["category"]: row for row in payload["categories"]}
        self.assertAlmostEqual(categories["communication"]["total_ms"], 8.0)
        self.assertAlmostEqual(payload["overlap"]["communication_overlap_pct"], 62.5)
        self.assertAlmostEqual(payload["baseline"]["profile_throughput_delta_pct"], -100 / 6)
        self.assertIn('from "cursor/canvas"', canvas)
        self.assertTrue((output / "summary.json").name == "summary.json")


class CaptureTests(unittest.TestCase):
    def test_none_does_not_wrap(self) -> None:
        self.assertEqual(
            wrap_launch_command(ProfilingConfig(provider="none"), "python train.py"),
            "python train.py",
        )

    def test_rocprof_wrap_contains_generic_launcher(self) -> None:
        config = ProfilingConfig(
            provider="rocprofv3",
            output_dir="/tmp/profile",
            ranks=[0, 2],
            provider_options={"output_formats": ["csv"]},
        )
        command = wrap_launch_command(config, "python train.py --x 1")
        self.assertIn("scripts/profiling/run_profile.py", command)
        self.assertIn("--provider rocprofv3", command)
        self.assertIn("--ranks 0,2", command)
        self.assertIn("--analysis-options", command)
        self.assertTrue(command.endswith("-- python train.py --x 1"))

    def test_provider_commands(self) -> None:
        module = _load_run_profile()
        rocprof, roc_expected = module._rocprof_command(
            Path("/tmp/out"),
            0,
            {"kernel", "communication"},
            {"output_formats": ["csv"], "stats": True},
            ["python", "train.py"],
        )
        nsys, nsys_expected = module._nsys_command(
            Path("/tmp/out"),
            1,
            {"kernel", "markers"},
            {"export": "sqlite"},
            ["python", "train.py"],
        )
        self.assertIn("--kernel-trace", rocprof)
        self.assertIn("--rccl-trace", rocprof)
        self.assertIn("rank0_pid*_kernel_trace.csv", roc_expected)
        self.assertIn("--trace=cuda,nvtx", nsys)
        self.assertIn("rank1.sqlite", nsys_expected)

    def test_unselected_rank_passes_through(self) -> None:
        module = _load_run_profile()
        with mock.patch.dict(os.environ, {"SLURM_PROCID": "3"}, clear=False):
            with mock.patch.object(module.os, "execvp", side_effect=RuntimeError("exec")) as execvp:
                with self.assertRaisesRegex(RuntimeError, "exec"):
                    module.main(
                        [
                            "--provider",
                            "rocprofv3",
                            "--ranks",
                            "0",
                            "--output-dir",
                            "/tmp/profile",
                            "--",
                            "python",
                            "train.py",
                        ]
                    )
        execvp.assert_called_once_with("python", ["python", "train.py"])

    def test_profiling_choice_expands_to_root_overrides(self) -> None:
        module = _load_run_autoexp()
        config_dir = Path(__file__).parents[3] / "config"
        overrides = module._normalize_profiling_overrides(
            ["profiling=rocprofv3", "profiling.ranks=[2]"], config_dir
        )
        self.assertIn("++profiling.provider=rocprofv3", overrides)
        self.assertIn("++profiling.capture.kernel=true", overrides)
        self.assertEqual(overrides[-1], "profiling.ranks=[2]")


if __name__ == "__main__":
    unittest.main()
