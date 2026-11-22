import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("bsc_b1_annealing.py")
spec = importlib.util.spec_from_file_location("bsc_b1_annealing", SCRIPT)
bsc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsc)


class BSCAnnealing(unittest.TestCase):
    def test_remap_preserves_weights_and_order(self):
        rows = "0.1 /jsc/flag/a\n0.9 /jsc/flag/b\n"
        self.assertEqual(
            bsc.remap(rows, "/jsc/flag", {"a": "/bsc/z", "b": "/bsc/y"}),
            "0.1 /bsc/z\n0.9 /bsc/y\n",
        )

    def test_remap_accepts_two_explicit_flag_roots(self):
        self.assertEqual(
            bsc.remap("0.1 /jsc/flag/a 0.9 /jsc/data1/flag/b",
                      ["/jsc/flag", "/jsc/data1/flag"],
                      {"a": "/bsc/z", "b": "/bsc/y"}),
            "0.1 /bsc/z\n0.9 /bsc/y\n",
        )

    def test_missing_or_duplicate_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            bsc.remap("1 /jsc/flag/a", "/jsc/flag", {})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bsc.remap(
                "0.1 /jsc/flag/a 0.9 /jsc/flag/b",
                "/jsc/flag",
                {"a": "/bsc/z", "b": "/bsc/z"},
            )

    def test_normal_qos_and_cap_are_enforced(self):
        self.assertEqual(bsc.smoke_qos(64), "acc_ehpc")
        self.assertEqual(bsc.smoke_qos(32), "acc_ehpc")
        for nodes in (0, 16, 100, 128, 256, 512):
            with self.assertRaises(ValueError):
                bsc.smoke_qos(nodes)

    def test_render_binds_only_reviewed_path_arguments(self):
        args = ["--load", "/e/old", "--eval-iters", "1", "--wandb-project", "old"]
        self.assertEqual(
            bsc.bind_arguments(args, {"--load": "/gpfs/staged"}, {"--wandb-project"}),
            ["--load", "/gpfs/staged", "--eval-iters", "1"],
        )
        with self.assertRaisesRegex(ValueError, "unbound"):
            bsc.bind_arguments(args, {}, {"--wandb-project"})

    def test_bsc_smoke_disables_only_tensor_parallel_communication_overlap(self):
        args = ["--load", "/e/old", "--tp-comm-overlap", "--overlap-grad-reduce",
                "--overlap-param-gather", "--sequence-parallel"]
        self.assertEqual(
            bsc.bind_arguments(args, {"--load": "/gpfs/seed"}, set(), {"--tp-comm-overlap"}),
            ["--load", "/gpfs/seed", "--overlap-grad-reduce",
             "--overlap-param-gather", "--sequence-parallel"],
        )

    def test_large_node_admission_is_test_only(self):
        run = Path("/gpfs/arm")
        with patch.object(bsc, "read", return_value={}), patch.object(bsc, "check_pins"):
            with patch.object(bsc.subprocess, "run") as command:
                bsc.admission(run, 256)
                command.assert_called_once_with(
                    ["sbatch", "--test-only", "--nodes=256", "--qos=acc_xehpc",
                     str(run / "smoke.sbatch")], check=True)
                with self.assertRaises(ValueError):
                    bsc.admission(run, 512)

    def test_cache_reuse_rejects_changed_sample_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            run, prior = (Path(directory) / name for name in ("next", "prior"))
            for arm, valid in ((run, 32), (prior, 16)):
                arm.mkdir()
                (arm / "cache").mkdir()
                bsc.write(arm / "recipe.json", {})
                bsc.write(arm / "cache-settings.json", {"target": {}, "samples": [12288, valid, 16]})
            with patch.object(bsc, "check_pins"):
                with self.assertRaisesRegex(ValueError, "settings"):
                    bsc.reuse_cache(run, prior)
            self.assertFalse((run / "cache-submitted.json").exists())
            self.assertFalse((run / "cache").is_symlink())

    def test_smoke_script_caps_resources_and_uses_runtime_wrapper(self):
        text = bsc.batch(Path("/gpfs/run"), Path("/gpfs/source"), Path("/gpfs/image"), "smoke", 64)
        self.assertIn("#SBATCH --time=00:10:00", text)
        self.assertIn("#SBATCH --nodes=64", text)
        self.assertIn("#SBATCH --account=ehpc533", text)
        self.assertIn("#SBATCH --qos=acc_ehpc", text)
        self.assertIn("flock -n 9", text)
        self.assertIn("srun --kill-on-bad-exit=1", text)

    def test_probe_pending_prevents_cache_submissions(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "probe.job").write_text("123\n")
            with patch.object(bsc, "accounting", return_value={"job": "123", "state": "PENDING"}):
                with patch.object(bsc, "submit") as submit:
                    self.assertEqual(bsc.advance(runtime)["status"], "waiting")
                    submit.assert_not_called()

    def test_one_debug_slot_and_missing_seed_gate_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "probe.job").write_text("123\n")
            (runtime / "probe-123.log").write_text('{"BSC_B1_RUNTIME": "passed"}\n')
            first, second = (runtime / "arms" / name for name in ("dominant1", "dominant2"))
            first.mkdir(parents=True)
            second.mkdir()
            bsc.write(first / "recipe.json", {"seed": str(runtime / "unavailable")})
            accounting = {"job": "123", "state": "COMPLETED", "exit_code": "0:0"}
            with patch.object(bsc, "accounting", return_value=accounting):
                with patch.object(bsc, "progress", side_effect=[
                        {"cache": {"state": "COMPLETED"}}, {}, {"cache": {"state": "PENDING"}}]):
                    with patch.object(bsc, "result", return_value={"cache_coverage": {"train": {"prefixes": 294}}}):
                        with patch.object(bsc, "submit") as submit:
                            status = bsc.advance(runtime)
                            self.assertEqual(status["status"], "waiting")
                            self.assertTrue(status["arms"]["dominant1"]["cache"]["coverage_verified"])
                            self.assertTrue((first / "cache-proof.json").is_file())
                            submit.assert_called_once_with(second, "cache")

    def test_audit_checks_pair_and_reports_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.idx").write_bytes(b"index")
            (root / "a.bin").write_bytes(b"data")
            facts = bsc.probe(root / "a")
            expected = {"a": facts, "b": [4, 100, "other"]}
            result = bsc.audit(expected, [root])
            self.assertEqual(result["matches"], {"a": str(root / "a")})
            self.assertEqual(result["missing_bytes"], 104)
            self.assertEqual(result["prefixes"]["b"]["different"], {})


if __name__ == "__main__":
    unittest.main()
