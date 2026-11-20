import tempfile
import unittest
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import training
from training import fork_validation_cursor


class ForkValidationCursor(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.load = root / "arm/training_ckpts/checkpoints"
        self.load.mkdir(parents=True)
        seed = root / "original/iter_0416000"
        seed.mkdir(parents=True)
        self.link = self.load / seed.name
        self.link.symlink_to(seed, target_is_directory=True)
        self.recipe = dict(seed=str(seed), seed_iteration=416000, endpoint=417193)
        self.args = SimpleNamespace(load=str(self.load), iteration=416000,
                                    consumed_valid_samples=1679360, train_samples=417193 * 4096,
                                    consumed_train_samples=416000 * 4096, start_eval_at_iter=416000,
                                    finetune=False, override_ckpt_iteration=None)

    def test_only_validation_cursor_changes_on_seed_restore(self):
        before = vars(self.args).copy()
        self.assertEqual(fork_validation_cursor(self.args, self.recipe), 1679360)
        self.assertEqual(vars(self.args), dict(before, consumed_valid_samples=0))

    def test_resume_preserves_arm_cursor(self):
        self.args.iteration = 416125
        self.args.consumed_valid_samples = 16
        self.assertIsNone(fork_validation_cursor(self.args, self.recipe))
        self.assertEqual(self.args.consumed_valid_samples, 16)

    def test_unknown_seed_or_missing_restore_fails_closed(self):
        self.recipe["seed"] = str(self.load / "other")
        with self.assertRaisesRegex(ValueError, "seed link"):
            fork_validation_cursor(self.args, self.recipe)
        self.args.iteration = 0
        with self.assertRaisesRegex(ValueError, "iteration"):
            fork_validation_cursor(self.args, self.recipe)

    def test_native_restore_runs_before_cursor_reset_and_returns_original_state(self):
        (self.load.parents[1] / "recipe.json").write_text(json.dumps(self.recipe))
        marker = object()
        self.args.consumed_valid_samples = 0

        def setup():
            self.args.consumed_valid_samples = 1679360
            return marker

        events = []
        native = SimpleNamespace(setup_model_and_optimizer=setup,
                                 get_args=lambda: self.args, print_rank_0=events.append)

        def run(*args, **kwargs):
            self.assertIs(native.setup_model_and_optimizer(), marker)
            self.assertEqual(self.args.consumed_valid_samples, 0)

        with (
            patch.object(sys, "path", list(sys.path)),
            patch.object(training.importlib, "import_module", return_value=native),
            patch.object(training.runpy, "run_path", side_effect=run),
        ):
            training.main()
        self.assertIs(native.setup_model_and_optimizer, setup)
        self.assertEqual(events, ["Annealing fork validation cursor at iteration 416000: 1679360 -> 0"])


if __name__ == "__main__":
    unittest.main()
