"""CPU-only tests for bounded cache-gated annealing validation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE = Path(__file__).with_name('validate.py')
if MODULE.exists():
    SPEC = importlib.util.spec_from_file_location('annealing_validate', MODULE)
    validation = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(validation)
else:
    validation = None


# Captured from the completed historical 2150282 allocation, without changing its values.
REAL_STEP = '[default3]: [2026-10-03 02:35:36.277296] iteration   117751/  120000 | consumed samples:    482308096 | elapsed time per iteration (ms): 47919.1 | mem usages: 0.7580 | throughput per GPU (TFLOP/s/GPU): 34.7 | Tokens per second per GPU (Tok/s/GPU): 171.0 | learning rate: 1.737038E-05 | global batch size:  4096 | lm loss: 1.094693E+00 | output_z_loss: 4.611339E+01 | loss scale: 1.0 | grad norm: 0.026 | num zeros: 403532608.0 | params norm: 5231.368 | number of skipped iterations:   0 | number of nan iterations:   0 |'


def log_text():
    rows = ['[default0]:  successfully loaded checkpoint from /seed/checkpoints [ t 1/4, p 1/4 ] at iteration 416000']
    for iteration, loss in [(416001, '1.10'), (416002, '1.12'), (416003, '1.09')]:
        rows.append(REAL_STEP.replace('117751/  120000', f'{iteration}/  417193').replace('1.094693E+00', loss))
    return '\n'.join(rows)


def git_at(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


def make_source_repo(repo):
    """Real committed baseline and nested Git dependency; package stays untracked."""
    repo = Path(repo)
    (repo / 'scripts').mkdir(exist_ok=True)
    (repo / 'templates').mkdir(exist_ok=True)
    (repo / 'scripts/run_autoexp.py').write_text('# baseline orchestrator\n')
    (repo / 'templates/production.sbatch').write_text('# baseline launch contract\n')
    (repo / '.gitignore').write_text('__pycache__/\n*.pyc\n.compiler-cache/\n')
    mega = repo / 'submodules/Megatron-LM'
    mega.mkdir(parents=True)
    (mega / 'training.py').write_text('# pinned numerical source\n')
    (mega / '.gitignore').write_text('__pycache__/\n*.pyc\n.compiler-cache/\n')
    subprocess.run(['git', 'init', '-q', str(mega)], check=True)
    git_at(mega, 'add', '--', 'training.py', '.gitignore')
    git_at(mega, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Pinned Megatron')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / '.gitmodules').write_text('[submodule "Megatron-LM"]\n\tpath = submodules/Megatron-LM\n\turl = ../fixture-only\n')
    git_at(repo, 'add', '--', '.gitignore', '.gitmodules', 'scripts/run_autoexp.py', 'templates/production.sbatch')
    git_at(repo, 'update-index', '--add', '--cacheinfo', '160000,' + git_at(mega, 'rev-parse', 'HEAD') + ',submodules/Megatron-LM')
    git_at(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Pinned autoexp baseline')
    return {'base_head': git_at(repo, 'rev-parse', 'HEAD'), 'megatron_head': git_at(mega, 'rev-parse', 'HEAD')}


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(validation, 'Standalone validation coordinator is not implemented')

    def test_real_historical_log_parser_preserves_measured_values(self):
        row = validation.parse_step(REAL_STEP)
        self.assertEqual(row['iteration'], 117751)
        self.assertAlmostEqual(row['lm_loss'], 1.094693)
        self.assertAlmostEqual(row['iteration_ms'], 47919.1)
        self.assertAlmostEqual(row['throughput_tflops_per_gpu'], 34.7)
        self.assertAlmostEqual(row['grad_norm'], 0.026)

    def test_three_consecutive_steps_allow_nonmonotonic_loss_and_report_deltas(self):
        report = validation.validate_log(log_text())
        self.assertEqual([s['iteration'] for s in report['steps']], [416001, 416002, 416003])
        self.assertAlmostEqual(report['steps'][1]['loss_delta'], 0.02)
        self.assertAlmostEqual(report['steps'][2]['loss_rate_per_update'], -0.03)

    def test_missing_restore_or_consecutive_update_is_rejected(self):
        for text in [log_text().replace('iteration 416000', 'iteration 415000'), log_text().replace('416002/', '416004/')]:
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                validation.validate_log(text)

    def test_duplicate_or_missing_step_is_rejected(self):
        for text in [log_text() + '\n' + log_text().splitlines()[-1], '\n'.join(log_text().splitlines()[:-1])]:
            with self.subTest(text=text[-50:]), self.assertRaises(ValueError):
                validation.validate_log(text)

    def test_nonfinite_bad_loss_and_zero_runtime_metrics_are_rejected(self):
        replacements = [('lm loss: 1.10', 'lm loss: nan'), ('lm loss: 1.10', 'lm loss: 100'), ('grad norm: 0.026', 'grad norm: inf'), ('(ms): 47919.1', '(ms): 0'), ('(TFLOP/s/GPU): 34.7', '(TFLOP/s/GPU): 0'), ('learning rate: 1.737038E-05', 'learning rate: 0'), ('skipped iterations:   0', 'skipped iterations:   1'), ('nan iterations:   0', 'nan iterations:   1')]
        for old, new in replacements:
            with self.subTest(new=new), self.assertRaises(ValueError):
                validation.validate_log(log_text().replace(old, new))

    def test_unknown_training_row_format_is_rejected(self):
        with self.assertRaises(ValueError):
            validation.validate_log(log_text().replace('grad norm:', 'unrecognized gradient:'))

    def test_accounting_uses_allocated_gpu_time_and_rejects_conflicting_rows(self):
        text = '101|COMPLETED|30|billing=128,cpu=4096,gres/gpu=128,node=32|0:0|10\n'
        rows = validation.parse_accounting(text, ['101'])
        self.assertAlmostEqual(rows['101']['gpu_hours'], 1.0666666666666667)
        with self.assertRaises(ValueError):
            validation.parse_accounting(text + text.replace('|30|', '|60|'), ['101'])
        with self.assertRaises(ValueError):
            validation.parse_accounting('', ['101'])


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(validation, 'Standalone validation coordinator is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / 'repo'
        package = self.repo / 'annealing/b1_prelude'
        package.mkdir(parents=True)
        for name in ['prepare.py', 'cache.py', 'validate.py']:
            (package / name).write_text('# frozen source\n')
        (self.repo / 'config/experiments/oellm_32b_dense').mkdir(parents=True)
        (self.repo / 'config/experiments/oellm_32b_dense/b1_prelude_annealing.yaml').write_text('model: pinned\n')
        (package / 'origins.json').write_text('{"origin": "pinned"}\n')
        self.baseline = make_source_repo(self.repo)
        self.state = self.base / 'validation'
        self.arms = []
        self.scheduler = {}
        self.submissions = []
        self.cache_checks = []
        for i, mix in enumerate(['dominant1', 'dominant2']):
            arm = self.base / ('smoke-' + mix)
            self.arms.append(arm)
            for sub in ['config', 'cache-preparation', 'training_core/logs', 'cache']:
                (arm / sub).mkdir(parents=True)
            (arm / 'config/annealing_arm.yaml').write_text('resolved: pinned\n')
            (arm / 'datamix.txt').write_text('1 /dataset\n')
            recipe = dict(seed_iteration=416000, smoke=True, nodes=32, maximum_allocations=1, allocation_walltime='00:10:00', mix=mix, source='a' * 40)
            (arm / 'recipe.json').write_text(json.dumps(recipe))
            target = dict(exit_interval=416003, eval_iters=1, eval_global_batch_size=16, global_batch_size=4096, seq_length=4096, save=None, non_persistent_save_interval=None, fp8='hybrid', fp8_recipe='tensorwise', first_last_layers_bf16=True, num_layers_at_start_in_bf16=1, num_layers_at_end_in_bf16=1, data_cache_path=str(arm / 'cache'))
            (arm / 'cache-preparation/settings.json').write_text(json.dumps(dict(target=target, samples=[12288, 16, 16])))
            job = str(101 + i)
            (arm / 'cache-preparation/submitted.json').write_text(json.dumps({'job': job}))
            (arm / 'cache-preparation/render.json').write_text('{}')
            batch = arm / 'training_core/job_20261117_000001.sbatch'
            batch.write_text('#!/bin/bash\n#SBATCH --nodes=32\n#SBATCH --gpus-per-node=4\n#SBATCH --time=00:10:00\nexec 9>' + str(arm / 'cache/.writer.lock') + '\nflock -n 9 || exit 75\necho "Launching training"\nsrun training\n')
            self.scheduler[job] = ('PENDING', 0, 0, 240)
        validation.initialize(self.state, self.arms, [], repo=self.repo)

    def runner(self, command, **kwargs):
        if command[0] == 'sacct':
            jobs = command[command.index('-j') + 1].split(',')
            lines = []
            for job in jobs:
                state, seconds, gpus, minutes = self.scheduler[job]
                lines.append(f'{job}|{state}|{seconds}|gres/gpu={gpus}|0:0|{minutes}')
            return subprocess.CompletedProcess(command, 0, '\n'.join(lines) + '\n', '')
        if command[0] == 'sbatch':
            self.submissions.append((command, kwargs))
            job = str(201 + len(self.submissions) - 1)
            self.scheduler[job] = ('PENDING', 0, 0, 10)
            return subprocess.CompletedProcess(command, 0, job + ';cluster\n', '')
        if Path(command[1]).name == 'cache.py':
            arm = Path(command[-1])
            self.cache_checks.append(arm)
            ready = dict(job='101' if arm == self.arms[0] else '102', settings_sha256=validation.sha(arm / 'cache-preparation/settings.json'), coverage={'train': {'prefixes': 294, 'requested_samples': 12288}, 'valid': {'prefixes': 294, 'requested_samples': 16}})
            (arm / 'cache-preparation/ready.json').write_text(json.dumps(ready))
            return subprocess.CompletedProcess(command, 0, 'passed\n', '')
        raise AssertionError(command)

    def test_pending_preparation_never_submits_and_reports_finite_reservations(self):
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'waiting_cache')
        self.assertEqual(self.submissions, [])
        self.assertEqual(result['budget']['preparation']['reserved_gpu_hours'], 32)
        self.assertAlmostEqual(result['budget']['smoke']['unsubmitted_forecast_gpu_hours'], 42.666666666666664)

    def test_native_template_cache_lock_descriptor_eight_is_accepted(self):
        batch = self.arms[0] / 'training_core/job_20261117_000001.sbatch'
        batch.write_text(batch.read_text().replace('exec 9>', 'exec 8>').replace('flock -n 9', 'flock -n 8'))
        validation.initialize(self.base / 'validation-eight', self.arms, [], repo=self.repo)

    def test_comment_only_writer_lock_is_rejected(self):
        batch = self.arms[0] / 'training_core/job_20261117_000001.sbatch'
        batch.write_text(batch.read_text().replace('exec 9>', '# exec 9>'))
        with self.assertRaises(ValueError):
            validation.initialize(self.base / 'validation-no-lock', self.arms, [], repo=self.repo)

    def test_duplicate_identical_accounting_is_not_double_charged(self):
        text = '101|COMPLETED|30|gres/gpu=4|0:0|240\n'
        result = validation.parse_accounting(text + text, ['101'])
        self.assertAlmostEqual(result['101']['gpu_hours'], 1 / 30)

    def test_successful_cache_gate_submits_latest_pinned_batch_only_once(self):
        self.scheduler['101'] = ('COMPLETED', 120, 4, 240)
        first = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(len(self.submissions), 1)
        command, kwargs = self.submissions[0]
        self.assertEqual(command[-1], str(self.arms[0] / 'training_core/job_20261117_000001.sbatch'))
        self.assertEqual(Path(kwargs['cwd']), self.repo)
        validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(len(self.submissions), 1)
        self.assertTrue((self.arms[0] / 'submitted.txt').is_file())

    def test_source_or_script_drift_blocks_submission(self):
        self.scheduler['101'] = ('COMPLETED', 120, 4, 240)
        (self.arms[0] / 'training_core/job_20261117_000001.sbatch').write_text('changed')
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'held')
        self.assertEqual(self.submissions, [])

    def test_dirty_baseline_template_prevents_init_and_admission(self):
        self.scheduler['101'] = ('COMPLETED', 120, 4, 240)
        (self.repo / 'templates/production.sbatch').write_text('changed launch contract\n')
        with self.assertRaises(ValueError):
            validation.initialize(self.base / 'dirty-init', self.arms, [], repo=self.repo)
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'held')
        self.assertEqual(self.submissions, [])

    def test_staged_baseline_change_is_rejected_without_clearing_the_index(self):
        source = self.repo / 'templates/production.sbatch'
        source.write_text('other owner staged this\n')
        git_at(self.repo, 'add', '--', 'templates/production.sbatch')
        source.write_text('# baseline launch contract\n')
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'held')
        self.assertEqual(git_at(self.repo, 'show', ':templates/production.sbatch'), 'other owner staged this')

    def test_dirty_megatron_and_unknown_package_files_are_rejected(self):
        mega = self.repo / 'submodules/Megatron-LM'
        (mega / 'training.py').write_text('changed numerical source\n')
        self.assertEqual(validation.cycle(self.state, runner=self.runner)['state'], 'held')
        (mega / 'training.py').write_text('# pinned numerical source\n')
        (self.repo / 'annealing/b1_prelude/shadow_import.py').write_text('unreviewed\n')
        self.assertEqual(validation.cycle(self.state, runner=self.runner)['state'], 'held')

    def test_json_origins_are_pinned_and_ignored_compiler_artifacts_are_allowed(self):
        for tree in [self.repo, self.repo / 'submodules/Megatron-LM']:
            (tree / '__pycache__').mkdir()
            (tree / '__pycache__/cached.pyc').write_bytes(b'cache')
        self.assertEqual(validation.cycle(self.state, runner=self.runner)['state'], 'waiting_cache')
        (self.repo / 'annealing/b1_prelude/origins.json').write_text('{"origin": "changed"}\n')
        self.assertEqual(validation.cycle(self.state, runner=self.runner)['state'], 'held')

    def test_clean_but_different_megatron_revision_is_rejected(self):
        mega = self.repo / 'submodules/Megatron-LM'
        (mega / 'training.py').write_text('new committed numerical source\n')
        git_at(mega, 'add', '--', 'training.py')
        git_at(mega, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Different Megatron')
        self.assertEqual(git_at(mega, 'status', '--porcelain'), '')
        self.assertEqual(validation.cycle(self.state, runner=self.runner)['state'], 'held')

    def test_changed_baseline_gitlink_is_rejected_even_with_original_dependency_checkout(self):
        mega = self.repo / 'submodules/Megatron-LM'
        (mega / 'training.py').write_text('another numerical revision\n')
        git_at(mega, 'add', '--', 'training.py')
        git_at(mega, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Another Megatron')
        other = git_at(mega, 'rev-parse', 'HEAD')
        git_at(mega, 'checkout', '--detach', self.baseline['megatron_head'])
        git_at(self.repo, 'update-index', '--cacheinfo', '160000,' + other + ',submodules/Megatron-LM')
        self.assertEqual(validation.cycle(self.state, runner=self.runner)['state'], 'held')
        git_at(self.repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Changed baseline gitlink')
        self.assertEqual(validation.cycle(self.state, runner=self.runner)['state'], 'held')

    def test_package_only_commit_preserves_the_approved_baseline(self):
        git_at(self.repo, 'add', '--', 'annealing/b1_prelude', 'config/experiments/oellm_32b_dense/b1_prelude_annealing.yaml')
        git_at(self.repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Reviewed package')
        result = validation.cycle(self.state, runner=self.runner)
        self.assertEqual(result['state'], 'waiting_cache')
        manifest = validation.read(self.state / 'validation.json')
        self.assertEqual(manifest['source_tree']['base_head'], self.baseline['base_head'])
        self.assertEqual(manifest['source_tree']['megatron_head'], self.baseline['megatron_head'])

    def test_stop_blocks_submission(self):
        self.scheduler['101'] = ('COMPLETED', 120, 4, 240)
        (self.state / 'STOP').touch()
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'stopped')
        self.assertEqual(self.submissions, [])

    def test_failed_second_preparation_prevents_new_smoke_admission(self):
        self.scheduler['101'] = ('COMPLETED', 120, 4, 240)
        self.scheduler['102'] = ('FAILED', 120, 4, 240)
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'held')
        self.assertEqual(self.submissions, [])

    def test_budget_exhaustion_blocks_submission(self):
        self.scheduler['101'] = ('COMPLETED', 30000, 4, 240)
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'held')
        self.assertEqual(self.submissions, [])
        self.assertFalse(result['budget']['preparation']['within_ceiling'])

    def test_historical_failed_cost_is_separate_from_new_budget(self):
        self.state = self.base / 'validation-history'
        validation.initialize(self.state, self.arms, [{'job': '99', 'gpus': 64}], repo=self.repo)
        self.scheduler['99'] = ('TIMEOUT', 600, 64, 10)
        result = validation.cycle(self.state, execute=False, runner=self.runner)
        self.assertAlmostEqual(result['budget']['historical']['actual_gpu_hours'], 10.666666666666666)
        self.assertEqual(result['budget']['preparation']['actual_gpu_hours'], 0)
        self.assertTrue(result['budget']['preparation']['within_ceiling'])

    def test_verify_never_submits_completed_preparations(self):
        self.scheduler['101'] = self.scheduler['102'] = ('COMPLETED', 120, 4, 240)
        with self.assertRaises(ValueError):
            validation.verify(self.state, runner=self.runner)
        self.assertEqual(self.submissions, [])

    def test_uncertain_submission_intent_is_not_retried(self):
        self.scheduler['101'] = ('COMPLETED', 120, 4, 240)
        folder = self.state / 'arms' / self.arms[0].name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'submission-intent.json').write_text('{}')
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'held')
        self.assertEqual(self.submissions, [])

    def test_failed_or_operator_cancelled_smoke_is_never_retried(self):
        self.scheduler['101'] = ('COMPLETED', 120, 4, 240)
        validation.cycle(self.state, execute=True, runner=self.runner)
        self.scheduler['201'] = ('CANCELLED by 1000', 120, 128, 10)
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'held')
        self.assertEqual(len(self.submissions), 1)
        self.assertFalse((self.state / 'all_passed.json').exists())

    def test_both_completed_valid_logs_create_publication_gate_and_csv(self):
        self.scheduler['101'] = self.scheduler['102'] = ('COMPLETED', 120, 4, 240)
        validation.cycle(self.state, execute=True, runner=self.runner)
        self.scheduler['201'] = self.scheduler['202'] = ('COMPLETED', 300, 128, 10)
        for i, arm in enumerate(self.arms):
            (arm / f'training_core/logs/slurm-{201+i}.log').write_text(log_text())
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'all_passed')
        self.assertTrue((self.state / 'all_passed.json').is_file())
        self.assertIn('416002', (self.state / 'arms' / self.arms[0].name / 'steps.csv').read_text())
        self.assertAlmostEqual(result['budget']['smoke']['actual_gpu_hours'], 21.333333333333332)

    def test_completed_without_three_valid_rows_does_not_create_gate(self):
        self.scheduler['101'] = ('COMPLETED', 120, 4, 240)
        validation.cycle(self.state, execute=True, runner=self.runner)
        self.scheduler['201'] = ('COMPLETED', 300, 128, 10)
        (self.arms[0] / 'training_core/logs/slurm-201.log').write_text(log_text().replace('lm loss: 1.10', 'lm loss: nan'))
        result = validation.cycle(self.state, execute=True, runner=self.runner)
        self.assertEqual(result['state'], 'held')
        self.assertFalse((self.state / 'all_passed.json').exists())

    def test_changed_successful_log_invalidates_publication_gate(self):
        self.scheduler['101'] = self.scheduler['102'] = ('COMPLETED', 120, 4, 240)
        validation.cycle(self.state, execute=True, runner=self.runner)
        self.scheduler['201'] = self.scheduler['202'] = ('COMPLETED', 300, 128, 10)
        for i, arm in enumerate(self.arms):
            (arm / f'training_core/logs/slurm-{201+i}.log').write_text(log_text())
        validation.verify(self.state, runner=self.runner)
        (self.arms[0] / 'training_core/logs/slurm-201.log').write_text('damaged')
        with self.assertRaises(ValueError):
            validation.verify(self.state, runner=self.runner)
        self.assertFalse((self.state / 'all_passed.json').exists())


if __name__ == '__main__':
    unittest.main()
