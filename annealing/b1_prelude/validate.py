"""Bounded cache-gated B1 smoke validation; standard library only, no automatic retries.

Initialize only after all source and rendered-script changes are final. ``watch
--execute`` admits each pinned smoke once after cache.py verifies its preparation.
``verify`` refreshes scheduler/log evidence without submitting jobs and is the
publication gate. STOP prevents further admissions without cancelling allocations.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
PACKAGE_PREFIX = 'annealing/b1_prelude/'
MODEL_OVERLAY = 'config/experiments/oellm_32b_dense/b1_prelude_annealing.yaml'
RESULT_PATH = PACKAGE_PREFIX + 'validation_result.json'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}
MODEL_GUARD = dict(fp8='hybrid', fp8_recipe='tensorwise', first_last_layers_bf16=True,
                   num_layers_at_start_in_bf16=1, num_layers_at_end_in_bf16=1)
METRICS = {
    'elapsed time per iteration (ms)': 'iteration_ms',
    'throughput per GPU (TFLOP/s/GPU)': 'throughput_tflops_per_gpu',
    'Tokens per second per GPU (Tok/s/GPU)': 'tokens_per_second_per_gpu',
    'learning rate': 'learning_rate', 'lm loss': 'lm_loss', 'grad norm': 'grad_norm',
    'number of skipped iterations': 'skipped_iterations',
    'number of nan iterations': 'nan_iterations',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_step(line):
    match = re.search(r'\biteration\s+(\d+)\s*/\s*(\d+)\s*\|', line)
    if not match:
        return None
    result = {'iteration': int(match[1]), 'configured_endpoint': int(match[2])}
    for field in line[match.end():].split('|'):
        key, separator, value = field.strip().partition(':')
        if separator and key in METRICS:
            name = METRICS[key]
            if name in result:
                raise ValueError('Duplicate training metric: ' + key)
            try:
                result[name] = float(value.strip())
            except ValueError as error:
                raise ValueError('Unknown numeric training metric: ' + field) from error
    required = set(METRICS.values()) - {'tokens_per_second_per_gpu'}
    if not required.issubset(result):
        raise ValueError('Unknown/incomplete training row; missing ' + str(sorted(required - result.keys())))
    for key in set(METRICS.values()) & result.keys():
        if not math.isfinite(result[key]):
            raise ValueError('Nonfinite training metric: ' + key)
    if not 0 < result['lm_loss'] < 100 or result['grad_norm'] < 0:
        raise ValueError('Invalid loss or gradient norm')
    for key in ['iteration_ms', 'throughput_tflops_per_gpu', 'learning_rate', 'tokens_per_second_per_gpu']:
        if key in result and result[key] <= 0:
            raise ValueError('Nonpositive training metric: ' + key)
    for key in ['skipped_iterations', 'nan_iterations']:
        if result[key] != 0:
            raise ValueError('Skipped or NaN training update')
        result[key] = int(result[key])
    return result


def validate_log(text, seed_iteration=416000, updates=3, require_cursor_reset=False):
    restored = [int(value) for value in re.findall(r'successfully loaded checkpoint[^\n]*? at iteration\s+(\d+)\b', text)]
    if not restored or set(restored) != {seed_iteration}:
        raise ValueError('Missing or conflicting native checkpoint restore evidence')
    if 'Traceback (most recent call last)' in text:
        raise ValueError('Training log contains a traceback')
    cursors = [int(value) for value in re.findall(
        rf'Annealing fork validation cursor at iteration {seed_iteration}: (\d+) -> 0\b', text)]
    if require_cursor_reset and (not cursors or len(set(cursors)) != 1):
        raise ValueError('Missing or conflicting fork validation-cursor evidence')
    steps = [row for line in text.splitlines() if (row := parse_step(line)) is not None]
    if [row['iteration'] for row in steps] != list(range(seed_iteration + 1, seed_iteration + updates + 1)):
        raise ValueError('Require exactly three consecutive, unique post-restore training rows')
    previous = None
    for row in steps:
        row['loss_delta'] = None if previous is None else row['lm_loss'] - previous['lm_loss']
        row['loss_rate_per_update'] = None if previous is None else row['loss_delta'] / (row['iteration'] - previous['iteration'])
        previous = row
    return {'restored_iteration': seed_iteration, 'steps': steps,
            'fork_validation_cursor': dict(previous=cursors[0], current=0) if cursors else None,
            'whole_window_loss_rate_per_update': (steps[-1]['lm_loss'] - steps[0]['lm_loss']) / (updates - 1),
            'loss_monotonicity_required': False}


def parse_accounting(text, jobs):
    records = {}
    for line in text.splitlines():
        values = line.split('|')
        if len(values) == 7 and not values[-1]:
            values.pop()
        if len(values) != 6:
            raise ValueError('Unknown sacct format: ' + line)
        job, state, elapsed, tres, exit_code, limit = values
        if job not in jobs:
            raise ValueError('Unrequested scheduler accounting row: ' + job)
        record = dict(job=job, state=state, elapsed_seconds=int(elapsed), alloc_tres=tres,
                      exit_code=exit_code, limit_minutes=int(limit))
        if job in records and records[job] != record:
            raise ValueError('Conflicting accounting rows: ' + job)
        records[job] = record
    if set(records) != set(jobs):
        raise ValueError('Missing scheduler accounting for ' + str(sorted(set(jobs) - records.keys())))
    for row in records.values():
        resources = dict(item.split('=', 1) for item in row['alloc_tres'].split(',') if '=' in item)
        gpus = int(resources['gres/gpu']) if 'gres/gpu' in resources else sum(int(v) for k, v in resources.items() if k.startswith('gres/gpu:'))
        if row['elapsed_seconds'] < 0 or gpus < 0 or (row['elapsed_seconds'] > 0 and gpus == 0):
            raise ValueError('Allocated GPU accounting is missing/invalid: ' + row['job'])
        row['allocated_gpus'] = gpus
        row['gpu_hours'] = gpus * row['elapsed_seconds'] / 3600
        row['base_state'] = row['state'].split()[0].rstrip('+')
    return records


def check_pins(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Pinned source/configuration/script changed: ' + path)


def source_git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT).decode()


def source_paths(repo, *args):
    return {name for name in source_git(repo, *args).split('\0') if name}


def capture_source_tree(repo, allowed_paths):
    """Capture actual Git provenance; allowed paths are reviewed exact filenames."""
    repo = Path(repo).resolve()
    allowed = sorted(set(allowed_paths))
    if any(not (name.startswith(PACKAGE_PREFIX) or name == MODEL_OVERLAY) or '..' in Path(name).parts for name in allowed):
        raise ValueError('Source-tree allowlist is outside the reviewed package')
    mega = repo / 'submodules/Megatron-LM'
    result = dict(base_head=source_git(repo, 'rev-parse', 'HEAD').strip(),
                  megatron_path='submodules/Megatron-LM', megatron_head=source_git(mega, 'rev-parse', 'HEAD').strip(),
                  allowed_paths=allowed, reserved_result_path=RESULT_PATH)
    check_source_tree(repo, result)
    return result


def check_source_tree(repo, identity):
    """Reject unreviewed baseline/index/worktree/dependency drift without writes."""
    repo = Path(repo).resolve()
    allowed = set(identity['allowed_paths'])
    if any(not (name.startswith(PACKAGE_PREFIX) or name == MODEL_OVERLAY) or '..' in Path(name).parts for name in allowed):
        raise ValueError('Invalid reviewed source allowlist')
    # The publisher alone binds this reserved generated file to a durable intent.
    allowed.add(RESULT_PATH)
    base = identity['base_head']
    source_git(repo, 'merge-base', '--is-ancestor', base, 'HEAD')
    changed = set()
    for args in [('diff', '--name-only', '-z', '--ignore-submodules=none', base, 'HEAD', '--'),
                 ('diff', '--cached', '--name-only', '-z', '--ignore-submodules=none', base, '--'),
                 ('diff', '--name-only', '-z', '--ignore-submodules=none', '--'),
                 ('ls-files', '--others', '--exclude-standard', '-z')]:
        changed.update(source_paths(repo, *args))
    if changed - allowed:
        raise ValueError('Unreviewed baseline/source changes: ' + repr(sorted(changed - allowed)))
    for record in source_paths(repo, 'ls-files', '-v', '-z'):
        if record[0] != 'H' and record[2:] not in allowed:
            raise ValueError('Protected index flag hides baseline source: ' + record[2:])
    if identity['megatron_path'] != 'submodules/Megatron-LM':
        raise ValueError('Unexpected Megatron source path')
    mega = repo / identity['megatron_path']
    expected = identity['megatron_head']
    gitlink = source_git(repo, 'ls-tree', base, '--', identity['megatron_path']).strip().split()
    if len(gitlink) != 4 or gitlink[:3] != ['160000', 'commit', expected]:
        raise ValueError('Megatron HEAD does not match the baseline gitlink')
    if source_git(mega, 'rev-parse', 'HEAD').strip() != expected:
        raise ValueError('Megatron revision changed')
    for args in [('diff', '--name-only', '-z', 'HEAD', '--'),
                 ('diff', '--cached', '--name-only', '-z', 'HEAD', '--'),
                 ('diff', '--name-only', '-z', '--'),
                 ('ls-files', '--others', '--exclude-standard', '-z')]:
        if source_paths(mega, *args):
            raise ValueError('Megatron source/index must remain clean')
    if any(record[0] != 'H' for record in source_paths(mega, 'ls-files', '-v', '-z')):
        raise ValueError('Protected index flags hide Megatron source')
    return identity


def batch_contract(path, cache):
    text = path.read_text()
    for option, value in [('nodes', '32'), ('gpus-per-node', '4'), ('time', '00:10:00')]:
        if not re.search(r'^#SBATCH\s+--' + option + r'(?:=|\s+)' + re.escape(value) + r'\s*$', text, re.M):
            raise ValueError('Wrong smoke batch resource: ' + option)
    lock = str(Path(cache) / '.writer.lock')
    launch = re.search(r'^\s*srun\b', text, re.M)
    for opened in re.finditer(r'^\s*exec\s+([0-9]+)\s*>\s*([^\n]+)$', text, re.M):
        if shlex.split(opened[2]) != [lock]:
            continue
        flock = re.search(r'^\s*flock\s+-n\s+' + opened[1] + r'\b', text, re.M)
        if flock and launch and opened.start() < flock.start() < launch.start():
            return
    raise ValueError('Smoke batch must hold the shared cache writer lock before srun')


def initialize(root, arms, historical_jobs, repo=REPO):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    if root.exists():
        raise ValueError('Use a new validation state directory; never replace a manifest')
    if len(arms) != 2 or len({Path(arm).resolve() for arm in arms}) != 2:
        raise ValueError('Pin exactly two distinct dominant1/dominant2 arms')
    package = repo / 'annealing/b1_prelude'
    reviewed = source_paths(repo, 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', PACKAGE_PREFIX, MODEL_OVERLAY)
    reviewed = {name for name in reviewed if '__pycache__' not in Path(name).parts and name != RESULT_PATH}
    source_files = [repo / name for name in sorted(reviewed)]
    identity = capture_source_tree(repo, reviewed)
    manifest = dict(version=1, created_utc=timestamp(), repo=str(repo), source_pins={str(p): sha(p) for p in source_files},
                    source_tree=identity,
                    arms=[], historical_jobs=historical_jobs, ceilings_gpu_hours={'preparation': 32, 'smoke': 128 * 600 * 2 / 3600},
                    policy='Two one-allocation non-resumable smoke tests; no retries; STOP preserves active allocations')
    for arm_path in arms:
        arm = Path(arm_path).resolve()
        recipe = read(arm / 'recipe.json')
        expected = dict(seed_iteration=416000, smoke=True, nodes=32, maximum_allocations=1, allocation_walltime='00:10:00')
        if any(recipe.get(key) != value for key, value in expected.items()):
            raise ValueError('Arm is not the approved 416000/32-node/three-update smoke: ' + str(arm))
        settings = read(arm / 'cache-preparation/settings.json')
        model = settings['target']
        guard = dict(MODEL_GUARD, exit_interval=416003, eval_iters=1, eval_global_batch_size=16,
                     global_batch_size=4096, seq_length=4096, save=None, non_persistent_save_interval=None)
        if any(key not in model or model[key] != value for key, value in guard.items()):
            raise ValueError('Resolved smoke model/validation/precision guard failed: ' + str(arm))
        scripts = sorted((arm / 'training_core').glob('job_*.sbatch'))
        if not scripts:
            raise ValueError('No rendered smoke script: ' + str(arm))
        script = scripts[-1]
        batch_contract(script, model['data_cache_path'])
        cursor_reset = recipe.get('fork_validation_cursor') == 'reset_on_seed_restore'
        if cursor_reset and str(package / 'training.py') not in script.read_text():
            raise ValueError('Smoke launcher is missing the fork cursor reset entry point')
        paths = [arm / 'recipe.json', arm / 'datamix.txt', arm / 'cache-preparation/settings.json',
                 arm / 'cache-preparation/submitted.json', arm / 'cache-preparation/render.json', script]
        paths.extend(p for p in (arm / 'config').rglob('*') if p.is_file())
        reuse = arm / 'cache-preparation/reuse.json'
        if reuse.exists():
            paths.append(reuse)
        rendered_cache = read(arm / 'cache-preparation/render.json')
        if rendered_cache.get('script'):
            paths.append(Path(rendered_cache['script']))
        cache_job = str(read(arm / 'cache-preparation/submitted.json')['job'])
        if not cache_job.isdigit():
            raise ValueError('Invalid preparation job receipt')
        manifest['arms'].append(dict(name=arm.name, root=str(arm), mix=recipe['mix'], cache_job=cache_job,
                                     script=str(script), pins={str(p): sha(p) for p in paths}, model_guard=guard,
                                     cache_gpu_count=4, cache_seconds=14400, smoke_gpu_count=128, smoke_seconds=600,
                                     require_cursor_reset=cursor_reset))
    if {arm['mix'] for arm in manifest['arms']} != {'dominant1', 'dominant2'}:
        raise ValueError('Require both mixture identities')
    jobs = [arm['cache_job'] for arm in manifest['arms']] + [job['job'] for job in historical_jobs]
    if len(set(jobs)) != len(jobs):
        raise ValueError('Duplicate scheduler identities in manifest')
    root.mkdir(parents=True)
    save(root / 'validation.json', manifest)
    return manifest


def account(root, manifest, runner):
    specs = []
    for arm in manifest['arms']:
        specs.append(dict(job=arm['cache_job'], category='preparation', gpus=4, seconds=14400))
        receipt = root / 'arms' / arm['name'] / 'submitted.json'
        if receipt.exists():
            specs.append(dict(job=str(read(receipt)['job']), category='smoke', gpus=128, seconds=600))
    specs.extend(dict(row, category='historical', seconds=0) for row in manifest['historical_jobs'])
    jobs = [row['job'] for row in specs]
    if len(set(jobs)) != len(jobs):
        raise ValueError('Repeated scheduler identity')
    command = ['sacct', '-X', '-nP', '-j', ','.join(jobs), '--format=JobIDRaw,State%64,ElapsedRaw,AllocTRES%2000,ExitCode,TimelimitRaw']
    result = runner(command, text=True, capture_output=True, timeout=60)
    (root / 'sacct-latest.txt').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError('Scheduler accounting unavailable')
    records = parse_accounting(result.stdout, jobs)
    budget = {category: dict(actual_gpu_hours=0, reserved_gpu_hours=0, unsubmitted_forecast_gpu_hours=0, ledger=[]) for category in ['preparation', 'smoke', 'historical']}
    for spec in specs:
        row = dict(records[spec['job']], category=spec['category'], expected_gpus=spec['gpus'])
        if spec['category'] != 'historical' and row['allocated_gpus'] not in (0, spec['gpus']):
            raise ValueError('Actual GPU allocation differs from approved resources: ' + spec['job'])
        reserve = 0
        if row['base_state'] in ACTIVE and spec['category'] != 'historical':
            reserve = max(0, spec['seconds'] - row['elapsed_seconds']) * spec['gpus'] / 3600
        row['remaining_reservation_gpu_hours'] = reserve
        entry = budget[spec['category']]
        entry['ledger'].append(row)
        entry['actual_gpu_hours'] += row['gpu_hours']
        entry['reserved_gpu_hours'] += reserve
    missing = len(manifest['arms']) - len(budget['smoke']['ledger'])
    budget['smoke']['unsubmitted_forecast_gpu_hours'] = missing * 128 * 600 / 3600
    for category, ceiling in manifest['ceilings_gpu_hours'].items():
        entry = budget[category]
        entry['ceiling_gpu_hours'] = ceiling
        entry['within_ceiling'] = sum(entry[k] for k in ['actual_gpu_hours', 'reserved_gpu_hours', 'unsubmitted_forecast_gpu_hours']) <= ceiling + 1e-9
    return records, budget


def stopped(root, manifest):
    return any(path.exists() for path in [root / 'STOP', root.parent / 'STOP'] + [Path(arm['root']) / 'STOP' for arm in manifest['arms']])


def submit_once(root, manifest, arm, runner):
    folder = root / 'arms' / arm['name']
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'submission-intent.json').exists():
        raise ValueError('Existing smoke submission intent requires reconciliation; no retry')
    if stopped(root, manifest):
        raise ValueError('STOP appeared before submission')
    check_pins(manifest['source_pins'])
    check_source_tree(manifest['repo'], manifest['source_tree'])
    check_pins(arm['pins'])
    command = ['sbatch', '--parsable', arm['script']]
    intent = dict(at=timestamp(), command=command, cwd=manifest['repo'], script_sha256=sha(arm['script']), maximum_allocations=1)
    with (folder / 'submission-intent.json').open('x') as stream:
        json.dump(intent, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    result = runner(command, cwd=manifest['repo'], text=True, capture_output=True, timeout=60)
    (folder / 'submission-output.txt').write_text(result.stdout + result.stderr)
    match = re.fullmatch(r'(\d+)(?:;[^;\s]+)?', result.stdout.strip())
    if result.returncode or not match:
        raise ValueError('Uncertain smoke submission; retain intent and reconcile scheduler')
    receipt = dict(intent, job=match[1])
    save(folder / 'submitted.json', receipt)
    (Path(arm['root']) / 'submitted.txt').write_text(match[1] + '\n')
    return receipt


def cycle_unlocked(root, execute, runner):
    manifest = read(root / 'validation.json')
    result = dict(at=timestamp(), manifest_sha256=sha(root / 'validation.json'), arms=[])
    try:
        records, budget = account(root, manifest, runner)
        result['budget'] = budget
        if stopped(root, manifest):
            result['state'] = 'stopped'
            return result
        check_pins(manifest['source_pins'])
        check_source_tree(manifest['repo'], manifest['source_tree'])
        for arm in manifest['arms']:
            check_pins(arm['pins'])
        if not all(budget[key]['within_ceiling'] for key in manifest['ceilings_gpu_hours']):
            raise ValueError('Validation allocation budget exceeded')
        # Review every known terminal outcome before making any new admission.
        for category in ['preparation', 'smoke']:
            for row in budget[category]['ledger']:
                if row['base_state'] not in ACTIVE and (row['base_state'] != 'COMPLETED' or row['exit_code'] != '0:0'):
                    raise ValueError('Failed/cancelled ' + category + ' allocation requires review; no retries: ' + row['job'])
        for arm in manifest['arms']:
            arm_root = Path(arm['root'])
            folder = root / 'arms' / arm['name']
            folder.mkdir(parents=True, exist_ok=True)
            item = dict(name=arm['name'], cache_job=arm['cache_job'])
            result['arms'].append(item)
            prep = records[arm['cache_job']]
            if prep['base_state'] in ACTIVE:
                item['state'] = 'waiting_cache'
                continue
            if prep['base_state'] != 'COMPLETED' or prep['exit_code'] != '0:0':
                raise ValueError('Preparation failed or was cancelled: ' + arm['name'])
            receipt_path = folder / 'submitted.json'
            if not receipt_path.exists():
                if not execute:
                    item['state'] = 'cache_check_and_submission_require_execute'
                    continue
                if (folder / 'submission-intent.json').exists():
                    raise ValueError('Uncertain prior smoke submission; no retries: ' + arm['name'])
                command = [sys.executable, str(Path(manifest['repo']) / 'annealing/b1_prelude/cache.py'), 'check', '--root', str(arm_root)]
                checked = runner(command, cwd=manifest['repo'], text=True, capture_output=True, timeout=900)
                (folder / 'cache-check.log').write_text(checked.stdout + checked.stderr)
                if checked.returncode:
                    raise ValueError('Production cache check failed: ' + arm['name'])
                ready_path = arm_root / 'cache-preparation/ready.json'
                ready = read(ready_path)
                if str(ready['job']) != arm['cache_job'] or ready['settings_sha256'] != sha(arm_root / 'cache-preparation/settings.json'):
                    raise ValueError('Cache readiness is not bound to the approved preparation')
                for split in ['train', 'valid']:
                    proof = ready['coverage'][split]
                    if proof['prefixes'] != 294 or proof['requested_samples'] <= 0:
                        raise ValueError('Incomplete cache prefix/sample coverage')
                save(folder / 'cache-ready.json', dict(path=str(ready_path), sha256=sha(ready_path), proof=ready))
                receipt = submit_once(root, manifest, arm, runner)
                item.update(state='submitted_smoke', job=receipt['job'])
                continue
            receipt = read(receipt_path)
            job = str(receipt['job'])
            if receipt['script_sha256'] != sha(arm['script']):
                raise ValueError('Smoke submission receipt script changed')
            smoke = records[job]
            item['job'] = job
            if smoke['base_state'] in ACTIVE:
                item['state'] = 'running_smoke'
                continue
            if smoke['base_state'] != 'COMPLETED' or smoke['exit_code'] != '0:0':
                raise ValueError('Smoke failed or was cancelled; no retries: ' + arm['name'])
            log = arm_root / 'training_core/logs' / ('slurm-' + job + '.log')
            report = dict(validate_log(log.read_text(), require_cursor_reset=arm.get('require_cursor_reset', False)),
                          job=job, log=str(log), log_sha256=sha(log),
                          script=arm['script'], script_sha256=sha(arm['script']), cache_ready=read(folder / 'cache-ready.json'),
                          scheduler=smoke, manifest_sha256=result['manifest_sha256'])
            save(folder / 'report.json', report)
            with (folder / 'steps.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(report['steps'][0]))
                writer.writeheader()
                writer.writerows(report['steps'])
            item.update(state='passed', report=str(folder / 'report.json'), report_sha256=sha(folder / 'report.json'))
        if all(arm['state'] == 'passed' for arm in result['arms']):
            result['state'] = 'all_passed'
            save(root / 'all_passed.json', result)
        else:
            states = {arm['state'] for arm in result['arms']}
            result['state'] = 'running_smoke' if states & {'running_smoke', 'submitted_smoke'} else 'waiting_cache'
            # Include newly submitted reservations immediately, rather than waiting a tick.
            if 'submitted_smoke' in states:
                _, result['budget'] = account(root, manifest, runner)
        return result
    except Exception as error:
        result.update(state='held', error_type=type(error).__name__, error=str(error))
        return result


def publish_status(root, result):
    if result['state'] != 'all_passed' and (root / 'all_passed.json').exists():
        (root / 'all_passed.json').replace(root / ('all_passed.invalidated-' + str(time.time_ns()) + '.json'))
    save(root / 'status.json', result)
    save(root / 'lease.json', dict(host=socket.gethostname(), pid=os.getpid(), at=timestamp(), state=result['state']))


def cycle(root, execute=False, runner=subprocess.run):
    root = Path(root).resolve()
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = cycle_unlocked(root, execute, runner)
        publish_status(root, result)
        return result


def verify(root, runner=subprocess.run):
    result = cycle(root, execute=False, runner=runner)
    if result['state'] != 'all_passed':
        raise ValueError('Validation publication gate is closed: ' + result.get('error', result['state']))
    manifest = read(Path(root) / 'validation.json')
    check_source_tree(manifest['repo'], manifest['source_tree'])
    return read(Path(root) / 'all_passed.json')


def watch(root, execute=False):
    root = Path(root).resolve()
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = cycle_unlocked(root, execute, subprocess.run)
            publish_status(root, result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if result['state'] in {'all_passed', 'held', 'stopped'}:
                return result
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'cycle', 'watch', 'verify'])
    parser.add_argument('--root', type=Path, required=True, help='Dedicated new validation state directory')
    parser.add_argument('--arm', type=Path, action='append', default=[], help='Two arm roots, used only by init')
    parser.add_argument('--historical-job', action='append', default=[], metavar='JOB:GPUS', help='Previously allocated/held job; separate historical ledger')
    parser.add_argument('--execute', action='store_true', help='Enable production cache checks and one smoke submission per arm')
    args = parser.parse_args()
    try:
        if args.action == 'init':
            history = []
            for entry in args.historical_job:
                job, gpus = entry.split(':')
                if not job.isdigit() or not gpus.isdigit() or int(gpus) <= 0:
                    raise ValueError('Historical jobs require JOB:POSITIVE_GPU_COUNT')
                history.append(dict(job=job, gpus=int(gpus)))
            result = initialize(args.root, args.arm, history)
        elif args.action == 'watch':
            result = watch(args.root, args.execute)
        elif args.action == 'verify':
            result = verify(args.root)
        else:
            result = cycle(args.root, args.execute)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2 if result.get('state') in {'held', 'stopped'} else 0
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
