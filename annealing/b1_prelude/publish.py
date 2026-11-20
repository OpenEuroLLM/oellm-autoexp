"""Publish the reviewed package after fresh validation, with intent-bound recovery.

cycle/watch resume only the exact reviewed index or single base-parent commit.
reconcile requires an existing intent. STOP prevents subsequent writes; existing
commits/index entries remain intact. Remote changes never authorize force pushes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
REMOTE = 'https://github.com/OpenEuroLLM/oellm-autoexp.git'
BRANCH = 'prod/oellm_32b_dense'
RESULT = 'annealing/b1_prelude/validation_result.json'
MESSAGE = ('Add validated standalone B1 annealing package and production cache gate\n\n'
           'Both mixtures passed bounded three-update restore/training checks; see '
           'annealing/b1_prelude/validation_result.json for evidence and limits.\n\nAssisted by GPT-6')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_files(spec, repo=None):
    repo = Path(repo or REPO).resolve()
    for name, digest in spec['sources'].items():
        path = repo / name
        allowed = name.startswith('annealing/b1_prelude/') or name == 'config/experiments/oellm_32b_dense/b1_prelude_annealing.yaml'
        if not allowed or name == RESULT or '..' in Path(name).parts or not path.resolve().is_relative_to(repo):
            raise ValueError('Outside reviewed package scope: ' + name)
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError('Reviewed source changed: ' + name)


def git_bytes(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.STDOUT)


def git(*args):
    return git_bytes(*args).decode().strip()


def remote_head():
    output = git('-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential',
                 'ls-remote', REMOTE, 'refs/heads/' + BRANCH)
    rows = output.splitlines()
    if len(rows) != 1:
        raise ValueError('Remote branch identity unavailable')
    return rows[0].split()[0]


def check_stop(root):
    manifest = read(root / 'validation.json')
    paths = [root / 'STOP', root.parent / 'STOP'] + [Path(arm['root']) / 'STOP' for arm in manifest.get('arms', [])]
    if any(path.exists() for path in paths):
        raise ValueError('Preserve STOP')


def write_guard(root, spec):
    import validate
    check_stop(root)
    verify_files(spec, REPO)
    if sha(root / 'validation.json') != spec['validation_manifest_sha256']:
        raise ValueError('Validation manifest changed after publication review')
    identity = read(root / 'validation.json')['source_tree']
    if identity['base_head'] != spec['base']:
        raise ValueError('Validation and publication baseline revisions differ')
    if set(spec['sources']) != set(identity['allowed_paths']):
        raise ValueError('Publication must include every exact reviewed package path')
    validate.check_source_tree(REPO, identity)
    check_stop(root)


def file_identity(data, mode, algorithm):
    header = b'blob ' + str(len(data)).encode() + b'\0'
    return dict(sha256=hashlib.sha256(data).hexdigest(), mode=mode,
                blob=hashlib.new(algorithm, header + data).hexdigest())


def tree(revision):
    result = {}
    for entry in git_bytes('ls-tree', '-r', '-z', revision).split(b'\0'):
        if not entry:
            continue
        header, name = entry.split(b'\t', 1)
        mode, kind, oid = header.decode().split()
        result[os.fsdecode(name)] = dict(mode=mode, blob=oid)
    return result


def desired_tree(intent):
    result = tree(intent['base'])
    result.update({name: dict(mode=row['mode'], blob=row['blob']) for name, row in intent['expected_files'].items()})
    return result


def index_tree():
    result = {}
    for entry in git_bytes('ls-files', '--stage', '-z').split(b'\0'):
        if not entry:
            continue
        header, name = entry.split(b'\t', 1)
        mode, oid, stage = header.decode().split()
        if stage != '0':
            raise ValueError('Preserve unmerged index entries')
        result[os.fsdecode(name)] = dict(mode=mode, blob=oid)
    return result


def checked_reports(evidence):
    result = {}
    for arm in evidence['arms']:
        path = Path(arm['report'])
        if arm['state'] != 'passed' or sha(path) != arm['report_sha256'] or arm['name'] in result:
            raise ValueError('Per-arm validation report changed or is incomplete')
        report = read(path)
        if report['job'] != arm['job'] or report['manifest_sha256'] != evidence['manifest_sha256']:
            raise ValueError('Per-arm validation report provenance changed')
        result[arm['name']] = report
    return result


def create_intent(root, spec, evidence):
    if git('rev-parse', 'HEAD') != spec['base'] or remote_head() != spec['remote_head']:
        raise ValueError('Local/remote revision changed; reviewed integration required')
    if git('diff', '--cached', '--name-only'):
        raise ValueError('Preserve existing staged work')
    if (REPO / RESULT).exists():
        raise ValueError('Unexpected validation result file; review before reuse')
    payload = dict(validation=evidence, arm_reports=checked_reports(evidence),
                   source_tree=read(root / 'validation.json')['source_tree'], validation_manifest_sha256=sha(root / 'validation.json'),
                   data_identity=spec.get('data_identity'), limitations=spec.get('limitations'), captured_time=time.time())
    algorithm = git('rev-parse', '--show-object-format')
    files = {}
    for name in sorted(spec['sources']):
        path = REPO / name
        mode = '100755' if path.stat().st_mode & stat.S_IXUSR else '100644'
        files[name] = file_identity(path.read_bytes(), mode, algorithm)
    files[RESULT] = file_identity(encoded(payload), '100644', algorithm)
    intent = dict(version=2, time=time.time(), base=spec['base'], remote_head=spec['remote_head'],
                  validation_manifest_sha256=spec['validation_manifest_sha256'], sources=spec['sources'],
                  paths=sorted(files), expected_files=files, result_payload=payload, object_format=algorithm,
                  validation_result_sha256=files[RESULT]['sha256'], commit_message=MESSAGE)
    # Persist before creating the generated file or modifying the index.
    write_guard(root, spec)
    save(root / 'publication-intent.json', intent)
    return intent


def verify_intent(root, spec, intent):
    if intent.get('version') != 2 or intent['base'] != spec['base'] or intent['remote_head'] != spec['remote_head']:
        raise ValueError('Unknown or mismatched publication intent; explicit review required')
    if intent['sources'] != spec['sources'] or intent['validation_manifest_sha256'] != spec['validation_manifest_sha256']:
        raise ValueError('Publication intent source/validation identity changed')
    expected_paths = sorted(list(spec['sources']) + [RESULT])
    if intent['paths'] != expected_paths or sorted(intent['expected_files']) != expected_paths or intent['commit_message'] != MESSAGE:
        raise ValueError('Publication intent does not bind the exact reviewed files/message')
    algorithm = git('rev-parse', '--show-object-format')
    if algorithm != intent['object_format']:
        raise ValueError('Git object identity changed')
    for name in spec['sources']:
        path = REPO / name
        mode = '100755' if path.stat().st_mode & stat.S_IXUSR else '100644'
        if file_identity(path.read_bytes(), mode, algorithm) != intent['expected_files'][name]:
            raise ValueError('Reviewed file differs from publication intent: ' + name)
    payload = intent['result_payload']
    if payload['validation_manifest_sha256'] != spec['validation_manifest_sha256'] or payload.get('data_identity') != spec.get('data_identity') or payload.get('limitations') != spec.get('limitations'):
        raise ValueError('Publication result payload provenance changed')
    if payload['arm_reports'] != checked_reports(payload['validation']) or payload['source_tree'] != read(root / 'validation.json')['source_tree']:
        raise ValueError('Embedded validation reports/source provenance changed')
    expected = file_identity(encoded(payload), '100644', algorithm)
    if expected != intent['expected_files'][RESULT] or expected['sha256'] != intent['validation_result_sha256']:
        raise ValueError('Publication result payload/hash mismatch')
    if (REPO / RESULT).exists() and (REPO / RESULT).read_bytes() != encoded(payload):
        raise ValueError('Local validation result differs from its publication intent')


def ensure_owned_index(intent):
    actual = index_tree()
    baseline = tree(intent['base'])
    desired = desired_tree(intent)
    for name in set(actual) | set(baseline):
        if actual.get(name) == baseline.get(name):
            continue
        if name not in intent['expected_files'] or actual.get(name) != desired.get(name):
            raise ValueError('Preserve staged work outside the exact publication intent: ' + name)


def verify_candidate(commit, intent):
    if git('rev-parse', 'HEAD') != commit or git('show', '-s', '--format=%P', commit).split() != [intent['base']]:
        raise ValueError('Publication candidate must be the exact single-base-parent HEAD')
    if git('show', '-s', '--format=%B', commit) != intent['commit_message'] or tree(commit) != desired_tree(intent):
        raise ValueError('Publication candidate tree/message differs from the reviewed intent')
    for name, row in intent['expected_files'].items():
        if hashlib.sha256(git_bytes('cat-file', 'blob', row['blob'])).hexdigest() != row['sha256']:
            raise ValueError('Publication candidate blob differs from reviewed bytes: ' + name)


def cycle(root, execute=False):
    root = Path(root).resolve()
    spec = read(root / 'publish-spec.json')
    check_stop(root)
    verify_files(spec, REPO)
    if sha(root / 'validation.json') != spec['validation_manifest_sha256']:
        raise ValueError('Validation manifest changed after publication review')
    import validate
    evidence = validate.verify(root)
    # A STOP written during validation must precede every subsequent write.
    write_guard(root, spec)
    if not execute:
        return dict(status='would_publish', validation=evidence)
    intent_path = root / 'publication-intent.json'
    intent = read(intent_path) if intent_path.exists() else create_intent(root, spec, evidence)
    verify_intent(root, spec, intent)
    commit = git('rev-parse', 'HEAD')
    if commit == intent['base']:
        if remote_head() != intent['remote_head']:
            raise ValueError('Remote branch changed; no publication writes permitted')
        ensure_owned_index(intent)
        path = REPO / RESULT
        expected_bytes = encoded(intent['result_payload'])
        if path.exists():
            if path.read_bytes() != expected_bytes:
                raise ValueError('Generated validation result differs from its owned intent')
        else:
            write_guard(root, spec)
            with path.open('xb') as stream:
                stream.write(expected_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        write_guard(root, spec)
        ensure_owned_index(intent)
        check_stop(root)
        git('add', '--', *intent['paths'])
        if index_tree() != desired_tree(intent):
            raise ValueError('Staged tree differs from the exact publication intent')
        write_guard(root, spec)
        git('commit', '-m', intent['commit_message'])
        commit = git('rev-parse', 'HEAD')
        verify_candidate(commit, intent)
        save(root / 'publication.json', dict(local_commit=commit, time=time.time(), validation_result_sha256=intent['validation_result_sha256']))
    else:
        verify_candidate(commit, intent)
    prior = read(root / 'publication.json') if (root / 'publication.json').exists() else {}
    if prior.get('local_commit', commit) != commit or prior.get('published_commit', commit) != commit:
        raise ValueError('Existing publication receipt names another candidate')
    remote = remote_head()
    if remote == commit:
        result = dict(local_commit=commit, published_commit=commit, time=time.time(), status='published', reconciled=True)
        save(root / 'publication.json', result)
        return result
    if remote != intent['remote_head']:
        raise ValueError('Remote branch changed before publication; no force push')
    write_guard(root, spec)
    verify_candidate(commit, intent)
    check_stop(root)
    git('-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential',
        'push', REMOTE, commit + ':refs/heads/' + BRANCH)
    if remote_head() != commit:
        raise ValueError('Remote verification failed')
    result = dict(local_commit=commit, published_commit=commit, time=time.time(), status='published')
    save(root / 'publication.json', result)
    return result


def reconcile(root, execute=False):
    if not (Path(root) / 'publication-intent.json').exists():
        raise ValueError('Reconciliation requires an existing reviewed publication intent')
    return cycle(root, execute)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['cycle', 'watch', 'reconcile'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / 'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                status = reconcile(root, args.execute) if args.action == 'reconcile' else cycle(root, args.execute)
            except Exception as error:
                status = dict(status='held', reason=repr(error), time=time.time())
            save(root / 'publication-status.json', status)
            print(json.dumps(status), flush=True)
            if args.action != 'watch' or status.get('published_commit'):
                break
            time.sleep(60)
