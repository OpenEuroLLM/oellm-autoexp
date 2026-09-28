#!/usr/bin/env python3
"""Package committed Megatron sources, then publish dependency before consumer.

Uses only Python's standard library and Git; no training imports or GPU work.
Preparation and publication are dry-run unless --execute is supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

SUBMODULE = 'submodules/Megatron-LM'
MANIFEST = 'versions/megatron/{name}.json'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *map(str, args)],
                                   text=True, stderr=subprocess.PIPE).strip()


def remote_git(repo, *args):
    # Command-scoped authentication; never extract/store the token.
    return git(repo, '-c', 'credential.helper=',
               '-c', 'credential.helper=!gh auth git-credential', *args)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def read(path):
    return json.loads(path.read_text())


def identity(repo, ignore_submodules=False):
    args = ['status', '--porcelain', '--untracked-files=all']
    if ignore_submodules:
        args.append('--ignore-submodules=all')
    if git(repo, *args):
        raise ValueError(f'Uncommitted source in {repo}; preserve and commit it first')
    # Do not allow assume-unchanged/skip-worktree bits to hide tracked edits.
    if any(line[:1] != 'H' for line in git(repo, 'ls-files', '-v').splitlines()):
        raise ValueError(f'Protected index entries in {repo}')
    return dict(commit=git(repo, 'rev-parse', 'HEAD'), tree=git(repo, 'rev-parse', 'HEAD^{tree}'))


def ref(repo, value, kind):
    git(repo, 'check-ref-format', 'refs/'+kind+'/'+value)
    return value


def url(value):
    # Public manifest URLs must have no userinfo, tokens, query, fragments or local paths.
    if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git', value):
        raise ValueError('Expected credential-free canonical GitHub HTTPS URL')
    return value


def clone(source, dest, commit):
    subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(source), str(dest)],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    git(dest, 'checkout', '--detach', commit)


def plan(a):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', a.name):
        raise ValueError('Version name must be a single safe path component')
    auto, mega = a.autoexp.resolve(), a.megatron.resolve()
    evidence = Path(a.evidence)
    if evidence.is_absolute() or '..' in evidence.parts:
        raise ValueError('Evidence must be a tracked relative autoexp path')
    git(auto, 'ls-files', '--error-unmatch', str(evidence))
    auto_id, mega_id = identity(auto, True), identity(mega)
    if git(mega, 'ls-tree', '-r', 'HEAD').find('160000 commit ') >= 0:
        raise ValueError('Nested Megatron submodules need an explicit additional release policy')
    for repo in (auto, mega):
        ref(repo, a.name, 'tags')
    if git(auto, 'config', '-f', '.gitmodules', '--get',
           'submodule.'+SUBMODULE+'.url') != url(a.megatron_remote):
        raise ValueError('Megatron publication target differs from .gitmodules')
    return dict(schema=1, name=a.name, autoexp_source=str(auto), megatron_source=str(mega),
        autoexp_base=auto_id, megatron=mega_id, autoexp_remote=url(a.autoexp_remote),
        megatron_remote=url(a.megatron_remote),
        autoexp_branch=ref(auto, a.autoexp_branch, 'heads'),
        megatron_branch=ref(mega, a.megatron_branch, 'heads'), evidence=str(evidence),
        evidence_sha256=hashlib.sha256((auto/evidence).read_bytes()).hexdigest())


def prepare(a):
    p = plan(a)
    if a.output.exists():
        raise FileExistsError('Output already exists; verify/publish it or choose a new directory')
    if not a.execute:
        return p
    # No source refs, index, working files or caches are modified.
    a.output.mkdir(parents=True)
    dump(a.output/'plan.json', p)
    auto = a.output/'oellm-autoexp'
    mega = auto/SUBMODULE
    clone(a.autoexp, auto, p['autoexp_base']['commit'])
    clone(a.megatron, mega, p['megatron']['commit'])
    if identity(a.autoexp, True) != p['autoexp_base'] or identity(a.megatron) != p['megatron']:
        raise ValueError('Source changed while cloning; preserve incomplete output for review')
    git(auto, 'remote', 'set-url', 'origin', p['autoexp_remote'])
    git(mega, 'remote', 'set-url', 'origin', p['megatron_remote'])
    version = dict(schema=1, name=p['name'], autoexp_base=p['autoexp_base'],
        megatron={**p['megatron'], 'url':p['megatron_remote'],
                  'branch':p['megatron_branch'], 'tag':p['name']},
        submodule=SUBMODULE, evidence=p['evidence'], evidence_sha256=p['evidence_sha256'],
        scope='Exact committed source; ignored binaries/caches/environments are not packaged')
    manifest = MANIFEST.format(name=p['name'])
    if (auto/manifest).exists():
        raise FileExistsError('Version manifests are immutable; choose a new version')
    dump(auto/manifest, version)
    git(auto, 'add', '--', SUBMODULE, manifest)
    message = f'Pin Megatron {p["megatron"]["commit"][:12]} as {p["name"]}'
    if a.assisted_by:
        message += '\n\nAssisted by '+a.assisted_by
    git(auto, 'commit', '-m', message)
    # Annotated version tags belong only to these isolated publication clones.
    for repo in (auto, mega):
        git(repo, 'tag', '-a', p['name'], '-m', f'{p["name"]}: exact source release; see {p["evidence"]}')
    record = dict(**p, autoexp=identity(auto), manifest=manifest,
                  autoexp_tag_object=git(auto,'rev-parse','refs/tags/'+p['name']),
                  megatron_tag_object=git(mega,'rev-parse','refs/tags/'+p['name']))
    dump(a.output/'release.json', record)
    verify_package(a.output)
    return record


def verify_checkout(auto, name):
    manifest = auto/MANIFEST.format(name=name)
    v = read(manifest)
    if v['name'] != name or v['submodule'] != SUBMODULE:
        raise ValueError('Unexpected version manifest identity')
    if identity(auto/SUBMODULE) != {k:v['megatron'][k] for k in ('commit','tree')}:
        raise ValueError('Initialized Megatron does not match release')
    mode, kind, commit_path = git(auto,'ls-tree','HEAD',SUBMODULE).split(' ',2)
    commit = commit_path.split('\t')[0]
    if (mode,kind,commit) != ('160000','commit',v['megatron']['commit']):
        raise ValueError('Superproject gitlink does not match manifest')
    if git(auto,'config','-f','.gitmodules','--get','submodule.'+SUBMODULE+'.url') != v['megatron']['url']:
        raise ValueError('Submodule URL drift')
    if hashlib.sha256((auto/v['evidence']).read_bytes()).hexdigest() != v['evidence_sha256']:
        raise ValueError('Version evidence changed')
    identity(auto)
    return v


def verify_package(root):
    r = read(root/'release.json')
    auto=root/'oellm-autoexp'; mega=auto/SUBMODULE
    if identity(auto) != r['autoexp'] or identity(mega) != r['megatron']:
        raise ValueError('Packaged source changed')
    v=verify_checkout(auto,r['name'])
    if v['autoexp_base'] != r['autoexp_base'] or v['megatron']['url'] != r['megatron_remote']:
        raise ValueError('Package manifest/receipt mismatch')
    for repo,key in ((auto,'autoexp_tag_object'),(mega,'megatron_tag_object')):
        if git(repo,'rev-parse','refs/tags/'+r['name']) != r[key]:
            raise ValueError('Local tag changed')
        if git(repo,'rev-parse',r['name']+'^{commit}') != git(repo,'rev-parse','HEAD'):
            raise ValueError('Tag does not point to packaged commit')
    return r


def advertised(repo, remote, refs):
    text=remote_git(repo,'ls-remote',remote,*refs)
    return dict(line.split('\t')[::-1] for line in text.splitlines())


def publish_repo(repo, remote, branch, tag, commit, tag_object, execute):
    branch_ref='refs/heads/'+branch; tag_ref='refs/tags/'+tag
    refs=advertised(repo,remote,[branch_ref,tag_ref,tag_ref+'^{}'])
    if tag_ref in refs and refs[tag_ref] != tag_object:
        raise ValueError('Remote version tag exists with a different object; never replace it')
    if branch_ref in refs and refs[branch_ref] != commit:
        remote_git(repo,'fetch','--no-tags',remote,branch_ref)
        git(repo,'merge-base','--is-ancestor',refs[branch_ref],commit)
    if execute:
        # Non-fast-forward races are rejected by Git. No force/ref deletions.
        remote_git(repo,'push','--atomic',remote,commit+':'+branch_ref,tag_ref+':'+tag_ref)
        refs=advertised(repo,remote,[branch_ref,tag_ref,tag_ref+'^{}'])
        if refs.get(branch_ref)!=commit or refs.get(tag_ref)!=tag_object or refs.get(tag_ref+'^{}')!=commit:
            raise ValueError('Remote publication verification failed')
    return refs


def publish(root, execute=False):
    r=verify_package(root)
    auto=root/'oellm-autoexp';mega=auto/SUBMODULE
    # Finish reachability of the dependency first. An interrupted second push is
    # safely repeatable; the consumer is never published with an unreachable pin.
    result={}
    for repo,key in ((mega,'megatron'),(auto,'autoexp')):
        result[key]=publish_repo(repo,r[key+'_remote'],r[key+'_branch'],r['name'],
            r[key]['commit'],r[key+'_tag_object'],execute)
    if execute:
        dump(root/'published.json',result)
    return result


def verify_remote(root):
    r=verify_package(root)
    with tempfile.TemporaryDirectory(prefix='megatron-release-verify-') as tmp:
        auto=Path(tmp)/'autoexp'
        subprocess.run(['git','-c','credential.helper=',
            '-c','credential.helper=!gh auth git-credential','clone','--no-checkout',
            '--single-branch','--branch',r['name'],r['autoexp_remote'],str(auto)],check=True)
        git(auto,'checkout','--detach',r['autoexp']['commit'])
        remote_git(auto,'submodule','update','--init','--',SUBMODULE)
        v=verify_checkout(auto,r['name'])
        if identity(auto)!=r['autoexp']:
            raise ValueError('Remote autoexp source mismatch')
        receipt=dict(autoexp=identity(auto),megatron=v['megatron'],fresh_remote_clone=True)
    dump(root/'remote-verification.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for key in ('autoexp','megatron','output'):q.add_argument('--'+key,type=Path,required=True)
    for key in ('name','autoexp-branch','megatron-branch','evidence'):q.add_argument('--'+key,required=True)
    q.add_argument('--autoexp-remote',default='https://github.com/OpenEuroLLM/oellm-autoexp.git')
    q.add_argument('--megatron-remote',default='https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM.git')
    q.add_argument('--execute',action='store_true')
    q.add_argument('--assisted-by',help='Optional attribution for agent-assisted preparation')
    for name in ('verify','publish','verify-remote'):
        q=sub.add_parser(name);q.add_argument('--package',type=Path,required=True)
        if name=='publish':q.add_argument('--execute',action='store_true')
    q=sub.add_parser('verify-checkout');q.add_argument('--autoexp',type=Path,default=Path.cwd());q.add_argument('--name',required=True)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a)
    elif a.command=='verify':result=verify_package(a.package)
    elif a.command=='publish':result=publish(a.package,a.execute)
    elif a.command=='verify-remote':result=verify_remote(a.package)
    else:result=verify_checkout(a.autoexp,a.name)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
