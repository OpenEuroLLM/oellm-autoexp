#!/usr/bin/env python3
"""Verify completed transfers, stage on MN5, then open the baseline gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parents[1]
LOCAL = Path('/tmp/ckda018/assets')
REMOTE = '/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018'
manifest = json.loads((HERE / 'results/source_manifest.json').read_text())
native = [f for f in manifest['files'] if f['role'] == 'native']
deadline = time.time() + 6 * 3600


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


for item in native:
    p = LOCAL / item['relative']
    while not p.exists() or p.stat().st_size != item['bytes']:
        if time.time() > deadline:
            raise TimeoutError(str(p))
        time.sleep(30)
    if sha(p) != item['sha256']:
        raise ValueError(f'Source checksum mismatch: {p}')
    (p.parents[2] / 'latest_checkpointed_iteration.txt').write_text('76294\n')
    subprocess.run(['rsync', '-az', '--partial', str(p.parents[2]) + '/',
                    f'marenostrum_new:{REMOTE}/assets/native/{item["arm"]}/'], check=True)
    print('Staged verified native checkpoint', item['arm'], flush=True)

subprocess.run(['rsync', '-aLz', str(LOCAL / 'tokenizer') + '/',
                f'marenostrum_new:{REMOTE}/assets/tokenizer/'], check=True)
subprocess.run(['rsync', '-a', str(HERE / 'results/source_manifest.json'),
                f'marenostrum_new:{REMOTE}/results/'], check=True)
code = 'import hashlib,json,time\nfrom pathlib import Path\n'
code += f'root=Path({REMOTE!r})\nitems={native!r}\n'
code += '''
for item in items:
    p=root/'assets'/item['relative']
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==item['sha256'], str(p)
assert (root/'results/probe_pass.json').is_file(), 'Compatibility probe has not passed'
tmp=root/'assets/ready.tmp'
tmp.write_text(json.dumps({'verified':True,'time':time.time()}))
tmp.replace(root/'assets/ready')
print('ASSETS_READY')
'''
subprocess.run(['ssh', '-o', 'BatchMode=yes', 'marenostrum_new', 'python3 -'],
               input=code, text=True, check=True)
