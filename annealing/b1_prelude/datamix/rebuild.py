"""Reproduce the bundled JSC datamixes from frozen original LUMI blends."""
import argparse,json,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent

def render(mix):
    fields=shlex.split((HERE.parent/'origins/mixtures'/f'datamix_lumi_multilingual1_{mix}.txt').read_text())
    mapping=json.loads((HERE/'source_map.json').read_text())
    if len(fields)!=588 or len(set(fields[1::2]))!=294:raise ValueError('Require complete294-prefix source blend')
    result=[]
    for weight,path in zip(fields[::2],fields[1::2]):result.extend((weight,mapping[path]))
    return ' '.join(result)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mix',choices=['dominant1','dominant2'],required=True);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    with args.output.open('x') as stream:stream.write(render(args.mix))
