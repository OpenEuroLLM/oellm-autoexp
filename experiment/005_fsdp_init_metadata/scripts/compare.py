"""Strict same-layout restore check plus resumed-step diagnostics; run on login node."""
import importlib.util
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('resume_compare', 'scripts/korbi/production/compare_resume_test.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
logs = sorted((root / 'cont').glob('slurm-*.log'), key=lambda p: p.stat().st_mtime)
text = logs[-1].read_text(errors='replace')
assert 'non-persistent iter 20' in text, 'Continuation did not select rolling iteration 20'
state_ok = m.compare_state(str(root), 'cont', 'opt')
ref, cont = m.read_log(str(root), 'ref'), m.read_log(str(root), 'cont')
expected = set(range(21, 31))
assert expected <= set(ref) and expected <= set(cont), 'Missing reference/continuation steps 21–30'
metrics = {}
for key in ['lm loss', 'grad norm']:
    deltas = [cont[s][key] - ref[s][key] for s in sorted(expected)]
    assert all(math.isfinite(x) for x in deltas), f'Nonfinite {key}'
    metrics[key] = {'max_abs_delta': max(map(abs, deltas)), 'rms_delta': math.sqrt(sum(x*x for x in deltas)/len(deltas))}
result = {'optimizer_state_matches': state_ok, 'restored_iteration': 20, 'resumed_steps': sorted(expected), 'metrics': metrics}
(root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
m.compare_losses(str(root), arms=('cont',))
assert state_ok, 'Saved/loaded optimizer or fp32 parameter state differs'
