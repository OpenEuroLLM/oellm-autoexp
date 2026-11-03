"""Report trajectory variation without treating one control as a statistical equivalence test."""
import importlib.util
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('resume_compare', 'scripts/korbi/production/compare_resume_test.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ref = m.read_log(str(root), 'ref')
result = {'note': 'Both continuations restore matching state summaries; subsequent training is not bitwise reproducible. One control does not establish trajectory equivalence.', 'arms': {}}
for arm in ['cont', 'cont_baseline']:
    assert m.compare_state(str(root), arm, 'opt'), f'{arm} restore mismatch'
    steps = m.read_log(str(root), arm)
    assert set(range(21, 31)) <= set(steps)
    entry = {'optimizer_state_matches': True, 'metrics': {}}
    for key in ['lm loss', 'grad norm']:
        ds = [steps[i][key] - ref[i][key] for i in range(21, 31)]
        assert all(math.isfinite(x) for x in ds)
        entry['metrics'][key] = {'max_abs_delta': max(map(abs, ds)), 'rms_delta': math.sqrt(sum(x*x for x in ds)/len(ds))}
    result['arms'][arm] = entry
(root / 'trajectory_control.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
m.compare_losses(str(root), arms=('cont', 'cont_baseline'))
