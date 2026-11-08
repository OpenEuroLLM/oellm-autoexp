"""Record frozen source hashes and the immutable input metadata identity."""
import hashlib
import json
from pathlib import Path

root = Path('/gpfs/projects/ehpc533/users/korbip/hybrid_v019')
snapshots = {'baseline': root/'autoexp_mhc_eval3700', 'candidate': root/'autoexp_hf_fast007'}
paths = ['oellm_autoexp/hf_export/fsdp_dtensor.py', 'oellm_autoexp/hf_export/fast_fsdp_reader.py', 'oellm_autoexp/hf_export/export_ambitious.py', 'oellm_autoexp/hf_export/score_on_flag.py', 'oellm_autoexp/hf_export/paired_valid_check.py', 'oellm_autoexp/hf_export/oellm_ambitious/modeling_oellm_ambitious.py', 'scripts/korbi/hf_export/export_main_run.sh']
result = {}
for label, snapshot in snapshots.items():
    result[label] = {'snapshot': str(snapshot), 'sync_stamp': (snapshot/'SYNC_STAMP').read_text(), 'sha256': {path: hashlib.sha256((snapshot/path).read_bytes()).hexdigest() for path in paths if (snapshot/path).exists()}}
meta = Path('/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph/ckpt_eval/iter_0003700/.metadata')
result['input'] = {'metadata': str(meta), 'metadata_bytes': meta.stat().st_size, 'metadata_sha256': hashlib.sha256(meta.read_bytes()).hexdigest()}
for path in paths[2:]:
    assert result['baseline']['sha256'][path] == result['candidate']['sha256'][path], path
candidate = snapshots['candidate']
result['v2_orchestration_sha256'] = {
    path: hashlib.sha256((candidate/path).read_bytes()).hexdigest()
    for path in [
        'experiment/007_fast_hf_export/scripts/export_main_run_v2.sh',
        'experiment/007_fast_hf_export/scripts/time_command.py',
        'scripts/korbi/hf_export/export_main_run_v2.sh',
        'scripts/korbi/hf_export/time_command.py',
        'config/experiments/korbi/production/hf_export_v2_mn5.yaml',
    ]
}
result['settings'] = {'expert_dtype': 'fp8', 'reader': 'fast', 'read_threads': 8,
    'v2_gpu_pairs': [[0,1],[2,3]],
    'container': str(root/'container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif')}
out = snapshots['candidate']/'experiment/007_fast_hf_export/results/provenance.json'
out.write_text(json.dumps(result, indent=2)+'\n')
print(out)
