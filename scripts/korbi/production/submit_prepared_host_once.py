#!/usr/bin/env python3
"""Submit exactly one NEW held host per prepared pool, without a recurring monitor.

No cancellation, updates, requeue, or retries. A reservation marker prevents
resubmission if sbatch succeeds but the reply or subsequent state write fails.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from oellm_autoexp.monitor.loop import JobFileStore
from oellm_autoexp.monitor.pool import PoolLayout, PoolManager, pool_tag
from oellm_autoexp.monitor.slurm_client import SlurmClient, SlurmClientConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pools', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.pools:
        layout = PoolLayout(path)
        cfg = layout.load_config()
        assert cfg.hold_hosts and not cfg.handover and cfg.max_own_hosts == 1
        assert path.name.startswith('mn5_pool_f72_'), 'Only new future72 pools are allowed'
        meta = layout.load_meta()
        store = JobFileStore(meta['session_dir'])
        marker = path / 'manual_submission_started.json'
        if marker.exists():
            raise SystemExit(f'{path}: submission already attempted; inspect manually, never retry automatically')
        assert not list(layout.hosts_dir.glob('host*')), 'Pool already has hosts'
        hid = PoolManager()._create_host(SimpleNamespace(_store=store), layout, cfg)
        record = store.load(f'{pool_tag(cfg.name)}.{hid}')
        assert record is not None and not record.runtime.submitted
        marker.write_text(json.dumps(dict(pool=str(path), host=hid, policy='submit once; never alter existing Slurm jobs')))
        job_id = SlurmClient(SlurmClientConfig()).submit(record.definition)
        record.runtime.runtime_job_id = job_id
        record.runtime.submitted = True
        record.runtime.last_status = 'PENDING'
        store.upsert(record)
        result = dict(pool=str(path), host=hid, job_id=job_id, nodes=record.definition.slurm.sbatch.nodes,
                      time=record.definition.slurm.sbatch.time, account=record.definition.slurm.sbatch.account,
                      recurring_monitor=False, hold_hosts=True)
        (path / 'manual_submission_result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
