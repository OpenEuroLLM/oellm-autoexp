#!/usr/bin/env python3
"""Create a portable, read-only Megatron run-context JSON document."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def first_existing(candidates):
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate.resolve())
    return None


def discover(run_root: Path, explicit, candidates):
    if explicit:
        return str(Path(explicit).resolve())
    return first_existing([run_root / candidate for candidate in candidates])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, help='Training run directory')
    parser.add_argument('--cluster', choices=('local', 'lumi', 'tensorwave'), default='local')
    parser.add_argument('--config')
    parser.add_argument('--tensorboard-dir')
    parser.add_argument('--log')
    parser.add_argument('--scheduler-id')
    parser.add_argument('--revision', default='unknown')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    if not run_root.is_dir():
        parser.error(f'--run-root is not a directory: {run_root}')
    context = {
        'schema_version': 1,
        'run': {
            'identifier': run_root.name,
            'root': str(run_root),
            'cluster': args.cluster,
            'scheduler_id': args.scheduler_id,
            'megatron_revision': args.revision,
        },
        'artifacts': {
            'config': discover(run_root, args.config, ('config.yaml', 'config.yml', 'config.json', 'args.json')),
            'tensorboard_dir': discover(run_root, args.tensorboard_dir, ('tensorboard', 'tb', 'events')),
            'log': discover(run_root, args.log, ('train.log', 'training.log', 'logs/train.log', 'stdout.log')),
        },
        'discovered_at': datetime.now(timezone.utc).isoformat(),
    }
    context['artifacts']['availability'] = {
        name: path is not None for name, path in context['artifacts'].items() if name != 'availability'
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(context, indent=2) + '\n')


if __name__ == '__main__':
    main()
