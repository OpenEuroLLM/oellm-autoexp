#!/usr/bin/env python3
"""Extract TensorBoard scalars into a portable JSON artifact."""

import argparse
import json
from pathlib import Path


def event_paths(root: Path):
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob('*') if path.name.startswith('events.out.tfevents.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Event file or TensorBoard directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tag', action='append', default=[], help='Exact scalar tag; repeatable')
    parser.add_argument('--prefix', action='append', default=[], help='Scalar tag prefix; repeatable')
    args = parser.parse_args()
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        parser.error(f'tensorboard package is required: {error}')

    paths = event_paths(Path(args.input))
    if not paths:
        parser.error('no TensorBoard event files found')
    values, available = {}, set()
    for path in paths:
        accumulator = EventAccumulator(str(path), size_guidance={'scalars': 0})
        accumulator.Reload()
        tags = accumulator.Tags().get('scalars', [])
        available.update(tags)
        selected = [
            tag for tag in tags
            if (not args.tag and not args.prefix) or tag in args.tag or any(tag.startswith(p) for p in args.prefix)
        ]
        for tag in selected:
            values.setdefault(tag, []).extend(
                {'step': event.step, 'wall_time': event.wall_time, 'value': event.value, 'source': str(path)}
                for event in accumulator.Scalars(tag)
            )
    for series in values.values():
        series.sort(key=lambda event: (event['step'], event['wall_time']))
    requested = set(args.tag)
    payload = {
        'schema_version': 1,
        'event_files': [str(path.resolve()) for path in paths],
        'scalars': values,
        'missing_tags': sorted(requested - available),
        'available_tags': sorted(available),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
