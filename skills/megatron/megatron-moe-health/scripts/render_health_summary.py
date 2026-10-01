#!/usr/bin/env python3
"""Render an evidence-first MoE health summary from normalized scalar JSON."""

import argparse
import json
from pathlib import Path


def last_value(scalars, tag):
    series = scalars.get(tag, [])
    return series[-1]['value'] if series else None


def finding(category, state, evidence, recommendation):
    return {'category': category, 'state': state, 'confidence': 'low' if not evidence else 'medium',
            'evidence': evidence, 'recommended_next_action': recommendation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalars', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.scalars).read_text())
    scalars = payload.get('scalars', {})
    entropy = last_value(scalars, 'moe/dispatched_expert_load_entropy_mean')
    collapsed = last_value(scalars, 'moe/expert_weight_collapsed_frac_max')
    ratio = last_value(scalars, 'moe/routed_expert_output_to_layer_output_rms_min')
    findings = []
    if entropy is None:
        findings.append(finding('routing', 'insufficient_evidence', [], 'Extract dispatched-load metrics.'))
    else:
        state = 'imbalanced' if entropy < 0.6 else 'healthy'
        findings.append(finding('routing', state, [{'metric': 'moe/dispatched_expert_load_entropy_mean', 'value': entropy}], 'Inspect dead and near-dead expert counts.'))
    if collapsed is None:
        findings.append(finding('parameters', 'insufficient_evidence', [], 'Enable expert viability metrics.'))
    else:
        state = 'collapsed' if collapsed > 0 else 'healthy'
        findings.append(finding('parameters', state, [{'metric': 'moe/expert_weight_collapsed_frac_max', 'value': collapsed}], 'Compare weight and gradient RMS by layer.'))
    if ratio is None:
        findings.append(finding('contribution', 'insufficient_evidence', [], 'Enable routed-output viability metrics.'))
    else:
        state = 'small_contribution' if ratio < 0.05 else 'healthy'
        findings.append(finding('contribution', state, [{'metric': 'moe/routed_expert_output_to_layer_output_rms_min', 'value': ratio}], 'Use paired masked validation before concluding functional silence.'))
    output = {'schema_version': 1, 'source': args.scalars, 'findings': findings, 'missing_tags': payload.get('missing_tags', [])}
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
