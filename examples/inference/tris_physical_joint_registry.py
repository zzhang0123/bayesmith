"""Build the D12 research record from completed artifacts and JUnit, not prose."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

RUNS = ['smoke8', 'optical8-smoke', 'metric8-smoke', 'marginal8-smoke', 'metricmarginal8-smoke']
CHECKS = ['bayes-regression-final', 'rho-regression', 'quadrature-fullenv']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, output):
    records = {}
    for name in RUNS:
        folder = root / name
        summary = json.loads((folder / 'summary.json').read_text())
        diagnostics = json.loads((folder / 'diagnostics.json').read_text())
        if sha(root / 'prepared8-v2' / 'data.npz') != summary['input_sha256']:
            raise ValueError(f'input hash mismatch for {name}')
        compact = {k: v for k, v in summary.items() if k not in ('diagnostics', 'nonfinite_diagnostic_paths')}
        compact['parameter_diagnostics'] = {
            k: {field: v.get(field) for field in ('r_hat_max', 'bulk_ess_min', 'tail_ess_min',
                                                 'ess_mean_min', 'mcse_mean_max')}
            for k, v in summary['diagnostics'].items() if isinstance(v, dict)}
        compact['nonfinite_metric_count'] = len(summary.get('nonfinite_diagnostic_paths', []))
        compact['original_execution_exit'] = int((root / (name + '.exit')).read_text().strip())
        compact['model_diagnostics'] = diagnostics
        compact['artifacts'] = {p.name: {'path': str(p.resolve()), 'sha256': sha(p)}
                                for p in [folder / 'summary.json', folder / 'samples.npz',
                                          folder / 'extras.npz', folder / 'diagnostics.json']}
        records[name] = compact
    validation = {}
    for name in CHECKS:
        folder = root / 'tests'
        xml_path = folder / (name + '.xml')
        suites = ET.parse(xml_path).getroot().iter('testsuite')
        counts = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.attrib.get(key, 0))
        counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
        counts['exit'] = int((folder / (name + '.exit')).read_text().strip())
        validation[name] = {**counts, 'junit': str(xml_path.resolve()), 'sha256': sha(xml_path)}
    for name in ('bayes-ruff', 'rho-ruff'):
        validation[name] = {'exit': int((root / 'tests' / (name + '.exit')).read_text().strip())}
    if any(v['exit'] != 0 for v in validation.values()):
        raise ValueError('a final validation command did not pass; record it before declaring completion')
    manifests = {name: json.loads((root / name / 'manifest.json').read_text())
                 for name in ('prepared8-v2', 'prepared16', 'prepared32')}
    injection = json.loads((root / 'injection8-v2.json').read_text())
    record = {
        'schema': 'tris.physical_joint_research_record.v1',
        'recorded_at': datetime.now(UTC).isoformat(),
        'task': 'T-001', 'scientific_certification': False,
        'physical_models': 1, 'completed_sampling_configurations': len(records),
        'completed_chains': sum(r['chains'] for r in records.values()),
        'retained_samples': sum(r['chains'] * r['draws'] for r in records.values()),
        'accepted_sampling_configurations': sum(r['model_diagnostics']['sampling_pass'] for r in records.values()),
        'runs': records,
        'aborted_run': {'name': 'baseline8', 'exit': int((root/'baseline8.exit').read_text().strip()),
                        'reason': 'persistent tree saturation after 800 warmup; stopped explicitly',
                        'saved_chains': len(list((root/'baseline8').glob('chain*.npz'))),
                        'log_sha256': sha(root/'baseline8.log')},
        'prepared_inputs': manifests, 'conditional_injection': injection, 'validation': validation,
        'prior_tests_not_duplicated': 'quadrature full-environment check overlaps one already executed test',
        'known_failed_attempts': [
            'Initial bayes regression: README 6936 vs measured 6940; fixed and final regression passed.',
            'Optical and old full-metric summaries: nonfinite ESS JSON serialization exit1; report-only recovery.',
            'One rheplicant collection refused by shared pytest gate; not a test failure; later collection passed.',
        ],
        'not_run': ['full nightly suites', 'posterior dust/scatter/noise sensitivity',
                    'NSIDE16/32 MCMC', 'curvature inference'],
        'source_urls': {
            'halpha': 'https://lambda.gsfc.nasa.gov/product/foreground/fg_halpha_get.html',
            'dust': 'https://lambda.gsfc.nasa.gov/product/foreground/fg_sfd_get.html',
            'halpha_model': 'https://arxiv.org/abs/astro-ph/0302024',
            'halpha_uncertainty': 'https://arxiv.org/abs/astro-ph/0301558',
            'cygnus_a_spectrum': 'https://arxiv.org/abs/1609.05940',
        },
        'papers_sha256': {p.name: sha(p) for p in (root/'sources').glob('*.pdf')},
        'registry_code_sha256': sha(Path(__file__)),
    }
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: record[k] for k in ('completed_chains', 'retained_samples',
                                             'accepted_sampling_configurations', 'validation')}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.root, args.output)


if __name__ == '__main__':
    main()
