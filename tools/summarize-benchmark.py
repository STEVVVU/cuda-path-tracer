"""Validate and curate existing Google Benchmark JSON; never run a benchmark."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def fixtures_from_raw(raw, compare_raycast_sync=False):
    if raw['context'].get('dry_run') == 'true' or raw['context']['library_build_type'] != 'release':
        raise ValueError('Dry run or non-Release result')
    expected = {f'{backend}/{clock}/{mode}' for mode, backend in enumerate(['naive', 'bvh', 'compact', 'grouped']) for clock in ['wall', 'event_span']}
    expected.add('raycast/kernel_event')
    if compare_raycast_sync:
        expected.update(f'{backend}/candidate/{clock}/{mode}' for mode, backend in [(2, 'compact'), (3, 'grouped')] for clock in ['wall', 'event_span'])
    groups = {}
    for row in raw['benchmarks']:
        if row.get('error_occurred'):
            raise ValueError('Framework reported failed work')
        if row['run_type'] == 'iteration':
            if row.get('oracle_pass') != 1 or row['iterations'] <= 0 or not math.isfinite(row['real_time']) or row['real_time'] <= 0 or row['time_unit'] != 'ms':
                raise ValueError('Incomplete or invalid work')
            if compare_raycast_sync and not row['run_name'].startswith('raycast/') and row.get('wait_after_raycast') != (0 if '/candidate/' in row['run_name'] else 1):
                raise ValueError('Mislabeled wait policy')
            groups.setdefault(row['run_name'], []).append(row)
    if {name.split('/min_time:')[0] for name in groups} != expected or len(groups) != len(expected) or any(len(rows) != 7 for rows in groups.values()):
        raise ValueError(f'Expected exact {len(expected)} fixtures and seven repetitions')
    if any({row['repetition_index'] for row in rows} != set(range(7)) for rows in groups.values()):
        raise ValueError('Duplicated or missing framework repetition')
    fixtures = []
    for name, rows in groups.items():
        medians = [row['real_time'] for row in raw['benchmarks'] if row.get('run_name') == name and row.get('aggregate_name') == 'median']
        if len(medians) != 1 or not math.isfinite(medians[0]):
            raise ValueError('Missing or duplicated framework median')
        values = [row['real_time'] for row in rows]
        if not min(values) <= medians[0] <= max(values):
            raise ValueError('Framework median outside repetitions')
        fixtures.append({'name': name, 'p50_framework_ms': medians[0], 'p95_nearest_rank_ms': max(values), 'range_ms': [min(values), max(values)], 'raw_repetitions': rows})
    return fixtures


def comparisons(fixtures):
    indexed = {fixture['name'].split('/min_time:')[0]: fixture for fixture in fixtures}
    result = []
    for mode, backend in [(2, 'compact'), (3, 'grouped')]:
        for clock in ['wall', 'event_span']:
            base = indexed[f'{backend}/{clock}/{mode}']
            candidate = indexed[f'{backend}/candidate/{clock}/{mode}']
            for counter in ['managed_bytes', 'launches_per_render']:
                if len({row[counter] for fixture in [base, candidate] for row in fixture['raw_repetitions']}) != 1:
                    raise ValueError('Candidate changed allocation or kernel launches')
            relation = 'overlap'
            if candidate['range_ms'][1] < base['range_ms'][0]:
                relation = 'candidate_below_baseline'
            elif candidate['range_ms'][0] > base['range_ms'][1]:
                relation = 'candidate_above_baseline'
            result.append({'backend': backend, 'metric': clock, 'baseline': base['name'], 'candidate': candidate['name'], 'median_change_percent': 100 * (candidate['p50_framework_ms'] / base['p50_framework_ms'] - 1), 'range_relation': relation})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare-raycast-sync', action='store_true')
    args = parser.parse_args()
    args.input = args.input or Path('evidence/google-benchmark-raycast-sync' if args.compare_raycast_sync else 'evidence/google-benchmark')
    args.output = args.output or Path('docs/evidence/metrics-raycast-sync-summary.json' if args.compare_raycast_sync else 'docs/evidence/metrics-summary.json')
    cases = ['room_360p_4spp', 'room_360p_16spp', 'lattice_360p_4spp', 'lattice_360p_16spp', 'dense_360p_4spp', 'dense_360p_16spp', 'room_720p_4spp']
    result = {'framework': 'Google Benchmark v1.9.5', 'scope': 'Framework repetition means, not individual frame percentiles; render events include host scheduling and managed-memory gaps. Raycast events bracket one kernel.', 'workloads': [], 'raw_sha256': {}}
    for case in cases:
        path = args.input / (case + '.Measure.json')
        raw = json.loads(path.read_text())
        fixtures = fixtures_from_raw(raw, args.compare_raycast_sync)
        workload = {'name': case, 'context': raw['context'], 'fixtures': fixtures}
        if args.compare_raycast_sync:
            workload['comparisons'] = comparisons(fixtures)
        result['workloads'].append(workload)
        result['raw_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.compare_raycast_sync:
        manifest = (args.input / 'hashes.txt').read_text()
        identities = {Path(path.strip()).name: digest.lower() for digest, path in re.findall(r'Hash\s*:\s*([0-9A-Fa-f]{64})\s+Path\s*:\s*([^\r\n]+)', manifest)}
        for workload in result['workloads']:
            context = workload['context']
            if identities.get('wavefront.cu') != context['source_sha256'] or identities.get('benchmark_main.cu') != context['adapter_sha256'] or workload['name'] + '.exe' not in identities:
                raise ValueError('Missing or inconsistent recorded build identity')
        result['mode'] = 'raycast-wait A/B; production default remains synchronized'
        result['recorded_build_sha256'] = identities
        result['recorded_base_commit'] = (args.input / 'source-sha.txt').read_text().strip()
        result['recorded_hardware_csv'] = (args.input / 'hardware-Measure.csv').read_text().strip()
        result['measured_oracle_rows'] = sum(len(fixture['raw_repetitions']) for workload in result['workloads'] for fixture in workload['fixtures'])
        result['measured_image_oracle_rows'] = sum(len(fixture['raw_repetitions']) for workload in result['workloads'] for fixture in workload['fixtures'] if not fixture['name'].startswith('raycast/'))
        for path in sorted(args.input.iterdir()):
            if path.is_file():
                result['raw_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
