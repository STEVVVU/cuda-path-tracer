"""Recorded-input controls for the existing curator; no CUDA work."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('summarize', ROOT / 'tools/summarize-benchmark.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def recorded_input(filename):
    workload = json.loads((ROOT / 'docs/evidence' / filename).read_text())['workloads'][0]
    return {
        'context': workload['context'],
        'benchmarks': [row for fixture in workload['fixtures'] for row in fixture['raw_repetitions']]
        + [{'run_type': 'aggregate', 'run_name': fixture['name'], 'aggregate_name': 'median',
            'real_time': fixture['p50_framework_ms']} for fixture in workload['fixtures']],
    }


class CuratorControls(unittest.TestCase):
    def setUp(self):
        self.raw = recorded_input('metrics-raycast-sync-summary.json')

    def test_recorded_nine_and_thirteen_groups(self):
        original = recorded_input('metrics-summary.json')
        self.assertEqual(len(MODULE.fixtures_from_raw(original)), 9)
        self.assertEqual(len(MODULE.fixtures_from_raw(self.raw, True)), 13)
        with self.assertRaises(ValueError):
            MODULE.fixtures_from_raw(self.raw)

    def test_missing_candidate(self):
        self.raw['benchmarks'] = [row for row in self.raw['benchmarks'] if not row['run_name'].startswith('compact/candidate/wall/')]
        with self.assertRaises(ValueError):
            MODULE.fixtures_from_raw(self.raw, True)

    def test_duplicated_candidate_repetition(self):
        rows = [row for row in self.raw['benchmarks'] if row['run_type'] == 'iteration' and row['run_name'].startswith('compact/candidate/wall/')]
        rows[1]['repetition_index'] = rows[0]['repetition_index']
        with self.assertRaises(ValueError):
            MODULE.fixtures_from_raw(self.raw, True)

    def test_duplicate_candidate_cannot_replace_control(self):
        for row in self.raw['benchmarks']:
            if row['run_name'].startswith('bvh/wall/'):
                row['run_name'] = row['run_name'].replace('bvh/wall/1', 'compact/candidate/wall/2')
        with self.assertRaises(ValueError):
            MODULE.fixtures_from_raw(self.raw, True)

    def test_wrong_candidate_policy(self):
        raw = copy.deepcopy(self.raw)
        next(row for row in raw['benchmarks'] if row['run_type'] == 'iteration' and '/candidate/' in row['run_name'])['wait_after_raycast'] = 1
        with self.assertRaises(ValueError):
            MODULE.fixtures_from_raw(raw, True)


if __name__ == '__main__':
    unittest.main()
