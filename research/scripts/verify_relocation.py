"""Read-only real checkpoint/prediction equivalence check after moving the package."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from vifactcheck.cli import predict_record
from vifactcheck.data import load_split
from vifactcheck.io import read_json, read_jsonl, sha256, write_json
from vifactcheck.pilot import release_gpu
from vifactcheck.training import system_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--segmenter', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    plan = read_json(args.run / 'full_plan.json')
    source_hashes = {name: sha256(root / 'vifactcheck' / name) for name in plan['source_code_sha256']}
    if source_hashes != plan['source_code_sha256']:
        raise AssertionError('Executed Python source bytes changed')
    if not torch.cuda.is_available():
        raise RuntimeError('Use the validated CUDA environment for relocation equivalence')
    rows = {r['id']: r for r in load_split(args.data, 'test')}
    report = {'purpose': 'checkpoint inference equivalence only; no training or test-driven selection',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'package_root': str(root), 'source_code_sha256': source_hashes, 'cases': []}
    for seed in (42, 13, 2026):
        for mode in ('M0', 'M1', 'A1', 'M2', 'A2'):
            directory = args.run / 'seeds' / f'seed{seed}'
            path = directory / f'{mode.lower()}_best.pt'
            marker = read_json(directory / f'{mode.lower()}_test_complete.json')
            if sha256(path) != marker['checkpoint_sha256']:
                raise AssertionError(f'Checkpoint checksum changed: {seed}/{mode}')
            reference = read_jsonl(directory / f'{mode.lower()}_test_predictions.jsonl')[0]
            row = rows[reference['id']]
            system = system_from_checkpoint(path, 'cuda')
            with torch.no_grad():
                loaded = system.prediction(row)
                raw = predict_record(system, row['claim'], row['context'], str(args.segmenter))
            reload_difference = max(abs(loaded['probabilities'][k] - reference['probabilities'][k]) for k in reference['probabilities'])
            raw_difference = max(abs(raw['probabilities'][k] - reference['probabilities'][k]) for k in reference['probabilities'])
            if max(reload_difference, raw_difference) > 1e-6 or any(p['ranking'] != reference['ranking'] or p['label'] != reference['label'] for p in [loaded, raw]):
                raise AssertionError(f'Relocated prediction changed: {seed}/{mode}')
            record = {'seed': seed, 'model': mode, 'claim_id': row['id'],
                      'checkpoint_sha256': marker['checkpoint_sha256'],
                      'reload_max_probability_difference': reload_difference,
                      'raw_input_max_probability_difference': raw_difference,
                      'same_label_and_ranking': True}
            report['cases'].append(record)
            write_json(args.output, report)
            print(f'{seed}/{mode}: reload={reload_difference}, raw={raw_difference}', flush=True)
            del system
            release_gpu()
    report.update(status='passed', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output, report)


if __name__ == '__main__':
    main()
