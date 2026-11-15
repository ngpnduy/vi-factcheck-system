"""Publication integrity fixtures; these are not research training results."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_results', ROOT / 'scripts/export_results.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, 'TEST_COUNT', 3)
    run = tmp_path / 'original run'
    run.mkdir()
    save = exporter.write_json
    predictions = [{'id': f'test:{i}', 'gold_label_id': i, 'label_id': i,
                    'label': exporter.LABELS[i], 'article_id': f'article:{i}',
                    'probabilities': {label: float(j == i) for j, label in enumerate(exporter.LABELS)},
                    'ranking': [0, 1], 'alignment': {'status': 'valid', 'gold_sentence_ids': [0]}}
                   for i in range(3)]
    metric = {'classification': exporter.classification(predictions),
              'retrieval': exporter.retrieval(predictions), 'bm25_retrieval': exporter.retrieval(predictions)}
    table = ['model,seed,test_macro_f1,accuracy']
    for seed in exporter.SEEDS:
        directory = run / 'seeds' / f'seed{seed}'
        directory.mkdir(parents=True)
        save(directory / 'experiment.json', {'data_manifest': {'fixture': True}})
        for model in exporter.MODELS:
            stem = model.lower()
            (directory / f'{stem}_test_predictions.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in predictions), encoding='utf-8')
            save(directory / f'{stem}_test_metrics.json', metric)
            save(directory / f'{stem}_selection.json', {'selected_global_epoch': 4, 'protocol_version': 2, 'checkpoint_sha256': 'fixture-checkpoint'})
            names = [f'{stem}_test_predictions.jsonl', f'{stem}_test_metrics.json']
            save(directory / f'{stem}_test_complete.json', {
                'test_count': 3, 'reload_max_probability_difference': 0,
                'raw_input_max_probability_difference': 0, 'checkpoint_sha256': 'fixture-checkpoint',
                'artifact_sha256': {name: exporter.digest(directory / name) for name in names}})
            table.append(f'{model},{seed},1.0,1.0')
    (run / 'results.csv').write_text('\n'.join(table) + '\n', encoding='utf-8-sig')
    (run / 'REPORT.md').write_text('fixture only', encoding='utf-8')
    save(run / 'comparison.json', {'models': {m: {'test_macro_f1': {'mean': 1.0, 'sample_std': 0.0}} for m in exporter.MODELS}})
    save(run / 'paired_bootstrap.json', {f'seed{s}:{a}-{b}': {'claim_count': 3, 'article_count': 3,
         'metrics': {'macro_f1': {'difference': 0.0, 'ci95': [0.0, 0.0]}}}
         for s in exporter.SEEDS for a, b in exporter.PAIRS})
    source = run / 'source_snapshot/vifactcheck'
    source.mkdir(parents=True)
    (source / 'fixture.py').write_bytes(b'# exact fixture bytes\r\n')
    hashes = {'fixture.py': exporter.digest(source / 'fixture.py')}
    monkeypatch.setattr(exporter, 'EXPECTED_SOURCE_SHA256', hashes)
    save(run / 'full_plan.json', {'source_code_sha256': hashes})
    save(run / 'completion.json', {'status': 'complete', 'protocol_version': 2,
         'seeds': list(exporter.SEEDS), 'models': list(exporter.MODELS),
         'started_utc': '2026-11-06T05:12:06+00:00', 'completed_utc': '2026-11-11T17:43:31+00:00',
         'report_checksums': {name: exporter.digest(run / name) for name in ['REPORT.md', 'results.csv', 'comparison.json', 'paired_bootstrap.json']}})
    return run, tmp_path / 'compact', tmp_path / exporter.ARCHIVE_NAME


def test_export_is_verified_idempotent_and_excludes_weights(study):
    run, compact, archive = study
    (run / 'seeds/seed42/m0_best.pt').write_bytes(b'never publish weights')
    result = exporter.export(run, compact, archive)
    assert result['test_sets'] == 15 and result['claims_per_test'] == 3
    assert not list(compact.rglob('*.jsonl'))
    assert not list(compact.rglob('*.pt'))
    assert exporter.export(run, compact, archive) == result
    assert exporter.verify(compact, archive) == result
    assert (compact / 'REPORT.md').read_bytes() == (run / 'REPORT.md').read_bytes()


def test_changed_run_cannot_overwrite_existing_publication(study):
    run, compact, archive = study
    exporter.export(run, compact, archive)
    exporter.write_json(run / 'status.json', {'changed': True})
    with pytest.raises(ValueError, match='different study'):
        exporter.export(run, compact, archive)


def test_compact_and_archive_tampering_are_rejected(study):
    run, compact, archive = study
    exporter.export(run, compact, archive)
    original = (compact / 'REPORT.md').read_bytes()
    (compact / 'REPORT.md').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        exporter.verify(compact, archive)
    (compact / 'REPORT.md').write_bytes(original)
    with archive.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='ZIP checksum'):
        exporter.verify(compact, archive)


@pytest.mark.parametrize('name', ['../secret.json', '/absolute.json', 'C:/secret.json', 'a\\b.json', '.env', 'model.pt', '.cache/model.json'])
def test_unsafe_or_forbidden_payload_paths(name):
    with pytest.raises(ValueError):
        exporter.safe_name(name)


def test_pairing_and_duplicate_predictions_rejected(study):
    run, _, _ = study
    path = run / 'seeds/seed13/m1_test_predictions.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]['id'] = rows[0]['id']
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    marker_path = path.with_name('m1_test_complete.json')
    marker = exporter.read_json(marker_path)
    marker['artifact_sha256'][path.name] = exporter.digest(path)
    exporter.write_json(marker_path, marker)
    with pytest.raises(ValueError, match='duplicate claim'):
        exporter.verify_run(run)


def test_missing_bootstrap_is_rejected(study):
    run, _, _ = study
    path = run / 'paired_bootstrap.json'
    bootstrap = exporter.read_json(path)
    bootstrap.pop(next(iter(bootstrap)))
    exporter.write_json(path, bootstrap)
    completed = exporter.read_json(run / 'completion.json')
    completed['report_checksums'][path.name] = exporter.digest(path)
    exporter.write_json(run / 'completion.json', completed)
    with pytest.raises(ValueError, match='12 paired'):
        exporter.verify_run(run)


def test_wrong_aggregate_sample_sd_is_rejected(study):
    run, _, _ = study
    path = run / 'comparison.json'
    comparison = exporter.read_json(path)
    comparison['models']['M2']['test_macro_f1']['sample_std'] = 0.1
    exporter.write_json(path, comparison)
    completed = exporter.read_json(run / 'completion.json')
    completed['report_checksums'][path.name] = exporter.digest(path)
    exporter.write_json(run / 'completion.json', completed)
    with pytest.raises(ValueError, match='aggregate'):
        exporter.verify_run(run)


def test_different_executed_source_cannot_be_mislabeled(study):
    run, _, _ = study
    plan = exporter.read_json(run / 'full_plan.json')
    plan['source_code_sha256']['fixture.py'] = 'different source'
    exporter.write_json(run / 'full_plan.json', plan)
    with pytest.raises(ValueError, match='declared executed commit'):
        exporter.verify_run(run)
