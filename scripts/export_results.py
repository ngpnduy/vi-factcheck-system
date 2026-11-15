"""Export/verify the completed Listwise v2 study using only the standard library.

Original experiment files are copied byte-for-byte. Model weights stay local.
The archive manifest hashes payload files; the ZIP itself has an external digest.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import tempfile
import zipfile

STUDY = 'listwise-best5-v2-three-seeds-20261112'
TAG = 'listwise-v2-three-seeds-20261112'
EXECUTED_COMMIT = 'baa0bdc80379d26e661bd1dcf5e37b3b48ba14a8'
ARCHIVE_NAME = 'ViFactCheck-listwise-v2-three-seeds-results.zip'
SEEDS = (42, 13, 2026)
MODELS = ('M0', 'M1', 'A1', 'M2', 'A2')
LABELS = ('SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO')
PAIRS = (('A1', 'M0'), ('M1', 'A1'), ('M2', 'M1'), ('M2', 'A2'))
TEST_COUNT = 1447
MANIFEST = 'EXPORT_MANIFEST.json'
FORBIDDEN = {'.pt', '.pth', '.bin', '.safetensors', '.zip', '.log'}
FORBIDDEN_PARTS = {'.venv', '.cache', '__pycache__', '.git', '.env', 'data'}
ROOT_FILES = {'REPORT.md', 'results.csv', 'comparison.json', 'completion.json',
              'full_plan.json', 'learning_rate_trials.json', 'lambda_trials.json',
              'paired_bootstrap.json', 'resource_check.json', 'runtime_estimate.json',
              'selected_hyperparameters.json', 'status.json'}
EXPECTED_SOURCE_SHA256 = json.loads((Path(__file__).resolve().parents[1] /
    'docs/validation/listwise-v2-20261106.json').read_text(encoding='utf-8'))['final_source_sha256']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe artifact path: {name}')
    if path.suffix.lower() in FORBIDDEN or any(p in FORBIDDEN_PARTS for p in path.parts):
        raise ValueError(f'Forbidden artifact: {name}')
    return path


def assert_close(actual, expected, path='metric'):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert_close(actual[key], value, f'{path}.{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'{path}: different length')
        for index, value in enumerate(expected):
            assert_close(actual[index], value, f'{path}[{index}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError(f'{path}: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'{path}: {actual!r} != {expected!r}')


def classification(predictions):
    matrix = [[0] * 3 for _ in range(3)]
    for row in predictions:
        gold, predicted = row['gold_label_id'], row['label_id']
        if gold not in range(3) or predicted not in range(3) or row['label'] != LABELS[predicted]:
            raise ValueError('Invalid prediction label')
        probabilities = [row['probabilities'][label] for label in LABELS]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-5:
            raise ValueError('Invalid prediction probabilities')
        matrix[gold][predicted] += 1
    per_label = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support, predicted_count = sum(matrix[i]), sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {'count': len(predictions), 'accuracy': sum(matrix[i][i] for i in range(3)) / len(predictions),
            'macro_f1': statistics.mean(x['f1'] for x in per_label.values()),
            'per_label': per_label, 'confusion_matrix': matrix, 'label_order': list(LABELS)}


def retrieval(predictions, rankings=None):
    ranking_by_id = rankings or {p['id']: p['ranking'] for p in predictions}
    eligible = [p for p in predictions if p['gold_label_id'] != 2 and p['alignment']['status'] == 'valid'
                and p['alignment']['gold_sentence_ids']]
    result = {'count': len(eligible),
              'alignment_coverage': sum(p['alignment']['status'] == 'valid' for p in predictions) / len(predictions),
              'alignment_status_counts': dict(collections.Counter(p['alignment']['status'] for p in predictions))}
    for k in (5, 20):
        hits, recalls, complete = [], [], []
        for p in eligible:
            gold, found = set(p['alignment']['gold_sentence_ids']), set(ranking_by_id[p['id']][:k])
            hits.append(bool(gold & found))
            recalls.append(len(gold & found) / len(gold))
            complete.append(gold <= found)
        result[f'hit@{k}'] = statistics.mean(hits) if hits else None
        result[f'sentence_recall@{k}'] = statistics.mean(recalls) if recalls else None
        result[f'complete_gold@{k}'] = statistics.mean(complete) if complete else None
    reciprocal = [next((1 / (i + 1) for i, sid in enumerate(ranking_by_id[p['id']])
                        if sid in p['alignment']['gold_sentence_ids']), 0.0) for p in eligible]
    result['mrr'] = statistics.mean(reciprocal) if reciprocal else None
    return result


def verify_run(run):
    run = Path(run)
    completed, plan = read_json(run / 'completion.json'), read_json(run / 'full_plan.json')
    if completed['status'] != 'complete' or completed.get('protocol_version') != 2:
        raise ValueError('Requires a completed protocol-v2 run')
    if set(completed['seeds']) != set(SEEDS) or set(completed['models']) != set(MODELS):
        raise ValueError('Requires all five models on all three seeds')
    if plan['source_code_sha256'] != EXPECTED_SOURCE_SHA256:
        raise ValueError('Run source does not match the declared executed commit')
    for name, expected in completed['report_checksums'].items():
        safe_name(name)
        if digest(run / name) != expected:
            raise ValueError(f'Report checksum mismatch: {name}')
    for name, expected in plan['source_code_sha256'].items():
        safe_name(name)
        if digest(run / 'source_snapshot/vifactcheck' / name) != expected:
            raise ValueError(f'Executed source mismatch: {name}')
    comparison, bootstrap = read_json(run / 'comparison.json'), read_json(run / 'paired_bootstrap.json')
    with (run / 'results.csv').open(encoding='utf-8-sig', newline='') as stream:
        table = list(csv.DictReader(stream))
    if len(table) != len(SEEDS) * len(MODELS) or len({(r['model'], int(r['seed'])) for r in table}) != 15:
        raise ValueError('Expected 15 distinct model/seed result rows')
    baseline, per_seed, selections = None, {}, {}
    for seed in SEEDS:
        directory, per_seed[seed] = run / 'seeds' / f'seed{seed}', {}
        bm25 = None
        for model in MODELS:
            stem = model.lower()
            marker = read_json(directory / f'{stem}_test_complete.json')
            if marker['test_count'] != TEST_COUNT or max(marker['reload_max_probability_difference'], marker['raw_input_max_probability_difference']) > 1e-6:
                raise ValueError('Incomplete test or checkpoint prediction mismatch')
            for name, expected in marker['artifact_sha256'].items():
                safe_name(name)
                if digest(directory / name) != expected:
                    raise ValueError(f'Test artifact checksum mismatch: {name}')
            with (directory / f'{stem}_test_predictions.jsonl').open(encoding='utf-8') as stream:
                predictions = [json.loads(line) for line in stream if line.strip()]
            identity = {p['id']: (p['gold_label_id'], p['article_id']) for p in predictions}
            if len(predictions) != TEST_COUNT or len(identity) != TEST_COUNT:
                raise ValueError('Wrong prediction count or duplicate claim IDs')
            if baseline is None:
                baseline = identity
            if identity != baseline:
                raise ValueError('Unpaired claim IDs, labels or articles')
            if model == 'M0':
                bm25 = {p['id']: p['ranking'] for p in predictions}
            metric = read_json(directory / f'{stem}_test_metrics.json')
            calculated = classification(predictions)
            assert_close(calculated, metric['classification'], 'classification')
            assert_close(retrieval(predictions), metric['retrieval'], 'retrieval')
            assert_close(retrieval(predictions, bm25), metric['bm25_retrieval'], 'bm25_retrieval')
            record = next(r for r in table if r['model'] == model and int(r['seed']) == seed)
            assert_close(calculated['macro_f1'], float(record['test_macro_f1']))
            assert_close(calculated['accuracy'], float(record['accuracy']))
            selection = read_json(directory / f'{stem}_selection.json')
            if not 1 <= selection['selected_global_epoch'] <= 5 or selection['protocol_version'] != 2:
                raise ValueError('Invalid checkpoint selection')
            if selection['checkpoint_sha256'] != marker['checkpoint_sha256']:
                raise ValueError('Selected checkpoint hash differs from test checkpoint')
            per_seed[seed][model] = calculated
            selections[f'{seed}:{model}'] = selection
    for model in MODELS:
        scores = [per_seed[seed][model]['macro_f1'] for seed in SEEDS]
        assert_close({'mean': statistics.mean(scores), 'sample_std': statistics.stdev(scores)},
                     comparison['models'][model]['test_macro_f1'], 'aggregate')
    expected_keys = {f'seed{seed}:{a}-{b}' for seed in SEEDS for a, b in PAIRS}
    if set(bootstrap) != expected_keys:
        raise ValueError('Expected 12 paired article-bootstrap records')
    for seed in SEEDS:
        for a, b in PAIRS:
            record = bootstrap[f'seed{seed}:{a}-{b}']
            if record['claim_count'] != TEST_COUNT or record['article_count'] != len({v[1] for v in baseline.values()}):
                raise ValueError('Bootstrap denominator mismatch')
            difference = per_seed[seed][a]['macro_f1'] - per_seed[seed][b]['macro_f1']
            assert_close(difference, record['metrics']['macro_f1']['difference'])
            bounds = record['metrics']['macro_f1']['ci95']
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError('Invalid bootstrap interval')
    return {'test_sets': 15, 'claims_per_test': TEST_COUNT, 'paired_ids_labels_articles': True,
            'classification_and_retrieval_recomputed': True, 'mean_and_sample_sd_verified': True,
            'paired_bootstrap_records': 12, 'checkpoint_selections': selections}


def payload_files(run):
    result = {}
    for path in sorted(Path(run).rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(run).as_posix()
        parts = PurePosixPath(relative).parts
        if len(parts) == 1 and relative not in ROOT_FILES:
            continue
        if len(parts) > 1 and parts[0] not in {'seeds', 'tuning', 'source_snapshot'}:
            continue
        if path.suffix.lower() not in {'.json', '.jsonl', '.csv', '.md', '.py'}:
            continue
        safe_name(relative)
        if path.is_symlink() or not path.resolve().is_relative_to(Path(run).resolve()):
            raise ValueError(f'Artifact escaped run directory: {relative}')
        result[relative] = path
    return result


def publication_readme(comparison, completed):
    url = f'https://github.com/ngpnduy/vi-factcheck-system/releases/download/{TAG}'
    score_rows = '\n'.join(f"| {model} | {comparison['models'][model]['test_macro_f1']['mean']:.4f} +/- {comparison['models'][model]['test_macro_f1']['sample_std']:.4f} |" for model in MODELS)
    from datetime import datetime, timedelta, timezone
    completed_date = datetime.fromisoformat(completed['completed_utc']).astimezone(timezone(timedelta(hours=7))).strftime('%d %B %Y')
    wall_hours = (datetime.fromisoformat(completed['completed_utc']) - datetime.fromisoformat(completed['started_utc'])).total_seconds() / 3600
    return f'''# Listwise v2: five models, three seeds

Completed {completed_date} (Vietnam time), protocol 2. All 5,062 train / 723 dev /
1,447 test claims were used, with seeds 42, 13 and 2026. Validation-only tuning
selected LR 1e-5 and lambda 0.1. See [the unchanged report](REPORT.md) and
[result table](results.csv) for means and sample SD, per-seed scores, selected
epochs, retrieval denominators, memory, runtime and paired article bootstrap.

| Model | Test Macro-F1, mean +/- sample SD |
|---|---:|
{score_rows}

## Download and verification

- [Full results ZIP]({url}/{ARCHIVE_NAME})
- [ZIP SHA-256]({url}/{ARCHIVE_NAME}.sha256)
- [GitHub Release](https://github.com/ngpnduy/vi-factcheck-system/releases/tag/{TAG})

Git contains compact reports, metrics, histories and provenance. The ZIP also
contains all predictions, original experiment configurations and manifests,
tuning artifacts, and the exact executed source snapshot. Weights and datasets
are not included. References to local weight paths in original manifests are
historical records, not downloadable files. Original absolute paths and file
bytes are preserved for hash verification; supply your own paths when running.

From the repository's `research/` directory:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\export_results.py verify --output results\\{STUDY} --archive <downloaded-zip>
```

`EXPORT_MANIFEST.json` hashes the original archive payload and compact files.
The ZIP digest is external to avoid a self-referential checksum. Verification
recomputes classification/retrieval and three-seed aggregates from predictions,
checks all 15 x 1,447 claims, test hashes, selection hashes and 12 bootstrap
records. It does not load weights or start training.

## Provenance and reproduction

Executed source: `{EXECUTED_COMMIT}`. The study tag identifies the later
publication commit, not the training commit. `PROVENANCE.json` maps the former
`rush/vifactcheck/` location to `research/vifactcheck/`. The package/CLI stays
`vifactcheck`; all 15 executed Python files retain their exact original bytes.
Use `REPRODUCTION_REQUIREMENTS.txt` and the current research setup/protocol guide.
Published October studies remain separate at their original `rush/results/` URLs.
The recorded {wall_hours:.2f}-hour wall time includes interruptions and repeated work.
'''


def export(run, output, archive, validation=None):
    run, output, archive = Path(run).resolve(), Path(output).resolve(), Path(archive).resolve()
    if output.is_relative_to(run) or archive.is_relative_to(run) or archive.is_relative_to(output):
        raise ValueError('Publication destinations must be outside the original run and separate from each other')
    verification = verify_run(run)
    files = payload_files(run)
    original_hashes = {name: digest(path) for name, path in files.items()}
    if (output / MANIFEST).exists():
        previous = read_json(output / MANIFEST)
        if previous['original_file_sha256'] != original_hashes:
            raise ValueError('Refusing to overwrite a different study or changed run')
        return verify(output, archive)
    if output.exists() and any(output.iterdir()) or archive.exists():
        raise ValueError('Refusing to overwrite existing publication files')
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vifactcheck-export-', dir=output.parent) as temp:
        staging = Path(temp)
        compact = staging / 'compact'
        compact.mkdir()
        for name, source in files.items():
            if source.suffix == '.jsonl' or name.startswith('source_snapshot/') or source.name == 'experiment.json':
                continue
            target = compact / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (compact / 'README.md').write_text(publication_readme(read_json(run / 'comparison.json'), read_json(run / 'completion.json')), encoding='utf-8')
        write_json(compact / 'PROVENANCE.json', {
            'study': STUDY, 'executed_source_commit': EXECUTED_COMMIT, 'publication_tag': TAG,
            'source_path_mapping': {'rush/vifactcheck/': 'research/vifactcheck/'},
            'source_code_sha256': read_json(run / 'full_plan.json')['source_code_sha256'],
            'original_completion_sha256': digest(run / 'completion.json'),
            'dataset_and_model_revisions': read_json(run / 'seeds/seed42/experiment.json')['data_manifest'],
            'verification': verification,
            'excluded': ['model weights', 'environments', 'download caches', 'datasets', 'logs', 'temporary files'],
        })
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'requirements-tested.txt', compact / 'REPRODUCTION_REQUIREMENTS.txt')
        if validation:
            shutil.copyfile(validation, compact / 'PUBLICATION_VALIDATION.json')
        for name in ['README.md', 'PROVENANCE.json', 'REPRODUCTION_REQUIREMENTS.txt', 'PUBLICATION_VALIDATION.json']:
            if (compact / name).exists():
                files[name] = compact / name
        compact_hashes = {p.relative_to(compact).as_posix(): digest(p) for p in compact.rglob('*') if p.is_file()}
        manifest = {'format_version': 1, 'study': STUDY, 'executed_source_commit': EXECUTED_COMMIT,
                    'publication_tag': TAG, 'archive_name': archive.name,
                    'original_file_sha256': original_hashes,
                    'archive_file_sha256': {name: digest(path) for name, path in files.items()},
                    'compact_file_sha256': compact_hashes, 'verification': verification}
        write_json(compact / MANIFEST, manifest)
        temporary_zip = staging / 'archive.zip'
        with zipfile.ZipFile(temporary_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
            for name, source in sorted({**files, MANIFEST: compact / MANIFEST}.items()):
                safe_name(name)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                target.writestr(info, source.read_bytes(), compresslevel=9)
        checksum = digest(temporary_zip)
        sidecar = compact / f'{archive.name}.sha256'
        sidecar.write_text(f'{checksum}  {archive.name}\n', encoding='ascii')
        if output.exists():
            output.rmdir()  # Only an already-checked empty destination.
        compact.rename(output)
        shutil.move(str(temporary_zip), archive)
        shutil.copyfile(output / sidecar.name, archive.with_name(sidecar.name))
    return verify(output, archive)


def verify(output, archive):
    output, archive = Path(output), Path(archive)
    manifest = read_json(output / MANIFEST)
    if manifest['study'] != STUDY or manifest['archive_name'] != archive.name:
        raise ValueError('Publication identity mismatch')
    for name, expected in manifest['compact_file_sha256'].items():
        safe_name(name)
        if digest(output / name) != expected:
            raise ValueError(f'Compact file checksum mismatch: {name}')
    allowed_compact = set(manifest['compact_file_sha256']) | {MANIFEST, archive.name + '.sha256'}
    if {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()} != allowed_compact:
        raise ValueError('Unexpected or missing compact publication file')
    expected_zip = (output / (archive.name + '.sha256')).read_text(encoding='ascii').split()[0]
    if digest(archive) != expected_zip:
        raise ValueError('ZIP checksum mismatch')
    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None or len(source.namelist()) != len(set(source.namelist())):
            raise ValueError('Invalid ZIP CRC or duplicate entries')
        if set(source.namelist()) != set(manifest['archive_file_sha256']) | {MANIFEST}:
            raise ValueError('Archive inventory mismatch')
        with tempfile.TemporaryDirectory(prefix='vifactcheck-verify-', dir=output.parent) as temp:
            extracted = Path(temp).resolve()
            for info in source.infolist():
                relative = safe_name(info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive contains a symbolic link')
                content = source.read(info)
                if info.filename == MANIFEST:
                    if json.loads(content) != manifest:
                        raise ValueError('Archive manifest differs from compact manifest')
                elif hashlib.sha256(content).hexdigest() != manifest['archive_file_sha256'][info.filename]:
                    raise ValueError(f'Archive payload checksum mismatch: {info.filename}')
                target = extracted.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(extracted):
                    raise ValueError('Archive path escaped extraction root')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            result = verify_run(extracted)
    return {'status': 'verified', 'study': STUDY, 'test_sets': result['test_sets'],
            'claims_per_test': result['claims_per_test'], 'paired_bootstrap_records': 12,
            'zip_sha256': expected_zip, 'archive_bytes': archive.stat().st_size,
            'archive_files': len(manifest['archive_file_sha256']) + 1,
            'compact_files': len(allowed_compact)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('export', help='Verify the completed run and create compact files plus a ZIP')
    create.add_argument('--run', required=True, type=Path)
    create.add_argument('--validation', type=Path)
    for child in (create, sub.add_parser('verify', help='Verify the exported files and ZIP offline')):
        child.add_argument('--output', required=True, type=Path)
        child.add_argument('--archive', required=True, type=Path)
    args = parser.parse_args()
    result = export(args.run, args.output, args.archive, args.validation) if args.command == 'export' else verify(args.output, args.archive)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
