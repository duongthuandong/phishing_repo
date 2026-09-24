import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline')
SOURCE_TEST_CSV = ROOT / 'data' / 'splits' / 'test.csv'
EXP_ROOT = ROOT / 'experiments' / 'topk_1000_balanced'
SAMPLE_DIR = EXP_ROOT / 'data' / 'sample'
DEFAULT_OUTPUT = SAMPLE_DIR / 'test_1000_balanced_seed42.csv'
DEFAULT_MANIFEST = SAMPLE_DIR / 'sample_manifest.json'


def parse_args():
    p = argparse.ArgumentParser(description='Sample a fixed balanced 1,000-page test subset for Top-K experiments.')
    p.add_argument('--per-class', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main():
    args = parse_args()
    if args.per_class <= 0:
        raise ValueError('--per-class must be > 0')

    rows = []
    with SOURCE_TEST_CSV.open('r', newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows.append({'filename': row['filename'], 'label': int(row['label'])})

    by_label = {0: [], 1: []}
    for row in rows:
        if row['label'] not in by_label:
            raise ValueError(f"Unexpected label: {row['label']}")
        by_label[row['label']].append(row)

    for label in (0, 1):
        if len(by_label[label]) < args.per_class:
            raise RuntimeError(
                f'Not enough label={label}: need {args.per_class}, have {len(by_label[label])}'
            )

    rng = random.Random(args.seed)
    selected = rng.sample(by_label[0], args.per_class) + rng.sample(by_label[1], args.per_class)
    rng.shuffle(selected)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    output = SAMPLE_DIR / f'test_{2 * args.per_class}_balanced_seed{args.seed}.csv'
    manifest_path = SAMPLE_DIR / f'sample_manifest_{2 * args.per_class}_seed{args.seed}.json'

    signature = {
        'source_test_csv': str(SOURCE_TEST_CSV),
        'source_test_csv_size': SOURCE_TEST_CSV.stat().st_size,
        'seed': args.seed,
        'per_class': args.per_class,
        'num_pages': len(selected),
        'source_population': {'benign': len(by_label[0]), 'phishing': len(by_label[1])},
    }

    if output.exists():
        with output.open('r', newline='', encoding='utf-8') as f:
            old = [{'filename': r['filename'], 'label': int(r['label'])} for r in csv.DictReader(f)]
        if old != selected:
            raise RuntimeError(f'{output} already exists with different sample; refusing to overwrite.')
        print(f'Reuse existing sample: {output}')
    else:
        with output.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['filename', 'label'])
            w.writeheader()
            w.writerows(selected)
        print(f'Saved sample: {output}')

    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old != signature:
            raise RuntimeError(f'{manifest_path} already exists with different signature.')
    else:
        atomic_json(manifest_path, signature)

    print(json.dumps(signature, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
