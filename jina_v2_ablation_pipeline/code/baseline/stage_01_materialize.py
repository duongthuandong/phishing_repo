import csv
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path('/content/drive/MyDrive/html_token_analysis')
ORIGINAL_DATA = ROOT / 'jina_v2_pipeline' / 'data'
ABLATION_DATA = ROOT / 'jina_v2_ablation_pipeline' / 'data'
SPLITS_DIR = ABLATION_DATA / 'splits'
OUT_DIR = ABLATION_DATA / 'ablations' / 'baseline'
PARTS_DIR = OUT_DIR / '_materialize_parts'
CHUNKS_OUT_DIR = OUT_DIR / 'chunks'
EMBED_OUT_DIR = OUT_DIR / 'embeddings'
FINAL_MANIFEST = OUT_DIR / 'materialize_manifest.json'

EXPECTED = {
    'train': {'pages': 2000, 'label_0': 1000, 'label_1': 1000},
    'test': {'pages': 500, 'label_0': 250, 'label_1': 250},
}

SOURCES = (
    (
        'original_train',
        ORIGINAL_DATA / 'chunks' / 'train_chunks.jsonl.gz',
        ORIGINAL_DATA / 'embeddings' / 'train_embeddings.npy',
    ),
    (
        'original_test',
        ORIGINAL_DATA / 'chunks' / 'test_chunks.jsonl.gz',
        ORIGINAL_DATA / 'embeddings' / 'test_embeddings.npy',
    ),
)

PROGRESS_EVERY_ROWS = 100_000


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_targets() -> tuple[dict, dict]:
    assignment = {}
    split_stats = {}

    for split in ('train', 'test'):
        csv_path = SPLITS_DIR / f'{split}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f'Missing split: {csv_path}')

        rows = []
        with csv_path.open('r', newline='', encoding='utf-8-sig') as f:
            for order, row in enumerate(csv.DictReader(f)):
                filename = row['filename']
                label = int(row['label'])
                url = row.get('url', '')
                if filename in assignment:
                    raise ValueError(f'Duplicate filename across target splits: {filename}')
                assignment[filename] = {
                    'split': split,
                    'label': label,
                    'url': url,
                    'order': order,
                }
                rows.append((filename, label, url))

        stats = {
            'pages': len(rows),
            'label_0': sum(label == 0 for _, label, _ in rows),
            'label_1': sum(label == 1 for _, label, _ in rows),
            'sha256': sha256_file(csv_path),
        }
        split_stats[split] = stats

        exp = EXPECTED[split]
        for key in ('pages', 'label_0', 'label_1'):
            if stats[key] != exp[key]:
                raise ValueError(
                    f'{split}.csv unexpected {key}: {stats[key]} != {exp[key]}'
                )

    if len(assignment) != 2500:
        raise ValueError(f'Expected 2500 unique pages, got {len(assignment)}')

    return assignment, split_stats


def source_signature(chunks_path: Path, embeddings_path: Path) -> dict:
    return {
        'chunks_path': str(chunks_path),
        'chunks_size': chunks_path.stat().st_size,
        'embeddings_path': str(embeddings_path),
        'embeddings_size': embeddings_path.stat().st_size,
    }


def part_paths(source_name: str, target_split: str) -> tuple[Path, Path]:
    return (
        PARTS_DIR / f'{source_name}__to_{target_split}.jsonl.gz',
        PARTS_DIR / f'{source_name}__to_{target_split}.npy',
    )


def source_manifest_path(source_name: str) -> Path:
    return PARTS_DIR / f'{source_name}.manifest.json'


def source_part_is_valid(source_name: str, signature: dict, split_stats: dict) -> bool:
    manifest_path = source_manifest_path(source_name)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception:
        return False

    if manifest.get('source_signature') != signature:
        return False
    if manifest.get('target_split_sha256') != {
        k: v['sha256'] for k, v in split_stats.items()
    }:
        return False

    for split in ('train', 'test'):
        c, e = part_paths(source_name, split)
        if not c.exists() or not e.exists():
            return False
        try:
            arr = np.load(e, mmap_mode='r')
        except Exception:
            return False
        expected_rows = int(manifest['selected_chunks'][split])
        if len(arr) != expected_rows:
            return False
    return True


def process_source(
    source_name: str,
    chunks_path: Path,
    embeddings_path: Path,
    assignment: dict,
    split_stats: dict,
) -> dict:
    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    signature = source_signature(chunks_path, embeddings_path)
    if source_part_is_valid(source_name, signature, split_stats):
        manifest = json.loads(source_manifest_path(source_name).read_text(encoding='utf-8'))
        print(
            f'[resume] {source_name}: reuse parts '
            f"train={manifest['selected_chunks']['train']:,}, "
            f"test={manifest['selected_chunks']['test']:,}"
        )
        return manifest

    print(f'\n=== MATERIALIZE FROM {source_name} ===')
    embeddings = np.load(embeddings_path, mmap_mode='r')
    if embeddings.ndim != 2:
        raise ValueError(f'Expected 2D embeddings at {embeddings_path}, got {embeddings.shape}')

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_chunk_paths = {}
    chunk_handles = {}
    embedding_rows = {'train': [], 'test': []}
    selected_pages = {'train': set(), 'test': set()}
    selected_chunks = {'train': 0, 'test': 0}
    source_rows = 0
    started = time.perf_counter()

    try:
        for split in ('train', 'test'):
            final_chunk, _ = part_paths(source_name, split)
            tmp_chunk = final_chunk.with_suffix(final_chunk.suffix + '.tmp')
            if tmp_chunk.exists():
                tmp_chunk.unlink()
            tmp_chunk_paths[split] = tmp_chunk
            chunk_handles[split] = gzip.open(
                tmp_chunk, 'wt', encoding='utf-8', compresslevel=1
            )

        with gzip.open(chunks_path, 'rt', encoding='utf-8') as f:
            for row_index, line in enumerate(f):
                source_rows = row_index + 1
                if row_index >= len(embeddings):
                    raise ValueError(
                        f'{source_name}: chunk rows exceed embedding rows at {row_index}'
                    )

                row = json.loads(line)
                source_file = row['source_file']
                target = assignment.get(source_file)
                if target is not None:
                    split = target['split']
                    if int(row['label']) != int(target['label']):
                        raise ValueError(
                            f'Label mismatch for {source_file}: '
                            f"old={row['label']} new={target['label']}"
                        )
                    # Old baseline chunks predate URL propagation. Inject only the
                    # query-page URL from the shared ablation split.
                    row['url'] = target['url']
                    chunk_handles[split].write(
                        json.dumps(row, ensure_ascii=False) + '\n'
                    )
                    embedding_rows[split].append(
                        np.asarray(embeddings[row_index], dtype=np.float32).copy()
                    )
                    selected_pages[split].add(source_file)
                    selected_chunks[split] += 1

                if source_rows % PROGRESS_EVERY_ROWS == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f'{source_name}: scanned={source_rows:,}/{len(embeddings):,} '
                        f'({100.0 * source_rows / len(embeddings):.1f}%) | '
                        f'selected train={selected_chunks["train"]:,}, '
                        f'test={selected_chunks["test"]:,} | '
                        f'{source_rows / elapsed:,.0f} rows/s'
                    )
    finally:
        for handle in chunk_handles.values():
            handle.close()

    if source_rows != len(embeddings):
        raise ValueError(
            f'{source_name}: chunk row count {source_rows:,} != '
            f'embedding row count {len(embeddings):,}'
        )

    for split in ('train', 'test'):
        final_chunk, final_emb = part_paths(source_name, split)
        tmp_chunk_paths[split].replace(final_chunk)

        if embedding_rows[split]:
            arr = np.stack(embedding_rows[split], axis=0).astype(np.float32, copy=False)
        else:
            arr = np.empty((0, embeddings.shape[1]), dtype=np.float32)
        tmp_emb = final_emb.with_suffix(final_emb.suffix + '.tmp')
        with tmp_emb.open('wb') as f:
            np.save(f, arr)
        tmp_emb.replace(final_emb)

    manifest = {
        'source_name': source_name,
        'source_signature': signature,
        'source_rows': source_rows,
        'embedding_dim': int(embeddings.shape[1]),
        'selected_chunks': {k: int(v) for k, v in selected_chunks.items()},
        'selected_pages': {k: len(v) for k, v in selected_pages.items()},
        'target_split_sha256': {k: v['sha256'] for k, v in split_stats.items()},
    }
    atomic_json(source_manifest_path(source_name), manifest)
    print(
        f'{source_name} done: '
        f"train chunks={selected_chunks['train']:,}, test chunks={selected_chunks['test']:,}"
    )
    return manifest


def merge_parts(split: str, source_manifests: list[dict]) -> dict:
    CHUNKS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_chunks = CHUNKS_OUT_DIR / f'{split}_chunks.jsonl.gz'
    out_embeddings = EMBED_OUT_DIR / f'{split}_embeddings.npy'

    expected_rows = sum(int(m['selected_chunks'][split]) for m in source_manifests)
    embedding_dim = next(int(m['embedding_dim']) for m in source_manifests)

    print(f'\n=== MERGE BASELINE {split.upper()} ({expected_rows:,} chunks) ===')

    tmp_chunks = out_chunks.with_suffix(out_chunks.suffix + '.tmp')
    with gzip.open(tmp_chunks, 'wt', encoding='utf-8', compresslevel=1) as out:
        for source_name, _, _ in SOURCES:
            part_chunk, _ = part_paths(source_name, split)
            with gzip.open(part_chunk, 'rt', encoding='utf-8') as src:
                shutil.copyfileobj(src, out)
    tmp_chunks.replace(out_chunks)

    tmp_embeddings = out_embeddings.with_suffix(out_embeddings.suffix + '.tmp')
    mmap = np.lib.format.open_memmap(
        tmp_embeddings,
        mode='w+',
        dtype=np.float32,
        shape=(expected_rows, embedding_dim),
    )
    offset = 0
    for source_name, _, _ in SOURCES:
        _, part_emb = part_paths(source_name, split)
        arr = np.load(part_emb, mmap_mode='r')
        n = len(arr)
        if n:
            mmap[offset:offset+n] = arr
        offset += n
    mmap.flush()
    del mmap
    tmp_embeddings.replace(out_embeddings)

    final_arr = np.load(out_embeddings, mmap_mode='r')
    if len(final_arr) != expected_rows:
        raise RuntimeError(f'{split}: final embeddings row mismatch')

    # Validate chunk row count + all expected pages + URL presence.
    page_names = set()
    row_count = 0
    with gzip.open(out_chunks, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            row_count += 1
            page_names.add(row['source_file'])
            if 'url' not in row:
                raise RuntimeError(f'{split}: missing url in materialized chunk')
    if row_count != expected_rows:
        raise RuntimeError(f'{split}: chunks={row_count:,} embeddings={expected_rows:,}')

    return {
        'pages_with_chunks': len(page_names),
        'chunks': row_count,
        'embedding_shape': list(final_arr.shape),
        'chunks_file': str(out_chunks),
        'embeddings_file': str(out_embeddings),
    }


def main() -> None:
    assignment, split_stats = load_targets()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    final_signature = {
        'split_sha256': {k: v['sha256'] for k, v in split_stats.items()},
        'sources': {
            name: source_signature(chunks, emb)
            for name, chunks, emb in SOURCES
        },
    }

    if FINAL_MANIFEST.exists():
        try:
            old = json.loads(FINAL_MANIFEST.read_text(encoding='utf-8'))
        except Exception:
            old = None
        if old and old.get('input_signature') == final_signature:
            outputs = old.get('outputs', {})
            if all(
                Path(outputs[s]['chunks_file']).exists()
                and Path(outputs[s]['embeddings_file']).exists()
                for s in ('train', 'test')
            ):
                print('Baseline materialization already complete; reuse existing outputs.')
                print(json.dumps(outputs, indent=2))
                return

    manifests = []
    for source_name, chunks_path, embeddings_path in SOURCES:
        manifests.append(
            process_source(
                source_name,
                chunks_path,
                embeddings_path,
                assignment,
                split_stats,
            )
        )

    # Every one of the 2500 target pages must occur in at least one old baseline
    # split. Check page coverage using the materialized chunk parts.
    found = {'train': set(), 'test': set()}
    for source_name, _, _ in SOURCES:
        for split in ('train', 'test'):
            part_chunk, _ = part_paths(source_name, split)
            with gzip.open(part_chunk, 'rt', encoding='utf-8') as f:
                for line in f:
                    found[split].add(json.loads(line)['source_file'])

    expected_names = {
        split: {name for name, meta in assignment.items() if meta['split'] == split}
        for split in ('train', 'test')
    }
    for split in ('train', 'test'):
        missing = sorted(expected_names[split] - found[split])
        if missing:
            preview = ', '.join(missing[:10])
            raise RuntimeError(
                f'{split}: {len(missing)} target pages missing from original baseline chunks. '
                f'Examples: {preview}'
            )

    outputs = {split: merge_parts(split, manifests) for split in ('train', 'test')}

    final_manifest = {
        'baseline': True,
        'reuse_original_jina_embeddings': True,
        'num_target_pages': len(assignment),
        'split_stats': split_stats,
        'input_signature': final_signature,
        'source_manifests': manifests,
        'outputs': outputs,
    }
    atomic_json(FINAL_MANIFEST, final_manifest)

    print('\nBASELINE MATERIALIZATION COMPLETE')
    print(json.dumps(outputs, indent=2))
    print(f'Manifest: {FINAL_MANIFEST}')


if __name__ == '__main__':
    main()
