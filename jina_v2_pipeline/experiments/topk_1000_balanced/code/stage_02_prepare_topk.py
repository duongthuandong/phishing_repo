import argparse
import csv
import gzip
import hashlib
import heapq
import itertools
import json
import math
import shutil
import time
from pathlib import Path

import chromadb
import numpy as np

ROOT = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline')
SOURCE_DATA = ROOT / 'data'
EXP_ROOT = ROOT / 'experiments' / 'topk_1000_balanced'
EXP_DATA = EXP_ROOT / 'data'

SOURCE_TEST_CHUNKS = SOURCE_DATA / 'chunks' / 'test_chunks.jsonl.gz'
SOURCE_TEST_EMBEDDINGS = SOURCE_DATA / 'embeddings' / 'test_embeddings.npy'
SOURCE_DB = SOURCE_DATA / 'chromadb'

DEFAULT_SAMPLE = EXP_DATA / 'sample' / 'test_1000_balanced_seed42.csv'
RETRIEVAL_DIR = EXP_DATA / 'retrieval'
RETRIEVAL_SHARDS = RETRIEVAL_DIR / 'shards'
RETRIEVAL_CHECKPOINT = RETRIEVAL_DIR / 'retrieval_checkpoint.json'
RETRIEVAL_MANIFEST = RETRIEVAL_DIR / 'retrieval_manifest.json'
INPUTS_DIR = EXP_DATA / 'inputs'

LOCAL_DIR = Path('/content/jina_v2_topk_1000')
LOCAL_TEST_CHUNKS = LOCAL_DIR / 'test_chunks.jsonl.gz'
LOCAL_TEST_EMBEDDINGS = LOCAL_DIR / 'test_embeddings.npy'
LOCAL_DB = LOCAL_DIR / 'chromadb'

BENIGN_COLLECTION = 'html_chunks_benign'
PHISHING_COLLECTION = 'html_chunks_phishing'
DEFAULT_K_VALUES = (1, 2, 4, 8, 16)
DEFAULT_QUERY_BATCH = 512
DEFAULT_SHARD_PAGES = 250
COPY_BUFFER = 16 * 1024 * 1024
CHECKPOINT_VERSION = 1


def parse_args():
    p = argparse.ArgumentParser(description='Retrieve once at max-K and materialize multiple Top-K LLM inputs.')
    p.add_argument('--sample-csv', type=Path, default=DEFAULT_SAMPLE)
    p.add_argument('--k-values', type=int, nargs='+', default=list(DEFAULT_K_VALUES))
    p.add_argument('--query-batch-size', type=int, default=DEFAULT_QUERY_BATCH)
    p.add_argument('--shard-pages', type=int, default=DEFAULT_SHARD_PAGES)
    return p.parse_args()


def fmt_bytes(n):
    x = float(n)
    for u in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if x < 1024 or u == 'TiB':
            return f'{x:,.1f} {u}'
        x /= 1024


def fmt_time(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return '--:--'
    s = int(round(seconds))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def copy_file_progress(src: Path, dst: Path, label: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print(f'Reuse local {label}: {dst}')
        return
    if dst.exists():
        dst.unlink()
    total = src.stat().st_size
    done = 0
    start = time.perf_counter()
    with src.open('rb') as fi, dst.open('wb') as fo:
        while True:
            b = fi.read(COPY_BUFFER)
            if not b:
                break
            fo.write(b)
            done += len(b)
            elapsed = max(time.perf_counter() - start, 1e-9)
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else math.inf
            print(
                f'\r{label}: {fmt_bytes(done)}/{fmt_bytes(total)} ({100*done/total:6.2f}%) | '
                f'{fmt_bytes(int(rate))}/s | ETA {fmt_time(eta)}', end='', flush=True
            )
    shutil.copystat(src, dst)
    print()


def copytree_progress(src_dir: Path, dst_dir: Path, label: str):
    # Local path is experiment-specific. If a valid local Chroma sqlite exists, reuse it
    # within the same Colab runtime instead of copying the full DB again.
    if dst_dir.exists() and (dst_dir / 'chroma.sqlite3').exists():
        print(f'Reuse local {label}: {dst_dir}')
        return
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    files = [p for p in src_dir.rglob('*') if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    done = 0
    start = time.perf_counter()
    for src in files:
        dst = dst_dir / src.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as fi, dst.open('wb') as fo:
            while True:
                b = fi.read(COPY_BUFFER)
                if not b:
                    break
                fo.write(b)
                done += len(b)
                elapsed = max(time.perf_counter() - start, 1e-9)
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else math.inf
                print(
                    f'\r{label}: {fmt_bytes(done)}/{fmt_bytes(total)} ({100*done/total:6.2f}%) | '
                    f'{fmt_bytes(int(rate))}/s | ETA {fmt_time(eta)}', end='', flush=True
                )
        shutil.copystat(src, dst)
    print()


def ensure_local():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    copy_file_progress(SOURCE_TEST_CHUNKS, LOCAL_TEST_CHUNKS, 'Copy test chunks')
    copy_file_progress(SOURCE_TEST_EMBEDDINGS, LOCAL_TEST_EMBEDDINGS, 'Copy test embeddings')
    copytree_progress(SOURCE_DB, LOCAL_DB, 'Copy full-train ChromaDB')


def load_sample(path: Path):
    labels = {}
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row['filename']
            label = int(row['label'])
            if name in labels:
                raise ValueError(f'Duplicate sample page: {name}')
            labels[name] = label
    if not labels:
        raise ValueError('Sample CSV is empty.')
    return labels


def db_signature():
    cp = SOURCE_DB / 'build_checkpoint.json'
    if cp.exists():
        try:
            data = json.loads(cp.read_text(encoding='utf-8'))
            return {
                'total_rows': data.get('total_rows'),
                'next_row_index': data.get('next_row_index'),
                'input_signature': data.get('input_signature'),
                'benign_collection': data.get('benign_collection'),
                'phishing_collection': data.get('phishing_collection'),
            }
        except Exception:
            pass
    sqlite = SOURCE_DB / 'chroma.sqlite3'
    return {'sqlite_size': sqlite.stat().st_size if sqlite.exists() else None}


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_jsonl_gz(path: Path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_jsonl_gz_exclusive(path: Path, rows):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=1) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def cosine_similarity(distance):
    return float(np.clip(1.0 - float(distance), 0.0, 1.0))


def query_top1(collection, q):
    result = collection.query(
        query_embeddings=np.asarray(q, dtype=np.float32).tolist(),
        n_results=1,
        include=['documents', 'distances'],
    )
    out = []
    for docs, distances in zip(result['documents'], result['distances']):
        if not docs or not distances:
            raise RuntimeError('ChromaDB returned empty nearest-neighbor result.')
        out.append({'chunk': docs[0], 'similarity': cosine_similarity(distances[0])})
    return out


def page_id(source):
    return hashlib.sha1(source.encode('utf-8')).hexdigest()[:16]


def push_top(heap, group, max_k):
    key = (float(group['importance']), -int(group['query_chunk_index']), int(group['query_chunk_index']), group)
    if len(heap) < max_k:
        heapq.heappush(heap, key)
    elif key[:3] > heap[0][:3]:
        heapq.heapreplace(heap, key)


def finalize_page(source, true_label, num_chunks, heap):
    groups = [x[3] for x in heap]
    groups.sort(key=lambda x: (-x['importance'], x['query_chunk_index']))
    return {
        'page_id': page_id(source),
        'true_label': int(true_label),
        'num_page_chunks': int(num_chunks),
        'num_selected_chunks': len(groups),
        'selected_chunks': groups,
    }


def retrieval_signature(sample_csv: Path, max_k: int):
    return {
        'sample_csv': str(sample_csv),
        'sample_csv_size': sample_csv.stat().st_size,
        'test_chunks_size': SOURCE_TEST_CHUNKS.stat().st_size,
        'test_embeddings_size': SOURCE_TEST_EMBEDDINGS.stat().st_size,
        'db_signature': db_signature(),
        'max_k': max_k,
        'retrieval_per_label': 1,
        'importance_formula': 'max(s_p,s_b)*abs(s_p-s_b)',
    }


def load_checkpoint(signature):
    if not RETRIEVAL_CHECKPOINT.exists():
        return None
    cp = json.loads(RETRIEVAL_CHECKPOINT.read_text(encoding='utf-8'))
    if cp.get('version') != CHECKPOINT_VERSION or cp.get('signature') != signature:
        raise RuntimeError('Retrieval checkpoint signature mismatch; refusing to mix experiment outputs.')
    return cp


def save_checkpoint(signature, next_row_index, next_shard_index, completed_pages, finalized):
    atomic_json(RETRIEVAL_CHECKPOINT, {
        'version': CHECKPOINT_VERSION,
        'signature': signature,
        'next_row_index': int(next_row_index),
        'next_shard_index': int(next_shard_index),
        'completed_pages': int(completed_pages),
        'finalized': bool(finalized),
    })


def build_retrieval(sample_csv: Path, max_k: int, query_batch_size: int, shard_pages: int):
    signature = retrieval_signature(sample_csv, max_k)
    sample_labels = load_sample(sample_csv)
    selected = set(sample_labels)

    RETRIEVAL_SHARDS.mkdir(parents=True, exist_ok=True)
    cp = load_checkpoint(signature)
    if cp is None:
        existing = sorted(RETRIEVAL_SHARDS.glob('part-*.jsonl.gz'))
        if existing:
            raise RuntimeError('Retrieval shards exist without checkpoint; refusing to overwrite/mix them.')
        start_row = 0
        next_shard = 0
        completed_pages = 0
    else:
        start_row = int(cp['next_row_index'])
        next_shard = int(cp['next_shard_index'])
        completed_pages = int(cp['completed_pages'])
        if cp.get('finalized'):
            print(f'Reuse finalized retrieval: {RETRIEVAL_DIR}')
            return

    ensure_local()
    embeddings = np.load(LOCAL_TEST_EMBEDDINGS, mmap_mode='r')
    total_rows = len(embeddings)
    client = chromadb.PersistentClient(path=str(LOCAL_DB))
    benign = client.get_collection(BENIGN_COLLECTION)
    phishing = client.get_collection(PHISHING_COLLECTION)

    print(f'Full train corpus collections: benign={benign.count():,}, phishing={phishing.count():,}')
    print(f'Sampled test pages: {len(selected):,}; max K={max_k}')
    print(f'Resume row: {start_row:,}/{total_rows:,}; completed pages: {completed_pages:,}')

    pending_pages = []
    current_source = None
    current_label = None
    current_num_chunks = 0
    current_heap = []
    last_global_row = start_row
    started = time.perf_counter()
    selected_rows_processed = 0

    def flush_pending(force=False, checkpoint_row=None):
        nonlocal pending_pages, next_shard, completed_pages
        if not pending_pages:
            return
        if len(pending_pages) < shard_pages and not force:
            return
        path = RETRIEVAL_SHARDS / f'part-{next_shard:05d}.jsonl.gz'
        write_jsonl_gz_exclusive(path, pending_pages)
        completed_pages += len(pending_pages)
        next_shard += 1
        row_for_cp = total_rows if checkpoint_row is None else int(checkpoint_row)
        save_checkpoint(signature, row_for_cp, next_shard, completed_pages, False)
        print(f'\nSaved retrieval {path.name}: {len(pending_pages)} pages | next row {row_for_cp:,}')
        pending_pages = []

    def iter_selected_rows(start):
        with gzip.open(LOCAL_TEST_CHUNKS, 'rt', encoding='utf-8') as f:
            for row_index, line in enumerate(itertools.islice(f, start, None), start=start):
                row = json.loads(line)
                if row['source_file'] not in selected:
                    continue
                yield row_index, {
                    'source_file': row['source_file'],
                    'chunk_index': int(row['chunk_index']),
                    'label': int(row['label']),
                    'document': row['document'],
                }

    row_iter = iter_selected_rows(start_row)
    while True:
        batch = list(itertools.islice(row_iter, query_batch_size))
        if not batch:
            break
        indices = np.asarray([idx for idx, _ in batch], dtype=np.int64)
        if int(indices[-1]) >= total_rows:
            raise RuntimeError('Selected chunk row exceeds embedding count.')
        q = np.asarray(embeddings[indices], dtype=np.float32)
        b_top = query_top1(benign, q)
        p_top = query_top1(phishing, q)

        for (row_index, row), b, p in zip(batch, b_top, p_top):
            last_global_row = row_index + 1
            source = row['source_file']
            label = int(row['label'])
            if sample_labels[source] != label:
                raise RuntimeError(f'Label mismatch for {source}: sample={sample_labels[source]}, chunks={label}')

            if current_source is None:
                current_source = source
                current_label = label
            elif source != current_source:
                pending_pages.append(finalize_page(current_source, current_label, current_num_chunks, current_heap))
                # row_index is the first selected chunk of the next selected page, so it is a safe resume point.
                flush_pending(force=False, checkpoint_row=row_index)
                current_source = source
                current_label = label
                current_num_chunks = 0
                current_heap = []

            s_b = float(b['similarity'])
            s_p = float(p['similarity'])
            group = {
                'query_chunk_index': int(row['chunk_index']),
                'query_chunk': row['document'],
                'benign_example': {'chunk': b['chunk'], 'similarity': s_b},
                'phishing_example': {'chunk': p['chunk'], 'similarity': s_p},
                'importance': max(s_p, s_b) * abs(s_p - s_b),
            }
            push_top(current_heap, group, max_k)
            current_num_chunks += 1
            selected_rows_processed += 1

        elapsed = max(time.perf_counter() - started, 1e-9)
        print(
            f'\rSelected chunks retrieved this run: {selected_rows_processed:,} | '
            f'global row {last_global_row:,}/{total_rows:,} | '
            f'{selected_rows_processed/elapsed:,.1f} selected chunk/s | '
            f'pages saved {completed_pages:,}', end='', flush=True
        )

    if current_source is not None:
        pending_pages.append(finalize_page(current_source, current_label, current_num_chunks, current_heap))
    flush_pending(force=True, checkpoint_row=total_rows)
    print()

    # Count completed pages from shards and verify exact sample coverage.
    pages = []
    for path in sorted(RETRIEVAL_SHARDS.glob('part-*.jsonl.gz')):
        pages.extend(read_jsonl_gz(path))
    if len(pages) != len(selected):
        raise RuntimeError(f'Retrieval page count mismatch: got {len(pages)}, expected {len(selected)}')
    ids = [p['page_id'] for p in pages]
    if len(set(ids)) != len(ids):
        raise RuntimeError('Duplicate page_id in retrieval output.')

    atomic_json(RETRIEVAL_MANIFEST, {
        'signature': signature,
        'finalized': True,
        'num_pages': len(pages),
        'num_shards': len(list(RETRIEVAL_SHARDS.glob('part-*.jsonl.gz'))),
        'full_train_chromadb': str(SOURCE_DB),
        'full_train_collection_counts': {'benign': benign.count(), 'phishing': phishing.count()},
        'source_file_in_llm_payload': False,
    })
    save_checkpoint(signature, total_rows, next_shard, len(pages), True)
    print(f'Retrieval finalized: {RETRIEVAL_DIR}')


def materialize_inputs(k_values, shard_pages: int):
    if not RETRIEVAL_MANIFEST.exists():
        raise FileNotFoundError('Missing retrieval manifest.')
    rmeta = json.loads(RETRIEVAL_MANIFEST.read_text(encoding='utf-8'))
    if not rmeta.get('finalized'):
        raise RuntimeError('Retrieval is not finalized.')
    retrieval_shards = sorted(RETRIEVAL_SHARDS.glob('part-*.jsonl.gz'))
    pages = []
    for p in retrieval_shards:
        pages.extend(read_jsonl_gz(p))

    for k in k_values:
        k_dir = INPUTS_DIR / f'k_{k:02d}'
        shard_dir = k_dir / 'shards'
        manifest_path = k_dir / 'input_manifest.json'
        signature = {
            'retrieval_signature': rmeta.get('signature'),
            'top_k': int(k),
            'num_pages': len(pages),
            'shard_pages': shard_pages,
            'prompt_scores_included': True,
            'source_file_in_prompt': False,
            'page_url_in_prompt': False,
        }
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding='utf-8'))
            if old.get('signature') == signature and old.get('finalized'):
                print(f'Reuse input K={k}: {k_dir}')
                continue
            raise RuntimeError(f'Existing K={k} input has different/incomplete signature.')
        if shard_dir.exists() and any(shard_dir.iterdir()):
            raise RuntimeError(f'Existing shards without valid manifest: {shard_dir}')
        shard_dir.mkdir(parents=True, exist_ok=True)

        batch = []
        shard_idx = 0
        for page in pages:
            chosen = page['selected_chunks'][:k]
            if not chosen:
                raise RuntimeError(f"Page {page['page_id']} has no evidence groups")
            batch.append({
                'page_id': page['page_id'],
                'true_label': int(page['true_label']),
                'num_page_chunks': int(page['num_page_chunks']),
                'num_selected_chunks': len(chosen),
                'selected_chunks': chosen,
            })
            if len(batch) >= shard_pages:
                write_jsonl_gz_exclusive(shard_dir / f'part-{shard_idx:05d}.jsonl.gz', batch)
                batch = []
                shard_idx += 1
        if batch:
            write_jsonl_gz_exclusive(shard_dir / f'part-{shard_idx:05d}.jsonl.gz', batch)
            shard_idx += 1

        atomic_json(manifest_path, {
            'signature': signature,
            'finalized': True,
            'top_k': k,
            'num_pages': len(pages),
            'num_shards': shard_idx,
            'payload': 'query chunk + nearest BENIGN chunk + nearest PHISHING chunk + similarities',
        })
        print(f'Saved input K={k}: {k_dir} ({shard_idx} shards)')


def main():
    args = parse_args()
    k_values = sorted(set(args.k_values))
    if not k_values or min(k_values) <= 0:
        raise ValueError('All K values must be > 0.')
    if args.query_batch_size <= 0 or args.shard_pages <= 0:
        raise ValueError('Batch/shard sizes must be > 0.')
    if not args.sample_csv.exists():
        raise FileNotFoundError(f'Missing sample CSV: {args.sample_csv}. Run stage_01_sample_test.py first.')

    build_retrieval(args.sample_csv, max(k_values), args.query_batch_size, args.shard_pages)
    materialize_inputs(k_values, args.shard_pages)
    print('Done. K values:', k_values)


if __name__ == '__main__':
    main()
