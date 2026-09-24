import argparse
import os
import gzip
import hashlib
import heapq
import itertools
import json
import shutil
import time
from pathlib import Path

import chromadb
import numpy as np


ABLATION = os.environ.get("ABLATION", "").strip().lower()
ABLATIONS = ("form", "input", "a", "iframe", "meta")
if ABLATION not in ABLATIONS:
    raise ValueError(f"ABLATION phải thuộc {ABLATIONS}, nhận được: {ABLATION!r}")

DRIVE_DATA_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_ablation_pipeline/data/ablations') / ABLATION
DRIVE_TEST_CHUNKS = DRIVE_DATA_DIR / 'chunks' / 'test_chunks.jsonl.gz'
DRIVE_TEST_EMBEDDINGS = DRIVE_DATA_DIR / 'embeddings' / 'test_embeddings.npy'
DRIVE_DB_DIR = DRIVE_DATA_DIR / 'chromadb'

# Reuse the same local cache as evaluate_rag_voting.py so a rerun does not copy
# the ~34 GiB ChromaDB again when the files are already present in the runtime.
LOCAL_SHARED_DIR = Path(f'/content/jina_rag_eval_ablation_{ABLATION}')
LOCAL_TEST_CHUNKS = LOCAL_SHARED_DIR / 'test_chunks.jsonl.gz'
LOCAL_TEST_EMBEDDINGS = LOCAL_SHARED_DIR / 'test_embeddings.npy'
LOCAL_DB_DIR = LOCAL_SHARED_DIR / 'chromadb'

BENIGN_COLLECTION = 'html_chunks_benign'
PHISHING_COLLECTION = 'html_chunks_phishing'
COPY_BUFFER_SIZE = 16 * 1024 * 1024
CHECKPOINT_VERSION = 1

DEFAULT_TOP_N = 8
DEFAULT_QUERY_BATCH_SIZE = 512
DEFAULT_SHARD_PAGES = 250
DEFAULT_RUN_NAME = 'qwen25_coder_7b_top8'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Tạo input page-level cho LLM ablation: với mỗi test chunk lấy nearest '
            'benign + nearest phishing, tính importance, rồi giữ top-N chunk/page.'
        )
    )
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_N)
    parser.add_argument('--query-batch-size', type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument('--shard-pages', type=int, default=DEFAULT_SHARD_PAGES)
    parser.add_argument('--run-name', type=str, default=DEFAULT_RUN_NAME)
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024.0 or unit == 'TiB':
            return f'{value:,.1f} {unit}'
        value /= 1024.0


def copy_file_with_progress(src: Path, dst: Path, label: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = src.stat().st_size
    copied = 0
    started_at = time.perf_counter()
    last_len = 0

    def show(final: bool = False):
        nonlocal last_len
        elapsed = max(time.perf_counter() - started_at, 0.0)
        rate = copied / elapsed if elapsed > 0 else 0.0
        remaining = max(total - copied, 0)
        eta = format_duration(remaining / rate) if rate > 0 else 'calculating...'
        pct = 100.0 * copied / total if total else 100.0
        line = (
            f'{label}: {format_bytes(copied)}/{format_bytes(total)} '
            f'({pct:6.2f}%) | {format_bytes(int(rate))}/s | ETA {eta}'
        )
        print('\r' + line.ljust(last_len), end='\n' if final else '', flush=True)
        last_len = len(line)

    show()
    with src.open('rb') as fsrc, dst.open('wb') as fdst:
        while True:
            block = fsrc.read(COPY_BUFFER_SIZE)
            if not block:
                break
            fdst.write(block)
            copied += len(block)
            show()
    shutil.copystat(src, dst)
    show(final=True)


def copytree_with_progress(src_dir: Path, dst_dir: Path, label: str) -> None:
    files = [p for p in src_dir.rglob('*') if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    copied = 0
    started_at = time.perf_counter()
    last_len = 0
    dst_dir.mkdir(parents=True, exist_ok=True)

    def show(final: bool = False):
        nonlocal last_len
        elapsed = max(time.perf_counter() - started_at, 0.0)
        rate = copied / elapsed if elapsed > 0 else 0.0
        remaining = max(total - copied, 0)
        eta = format_duration(remaining / rate) if rate > 0 else 'calculating...'
        pct = 100.0 * copied / total if total else 100.0
        line = (
            f'{label}: {format_bytes(copied)}/{format_bytes(total)} '
            f'({pct:6.2f}%) | {format_bytes(int(rate))}/s | ETA {eta}'
        )
        print('\r' + line.ljust(last_len), end='\n' if final else '', flush=True)
        last_len = len(line)

    show()
    for src in files:
        dst = dst_dir / src.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as fsrc, dst.open('wb') as fdst:
            while True:
                block = fsrc.read(COPY_BUFFER_SIZE)
                if not block:
                    break
                fdst.write(block)
                copied += len(block)
                show()
        shutil.copystat(src, dst)
    show(final=True)


def ensure_local_inputs() -> None:
    LOCAL_SHARED_DIR.mkdir(parents=True, exist_ok=True)

    if (
        not LOCAL_TEST_CHUNKS.exists()
        or LOCAL_TEST_CHUNKS.stat().st_size != DRIVE_TEST_CHUNKS.stat().st_size
    ):
        copy_file_with_progress(DRIVE_TEST_CHUNKS, LOCAL_TEST_CHUNKS, 'Copy test chunks')

    if (
        not LOCAL_TEST_EMBEDDINGS.exists()
        or LOCAL_TEST_EMBEDDINGS.stat().st_size != DRIVE_TEST_EMBEDDINGS.stat().st_size
    ):
        copy_file_with_progress(
            DRIVE_TEST_EMBEDDINGS,
            LOCAL_TEST_EMBEDDINGS,
            'Copy test embeddings',
        )

    if LOCAL_DB_DIR.exists():
        print(f'Dùng ChromaDB local có sẵn: {LOCAL_DB_DIR}')
    else:
        copytree_with_progress(DRIVE_DB_DIR, LOCAL_DB_DIR, 'Copy ChromaDB')


def open_split_collections():
    def open_local():
        client = chromadb.PersistentClient(path=str(LOCAL_DB_DIR))
        benign = client.get_collection(BENIGN_COLLECTION)
        phishing = client.get_collection(PHISHING_COLLECTION)
        return client, benign, phishing

    try:
        return open_local()
    except Exception as exc:
        print('ChromaDB local không mở được hai collection; copy lại từ Drive...')
        if LOCAL_DB_DIR.exists():
            shutil.rmtree(LOCAL_DB_DIR)
        copytree_with_progress(DRIVE_DB_DIR, LOCAL_DB_DIR, 'Copy ChromaDB')
        try:
            return open_local()
        except Exception:
            raise RuntimeError(
                f'Không mở được {BENIGN_COLLECTION} và {PHISHING_COLLECTION}.'
            ) from exc


def stable_db_signature() -> dict:
    """Dùng build checkpoint ổn định thay vì Collection.count()/file size mutable."""
    build_checkpoint = DRIVE_DB_DIR / 'build_checkpoint.json'
    if build_checkpoint.exists():
        with build_checkpoint.open('r', encoding='utf-8') as f:
            cp = json.load(f)
        return {
            'next_row_index': cp.get('next_row_index'),
            'total_rows': cp.get('total_rows'),
            'completed_percent': cp.get('completed_percent'),
            'input_signature': cp.get('input_signature'),
            'benign_collection': cp.get('benign_collection'),
            'phishing_collection': cp.get('phishing_collection'),
        }

    sqlite_path = DRIVE_DB_DIR / 'chroma.sqlite3'
    return {
        'fallback_sqlite_size': sqlite_path.stat().st_size if sqlite_path.exists() else None
    }


def run_signature(total_rows: int, top_n: int) -> dict:
    return {
        'test_chunks_size': DRIVE_TEST_CHUNKS.stat().st_size,
        'test_embeddings_size': DRIVE_TEST_EMBEDDINGS.stat().st_size,
        'total_rows': int(total_rows),
        'top_n': int(top_n),
        'retrieval_per_label': 1,
        'importance_formula': 'max(s_p,s_b)*abs(s_p-s_b)',
        'benign_collection': BENIGN_COLLECTION,
        'phishing_collection': PHISHING_COLLECTION,
        'db_signature': stable_db_signature(),
    }


def cosine_similarity(distance: float) -> float:
    return float(np.clip(1.0 - float(distance), 0.0, 1.0))


def query_top1(collection, query_embeddings):
    result = collection.query(
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32).tolist(),
        n_results=1,
        include=['documents', 'distances'],
    )
    out = []
    for docs, distances in zip(result['documents'], result['distances']):
        if not docs or not distances or docs[0] is None:
            raise RuntimeError('ChromaDB trả về nearest-neighbor rỗng.')
        out.append({
            'chunk': docs[0],
            'similarity': cosine_similarity(distances[0]),
        })
    return out


def page_id_from_source(source_file: str) -> str:
    return hashlib.sha1(source_file.encode('utf-8')).hexdigest()[:16]


def push_top_n(heap, group: dict, top_n: int) -> None:
    # Tie-break ưu tiên chunk_index nhỏ hơn để kết quả deterministic.
    key = (
        float(group['importance']),
        -int(group['query_chunk_index']),
        int(group['query_chunk_index']),
        group,
    )
    if len(heap) < top_n:
        heapq.heappush(heap, key)
    elif key[:3] > heap[0][:3]:
        heapq.heapreplace(heap, key)


def finalize_page(
    source_file: str,
    page_url: str,
    true_label: int,
    num_page_chunks: int,
    heap,
    next_row_index: int,
) -> dict:
    selected = [item[3] for item in heap]
    selected.sort(key=lambda x: (-x['importance'], x['query_chunk_index']))
    return {
        'page_id': page_id_from_source(source_file),
        'page_url': page_url,
        # true_label is evaluation metadata only. infer_qwen25_coder_rag.py never puts it in the prompt.
        'true_label': int(true_label),
        'num_page_chunks': int(num_page_chunks),
        'num_selected_chunks': len(selected),
        'selected_chunks': selected,
        # Internal resume metadata; no filename/source path is persisted.
        '_next_row_index': int(next_row_index),
    }


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
    tmp.replace(path)


def write_shard_exclusive(shard_path: Path, records) -> None:
    if shard_path.exists():
        raise FileExistsError(
            f'Không ghi đè shard đã tồn tại: {shard_path}. '
            'Dùng checkpoint hiện có hoặc đổi --run-name.'
        )
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard_path.with_suffix(shard_path.suffix + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=1) as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    tmp.replace(shard_path)


def load_checkpoint(path: Path, signature: dict):
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            cp = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Checkpoint không đọc được: {path}: {exc}') from exc

    if cp.get('version') != CHECKPOINT_VERSION or cp.get('signature') != signature:
        raise RuntimeError(
            'Run directory đã có checkpoint nhưng cấu hình/input không khớp. '
            'Để tránh ghi đè/mix kết quả, hãy dùng --run-name khác.'
        )
    return cp


def save_checkpoint(
    path: Path,
    signature: dict,
    next_row_index: int,
    next_shard_index: int,
    completed_pages: int,
    finalized: bool,
) -> None:
    atomic_write_json(path, {
        'version': CHECKPOINT_VERSION,
        'signature': signature,
        'next_row_index': int(next_row_index),
        'next_shard_index': int(next_shard_index),
        'completed_pages': int(completed_pages),
        'finalized': bool(finalized),
    })


def iter_test_rows(start_row: int):
    with gzip.open(LOCAL_TEST_CHUNKS, 'rt', encoding='utf-8') as f:
        for row_index, line in enumerate(
            itertools.islice(f, start_row, None), start=start_row
        ):
            row = json.loads(line)
            yield row_index, {
                'source_file': row['source_file'],
                'chunk_index': int(row['chunk_index']),
                'label': int(row['label']),
                'url': row['url'],
                'document': row['document'],
            }


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError('--top-n phải > 0')
    if args.query_batch_size <= 0:
        raise ValueError('--query-batch-size phải > 0')
    if args.shard_pages <= 0:
        raise ValueError('--shard-pages phải > 0')

    ensure_local_inputs()
    embeddings = np.load(LOCAL_TEST_EMBEDDINGS, mmap_mode='r')
    total_rows = len(embeddings)

    run_dir = DRIVE_DATA_DIR / 'llm_rag' / args.run_name
    input_dir = run_dir / 'input'
    shard_dir = input_dir / 'shards'
    checkpoint_path = input_dir / 'input_checkpoint.json'
    manifest_path = input_dir / 'input_manifest.json'
    shard_dir.mkdir(parents=True, exist_ok=True)

    signature = run_signature(total_rows, args.top_n)
    checkpoint = load_checkpoint(checkpoint_path, signature)

    if checkpoint is None:
        existing_shards = sorted(shard_dir.glob('part-*.jsonl.gz'))
        if existing_shards:
            raise RuntimeError(
                f'Có {len(existing_shards)} shard nhưng không có checkpoint hợp lệ. '
                'Không tự ghi đè. Hãy đổi --run-name hoặc kiểm tra output cũ.'
            )
        start_row = 0
        next_shard_index = 0
        completed_pages = 0
    else:
        start_row = int(checkpoint['next_row_index'])
        next_shard_index = int(checkpoint['next_shard_index'])
        completed_pages = int(checkpoint['completed_pages'])
        if checkpoint.get('finalized'):
            print(f'Input đã hoàn tất: {input_dir}')
            return

    client, benign_collection, phishing_collection = open_split_collections()
    print(f'Benign collection:   {BENIGN_COLLECTION}')
    print(f'Phishing collection: {PHISHING_COLLECTION}')
    print(f'Top-N/page:           {args.top_n}')
    print(f'Resume row:           {start_row:,}/{total_rows:,}')
    print(f'Output:               {input_dir}')

    manifest = {
        'version': 1,
        'run_name': args.run_name,
        'top_n': args.top_n,
        'query_batch_size': args.query_batch_size,
        'shard_pages': args.shard_pages,
        'importance_formula': 'max(s_p,s_b)*abs(s_p-s_b)',
        'retrieval': {
            'benign_collection': BENIGN_COLLECTION,
            'phishing_collection': PHISHING_COLLECTION,
            'n_results_per_label': 1,
        },
        'llm_payload': (
            'query page URL + query chunk + nearest benign chunk labeled BENIGN + nearest phishing '
            'chunk labeled PHISHING; source filename is never persisted/passed to LLM'
        ),
        'signature': signature,
        'finalized': False,
    }
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)
    else:
        with manifest_path.open('r', encoding='utf-8') as f:
            old_manifest = json.load(f)
        if old_manifest.get('signature') != signature:
            raise RuntimeError(
                'input_manifest.json hiện có không khớp run hiện tại. '
                'Dùng --run-name khác để tránh ghi đè.'
            )

    pending_pages = []
    current_source = None
    current_true_label = None
    current_page_url = None
    current_num_chunks = 0
    current_heap = []

    processed_rows = start_row
    run_started = time.perf_counter()
    last_progress_len = 0

    def show_progress(final: bool = False):
        nonlocal last_progress_len
        elapsed = max(time.perf_counter() - run_started, 0.0)
        done_this_run = max(processed_rows - start_row, 0)
        rate = done_this_run / elapsed if elapsed > 0 else 0.0
        remaining = max(total_rows - processed_rows, 0)
        eta = format_duration(remaining / rate) if rate > 0 else 'calculating...'
        pct = 100.0 * processed_rows / total_rows if total_rows else 100.0
        line = (
            f'LLM-input retrieval: {processed_rows:,}/{total_rows:,} ({pct:6.2f}%) | '
            f'{rate:,.1f} chunk/s | pages {completed_pages + len(pending_pages):,} | '
            f'ETA {eta}'
        )
        print('\r' + line.ljust(last_progress_len), end='\n' if final else '', flush=True)
        last_progress_len = len(line)

    def flush_page_shard_if_needed(force: bool = False):
        nonlocal pending_pages, next_shard_index, completed_pages
        if not pending_pages:
            return
        if len(pending_pages) < args.shard_pages and not force:
            return

        shard_path = shard_dir / f'part-{next_shard_index:05d}.jsonl.gz'
        write_shard_exclusive(shard_path, pending_pages)
        completed_pages += len(pending_pages)
        next_row = int(pending_pages[-1]['_next_row_index'])
        next_shard_index += 1
        save_checkpoint(
            checkpoint_path,
            signature,
            next_row_index=next_row,
            next_shard_index=next_shard_index,
            completed_pages=completed_pages,
            finalized=False,
        )
        print(f'\nSaved {shard_path.name}: {len(pending_pages)} pages; resume row={next_row:,}')
        pending_pages = []

    row_iter = iter_test_rows(start_row)
    while True:
        batch = list(itertools.islice(row_iter, args.query_batch_size))
        if not batch:
            break

        batch_indices = [idx for idx, _ in batch]
        batch_rows = [row for _, row in batch]
        start = batch_indices[0]
        end = batch_indices[-1] + 1
        if end > total_rows:
            raise ValueError(
                f'Số test chunk lớn hơn số embedding: cần row {end:,}, chỉ có {total_rows:,} embedding.'
            )
        query_embeddings = np.asarray(embeddings[start:end], dtype=np.float32)

        benign_top1 = query_top1(benign_collection, query_embeddings)
        phishing_top1 = query_top1(phishing_collection, query_embeddings)

        for (row_index, row), b, p in zip(batch, benign_top1, phishing_top1):
            source = row['source_file']
            true_label = row['label']

            if current_source is None:
                current_source = source
                current_true_label = true_label
                current_page_url = row['url']

            if source != current_source:
                pending_pages.append(finalize_page(
                    current_source,
                    current_page_url,
                    current_true_label,
                    current_num_chunks,
                    current_heap,
                    next_row_index=row_index,
                ))
                flush_page_shard_if_needed()
                current_source = source
                current_true_label = true_label
                current_page_url = row['url']
                current_num_chunks = 0
                current_heap = []

            if true_label != current_true_label:
                raise ValueError(f'Page hash={page_id_from_source(source)} có nhiều label.')

            s_b = float(b['similarity'])
            s_p = float(p['similarity'])
            importance = max(s_p, s_b) * abs(s_p - s_b)
            group = {
                'query_chunk_index': int(row['chunk_index']),
                'importance': float(importance),
                'query_chunk': row['document'],
                'benign_example': {
                    'label': 'benign',
                    'similarity': s_b,
                    'chunk': b['chunk'],
                },
                'phishing_example': {
                    'label': 'phishing',
                    'similarity': s_p,
                    'chunk': p['chunk'],
                },
            }
            push_top_n(current_heap, group, args.top_n)
            current_num_chunks += 1
            processed_rows = row_index + 1

        show_progress()

    if processed_rows != total_rows:
        raise ValueError(
            f'Số test chunk đọc được không khớp embedding: {processed_rows:,} != {total_rows:,}'
        )

    if current_source is not None:
        pending_pages.append(finalize_page(
            current_source,
            current_page_url,
            current_true_label,
            current_num_chunks,
            current_heap,
            next_row_index=total_rows,
        ))

    flush_page_shard_if_needed(force=True)
    processed_rows = total_rows
    save_checkpoint(
        checkpoint_path,
        signature,
        next_row_index=total_rows,
        next_shard_index=next_shard_index,
        completed_pages=completed_pages,
        finalized=True,
    )

    manifest['finalized'] = True
    manifest['num_pages'] = completed_pages
    manifest['num_test_chunks'] = total_rows
    # Manifest is only finalized once. If the file already exists, replace atomically
    # because this is metadata for the same run, not an experimental output result.
    atomic_write_json(manifest_path, manifest)
    show_progress(final=True)
    print(f'Done. Pages: {completed_pages:,}')
    print(f'Input shards: {shard_dir}')

    # Keep client referenced until all queries are done; explicit del helps Colab release RAM.
    del benign_collection, phishing_collection, client


if __name__ == '__main__':
    main()
