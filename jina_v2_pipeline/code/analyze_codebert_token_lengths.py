import gzip
import heapq
import json
import shutil
import time
from array import array
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DRIVE_DATA_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/data')
DRIVE_CHUNKS_DIR = DRIVE_DATA_DIR / 'chunks'

INPUT_FILES = {
    'train': DRIVE_CHUNKS_DIR / 'train_chunks.jsonl.gz',
    'test': DRIVE_CHUNKS_DIR / 'test_chunks.jsonl.gz',
}

LOCAL_DIR = Path('/content/codebert_token_analysis')
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'microsoft/codebert-base'
MODEL_LIMIT = 512
BATCH_SIZE = 512
TOP_LONGEST = 10
PROGRESS_EVERY = 10_000


def copy_if_needed(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f'Không tìm thấy input: {src}')

    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        print(f'Copy local: {src} -> {dst}')
        shutil.copy2(src, dst)
    else:
        print(f'Dùng bản local có sẵn: {dst}')


def percentile_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}

    percentiles = np.percentile(values, [50, 75, 90, 95, 99, 99.5, 99.9])
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'min': int(values.min()),
        'p50': float(percentiles[0]),
        'p75': float(percentiles[1]),
        'p90': float(percentiles[2]),
        'p95': float(percentiles[3]),
        'p99': float(percentiles[4]),
        'p99.5': float(percentiles[5]),
        'p99.9': float(percentiles[6]),
        'max': int(values.max()),
    }


def tokenize_batch(tokenizer, documents):
    encoded = tokenizer(
        documents,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_length=True,
    )
    return encoded['length']


def analyze_split(split_name: str, path: Path, tokenizer) -> dict:
    print('\n' + '=' * 80)
    print(f'ANALYZE: {split_name}')
    print(f'Input: {path}')
    print('=' * 80)

    codebert_lengths = array('I')
    jina_lengths = array('I')
    over_limit = 0
    top_longest = []

    batch_docs = []
    batch_meta = []

    def process_batch():
        nonlocal over_limit
        if not batch_docs:
            return

        lengths = tokenize_batch(tokenizer, batch_docs)

        for length, meta in zip(lengths, batch_meta):
            length = int(length)
            codebert_lengths.append(length)

            jina_count = int(meta['jina_token_count'])
            jina_lengths.append(jina_count)

            if length > MODEL_LIMIT:
                over_limit += 1

            item = (
                length,
                meta['id'],
                meta['source_file'],
                int(meta['chunk_index']),
                jina_count,
            )

            if len(top_longest) < TOP_LONGEST:
                heapq.heappush(top_longest, item)
            elif length > top_longest[0][0]:
                heapq.heapreplace(top_longest, item)

        batch_docs.clear()
        batch_meta.clear()

    print(f'[{split_name}] Đếm tổng số chunk...')
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        total_chunks = sum(1 for _ in f)

    print(f'[{split_name}] Tổng số chunk cần xử lý: {total_chunks:,}')

    started_at = time.perf_counter()
    processed = 0

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            batch_docs.append(row['document'])
            batch_meta.append({
                'id': row['id'],
                'source_file': row['source_file'],
                'chunk_index': row['chunk_index'],
                'jina_token_count': row['token_count'],
            })

            if len(batch_docs) >= BATCH_SIZE:
                process_batch()

            processed += 1

            if processed % PROGRESS_EVERY == 0 or processed == total_chunks:
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                rate = processed / elapsed
                remaining = total_chunks - processed
                eta_seconds = remaining / rate if rate > 0 else 0.0
                percent = 100.0 * processed / total_chunks if total_chunks else 100.0

                print(
                    f'[{split_name}] Progress: {processed:,}/{total_chunks:,} '
                    f'({percent:6.2f}%) | '
                    f'{rate:,.1f} chunk/s | '
                    f'elapsed={elapsed / 60:.1f} min | '
                    f'ETA={eta_seconds / 60:.1f} min'
                )

        process_batch()

    codebert_np = np.frombuffer(codebert_lengths, dtype=np.uint32)
    jina_np = np.frombuffer(jina_lengths, dtype=np.uint32)

    total = int(codebert_np.size)
    if total == 0:
        raise ValueError(f'{split_name}: không có chunk')

    within_limit = total - over_limit

    # Chỉ để mô tả mức khác biệt tokenizer; không dùng cho train.
    ratio = codebert_np.astype(np.float32) / np.maximum(jina_np.astype(np.float32), 1.0)

    stats = {
        'split': split_name,
        'model': MODEL_NAME,
        'model_limit': MODEL_LIMIT,
        'codebert_tokens': percentile_summary(codebert_np),
        'jina_tokens_from_chunk_file': percentile_summary(jina_np),
        'within_limit': within_limit,
        'within_limit_percent': 100.0 * within_limit / total,
        'over_limit': over_limit,
        'over_limit_percent': 100.0 * over_limit / total,
        'over_600': int(np.count_nonzero(codebert_np > 600)),
        'over_768': int(np.count_nonzero(codebert_np > 768)),
        'over_1024': int(np.count_nonzero(codebert_np > 1024)),
        'codebert_to_jina_token_ratio': {
            'mean': float(ratio.mean()),
            'p50': float(np.percentile(ratio, 50)),
            'p90': float(np.percentile(ratio, 90)),
            'p95': float(np.percentile(ratio, 95)),
            'p99': float(np.percentile(ratio, 99)),
            'max': float(ratio.max()),
        },
        'top_longest': [
            {
                'codebert_tokens': int(length),
                'jina_tokens': int(jina_count),
                'id': chunk_id,
                'source_file': source_file,
                'chunk_index': int(chunk_index),
            }
            for length, chunk_id, source_file, chunk_index, jina_count
            in sorted(top_longest, reverse=True)
        ],
    }

    print(f'\n[{split_name}] Tổng chunk: {total:,}')
    print(
        f'[{split_name}] <= {MODEL_LIMIT}: '
        f"{within_limit:,} ({stats['within_limit_percent']:.4f}%)"
    )
    print(
        f'[{split_name}] >  {MODEL_LIMIT}: '
        f"{over_limit:,} ({stats['over_limit_percent']:.4f}%)"
    )

    s = stats['codebert_tokens']
    print(
        f'[{split_name}] CodeBERT token length: '
        f"min={s['min']}, mean={s['mean']:.2f}, "
        f"P50={s['p50']:.0f}, P90={s['p90']:.0f}, "
        f"P95={s['p95']:.0f}, P99={s['p99']:.0f}, "
        f"P99.5={s['p99.5']:.0f}, P99.9={s['p99.9']:.0f}, "
        f"max={s['max']}"
    )

    print(
        f"[{split_name}] >600: {stats['over_600']:,} | "
        f">768: {stats['over_768']:,} | "
        f">1024: {stats['over_1024']:,}"
    )

    r = stats['codebert_to_jina_token_ratio']
    print(
        f'[{split_name}] Tỷ lệ token CodeBERT/Jina: '
        f"mean={r['mean']:.3f}, P50={r['p50']:.3f}, "
        f"P95={r['p95']:.3f}, P99={r['p99']:.3f}, max={r['max']:.3f}"
    )

    print(f'\n[{split_name}] {TOP_LONGEST} chunk dài nhất theo CodeBERT:')
    for i, item in enumerate(stats['top_longest'], 1):
        print(
            f"  {i:2d}. CodeBERT={item['codebert_tokens']:5d} | "
            f"Jina={item['jina_tokens']:4d} | "
            f"{item['source_file']} | chunk={item['chunk_index']} | "
            f"id={item['id']}"
        )

    return stats


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Giữ giới hạn thực tế riêng ở MODEL_LIMIT=512.
    # Tăng model_max_length tạm thời chỉ để đếm đầy đủ chuỗi >512,
    # không truncate và không phát cảnh báo sequence vượt giới hạn model.
    tokenizer.model_max_length = 1_000_000_000

    all_stats = {}

    for split_name, drive_path in INPUT_FILES.items():
        local_path = LOCAL_DIR / drive_path.name
        copy_if_needed(drive_path, local_path)
        all_stats[split_name] = analyze_split(split_name, local_path, tokenizer)

    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)

    for split_name in ('train', 'test'):
        s = all_stats[split_name]
        print(
            f'{split_name:5s}: '
            f"total={s['codebert_tokens']['count']:,} | "
            f">512={s['over_limit']:,} "
            f"({s['over_limit_percent']:.4f}%) | "
            f"P95={s['codebert_tokens']['p95']:.0f} | "
            f"P99={s['codebert_tokens']['p99']:.0f} | "
            f"max={s['codebert_tokens']['max']}"
        )


if __name__ == '__main__':
    main()
