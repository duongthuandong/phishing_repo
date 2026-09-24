import csv
import gzip
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DRIVE_DATA_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/data')
DRIVE_TEST_CHUNKS = DRIVE_DATA_DIR / 'chunks' / 'test_chunks.jsonl.gz'
DRIVE_TEST_EMBEDDINGS = DRIVE_DATA_DIR / 'embeddings' / 'test_embeddings.npy'
DRIVE_DB_DIR = DRIVE_DATA_DIR / 'chromadb'
DRIVE_OUT_DIR = DRIVE_DATA_DIR / 'rag_eval'
DRIVE_PREDICTIONS = DRIVE_OUT_DIR / 'test_rag_predictions.csv'
DRIVE_METRICS = DRIVE_OUT_DIR / 'test_rag_metrics.json'
DRIVE_CHECKPOINT = DRIVE_OUT_DIR / 'test_rag_checkpoint.json'

LOCAL_DIR = Path('/content/jina_rag_eval')
LOCAL_TEST_CHUNKS = LOCAL_DIR / 'test_chunks.jsonl.gz'
LOCAL_TEST_EMBEDDINGS = LOCAL_DIR / 'test_embeddings.npy'
LOCAL_DB_DIR = LOCAL_DIR / 'chromadb'

BENIGN_COLLECTION = 'html_chunks_benign'
PHISHING_COLLECTION = 'html_chunks_phishing'
TOP_K_PER_LABEL = 50
TOP_UNIQUE_SOURCE_PER_CHUNK = 10
QUERY_BATCH_SIZE = 512
MAX_SQL_RESULTS_PER_QUERY = 16000
CHECKPOINT_PERCENTS = tuple(range(10, 101, 10))
CHECKPOINT_VERSION = 2
COPY_BUFFER_SIZE = 16 * 1024 * 1024


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
    """Copy một file từ Drive xuống local và hiển thị tiến độ một dòng."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = src.stat().st_size
    copied = 0
    started_at = time.perf_counter()
    last_progress_len = 0

    def print_copy_progress(final: bool = False) -> None:
        nonlocal last_progress_len
        elapsed = max(time.perf_counter() - started_at, 0.0)
        rate = copied / elapsed if elapsed > 0 else 0.0
        remaining = max(total - copied, 0)
        percent = 100.0 * copied / total if total else 100.0
        eta_text = format_duration(remaining / rate) if rate > 0 else 'calculating...'
        line = (
            f'{label}: {format_bytes(copied)}/{format_bytes(total)} '
            f'({percent:6.2f}%) | {format_bytes(int(rate))}/s | ETA {eta_text}'
        )
        padded_line = line.ljust(last_progress_len)
        print('\r' + padded_line, end='\n' if final else '', flush=True)
        last_progress_len = len(line)

    print_copy_progress()
    with src.open('rb') as fsrc, dst.open('wb') as fdst:
        while True:
            chunk = fsrc.read(COPY_BUFFER_SIZE)
            if not chunk:
                break
            fdst.write(chunk)
            copied += len(chunk)
            print_copy_progress()

    shutil.copystat(src, dst)
    print_copy_progress(final=True)


def copytree_with_progress(src_dir: Path, dst_dir: Path, label: str) -> None:
    """Copy toàn bộ thư mục và hiển thị tiến độ theo tổng số byte."""
    files = [path for path in src_dir.rglob('*') if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    copied = 0
    started_at = time.perf_counter()
    last_progress_len = 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    def print_copy_progress(final: bool = False) -> None:
        nonlocal last_progress_len
        elapsed = max(time.perf_counter() - started_at, 0.0)
        rate = copied / elapsed if elapsed > 0 else 0.0
        remaining = max(total - copied, 0)
        percent = 100.0 * copied / total if total else 100.0
        eta_text = format_duration(remaining / rate) if rate > 0 else 'calculating...'
        line = (
            f'{label}: {format_bytes(copied)}/{format_bytes(total)} '
            f'({percent:6.2f}%) | {format_bytes(int(rate))}/s | ETA {eta_text}'
        )
        padded_line = line.ljust(last_progress_len)
        print('\r' + padded_line, end='\n' if final else '', flush=True)
        last_progress_len = len(line)

    print_copy_progress()
    for src in files:
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as fsrc, dst.open('wb') as fdst:
            while True:
                chunk = fsrc.read(COPY_BUFFER_SIZE)
                if not chunk:
                    break
                fdst.write(chunk)
                copied += len(chunk)
                print_copy_progress()
        shutil.copystat(src, dst)

    print_copy_progress(final=True)


def prepare_local_inputs() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if (
        not LOCAL_TEST_CHUNKS.exists()
        or LOCAL_TEST_CHUNKS.stat().st_size != DRIVE_TEST_CHUNKS.stat().st_size
    ):
        copy_file_with_progress(
            DRIVE_TEST_CHUNKS,
            LOCAL_TEST_CHUNKS,
            'Copy test_chunks.jsonl.gz',
        )

    if (
        not LOCAL_TEST_EMBEDDINGS.exists()
        or LOCAL_TEST_EMBEDDINGS.stat().st_size != DRIVE_TEST_EMBEDDINGS.stat().st_size
    ):
        copy_file_with_progress(
            DRIVE_TEST_EMBEDDINGS,
            LOCAL_TEST_EMBEDDINGS,
            'Copy test_embeddings.npy',
        )

    # ChromaDB gồm nhiều file; nếu local đã có thì ưu tiên dùng lại.
    if LOCAL_DB_DIR.exists():
        print(f"Dùng ChromaDB local có sẵn: {LOCAL_DB_DIR}")
    else:
        copytree_with_progress(
            DRIVE_DB_DIR,
            LOCAL_DB_DIR,
            'Copy ChromaDB',
        )



def local_db_fingerprint() -> dict:
    """Fingerprint nhẹ của ChromaDB, không gọi Collection.count()."""
    files = [path for path in LOCAL_DB_DIR.rglob('*') if path.is_file()]
    sqlite_path = LOCAL_DB_DIR / 'chroma.sqlite3'
    return {
        'file_count': len(files),
        'total_size': int(sum(path.stat().st_size for path in files)),
        'sqlite_size': int(sqlite_path.stat().st_size) if sqlite_path.exists() else None,
    }


def evaluation_signature(total_rows: int) -> dict:
    """Thông tin để chỉ resume khi input, DB và cấu hình evaluation vẫn khớp."""
    return {
        'test_chunks_size': LOCAL_TEST_CHUNKS.stat().st_size,
        'test_embeddings_size': LOCAL_TEST_EMBEDDINGS.stat().st_size,
        'total_rows': int(total_rows),
        'benign_collection': BENIGN_COLLECTION,
        'phishing_collection': PHISHING_COLLECTION,
        'db_fingerprint': local_db_fingerprint(),
        'top_k_per_label': TOP_K_PER_LABEL,
        'top_unique_source_per_chunk': TOP_UNIQUE_SOURCE_PER_CHUNK,
    }


def load_evaluation_checkpoint(total_rows: int):
    if not DRIVE_CHECKPOINT.exists():
        return None

    try:
        with DRIVE_CHECKPOINT.open('r', encoding='utf-8') as f:
            checkpoint = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f'Bỏ qua checkpoint evaluation không đọc được: {exc}')
        return None

    expected_signature = evaluation_signature(total_rows)
    next_row_index = checkpoint.get('next_row_index')
    if (
        checkpoint.get('version') != CHECKPOINT_VERSION
        or checkpoint.get('signature') != expected_signature
        or not isinstance(next_row_index, int)
        or not 0 <= next_row_index <= total_rows
        or not isinstance(checkpoint.get('predictions'), list)
    ):
        print('Bỏ qua checkpoint evaluation cũ vì không còn tương thích.')
        return None

    return checkpoint


def save_evaluation_checkpoint(
    completed_percent: int,
    next_row_index: int,
    total_rows: int,
    predictions,
    current_source,
    current_true_label,
    current_num_chunks: int,
    current_scores,
    finalized: bool = False,
) -> None:
    current_page = None
    if current_source is not None:
        current_page = {
            'source_file': current_source,
            'true_label': int(current_true_label),
            'num_query_chunks': int(current_num_chunks),
            'source_scores': {
                '0': dict(current_scores[0]),
                '1': dict(current_scores[1]),
            },
        }

    checkpoint = {
        'version': CHECKPOINT_VERSION,
        'completed_percent': int(completed_percent),
        'next_row_index': int(next_row_index),
        'total_rows': int(total_rows),
        'finalized': bool(finalized),
        'signature': evaluation_signature(total_rows),
        'predictions': predictions,
        'current_page': current_page,
    }

    DRIVE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DRIVE_CHECKPOINT.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    tmp.replace(DRIVE_CHECKPOINT)

    status = 'hoàn tất' if finalized else 'checkpoint'
    print(
        f'\n[{completed_percent}%] Đã lưu {status} evaluation lên Drive: '
        f'{DRIVE_CHECKPOINT}',
        flush=True,
    )


def restore_evaluation_state(checkpoint):
    if checkpoint is None:
        return (
            0,
            [],
            None,
            None,
            0,
            {0: defaultdict(float), 1: defaultdict(float)},
        )

    current_page = checkpoint.get('current_page')
    if current_page is None:
        current_source = None
        current_true_label = None
        current_num_chunks = 0
        current_scores = {0: defaultdict(float), 1: defaultdict(float)}
    else:
        saved_scores = current_page.get('source_scores', {})
        current_source = current_page['source_file']
        current_true_label = int(current_page['true_label'])
        current_num_chunks = int(current_page['num_query_chunks'])
        current_scores = {
            0: defaultdict(
                float,
                {k: float(v) for k, v in saved_scores.get('0', {}).items()},
            ),
            1: defaultdict(
                float,
                {k: float(v) for k, v in saved_scores.get('1', {}).items()},
            ),
        }

    return (
        int(checkpoint['next_row_index']),
        list(checkpoint['predictions']),
        current_source,
        current_true_label,
        current_num_chunks,
        current_scores,
    )


def load_test_metadata():
    rows = []
    with gzip.open(LOCAL_TEST_CHUNKS, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            rows.append({
                'source_file': row['source_file'],
                'chunk_index': int(row['chunk_index']),
                'label': int(row['label']),
            })
    return rows


def cosine_similarity_from_chroma_distance(distance: float) -> float:
    return float(np.clip(1.0 - float(distance), 0.0, 1.0))


def deduplicate_top_unique(label_items):
    """label_items đã là top-K chunk theo similarity của đúng một label."""
    best_by_source = {}
    for source_file, similarity in label_items:
        if similarity > best_by_source.get(source_file, -1.0):
            best_by_source[source_file] = similarity
    return sorted(
        best_by_source.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_UNIQUE_SOURCE_PER_CHUNK]


def saturating_add(old_score: float, similarity: float) -> float:
    return 1.0 - (1.0 - old_score) * (1.0 - similarity)


def query_collection_evidence(collection, query_embeddings):
    """
    Query trực tiếp một collection đã tách label với n_results=50.
    Sau retrieval, các chunk cùng source_file được gộp bằng max similarity,
    rồi chỉ giữ top-10 unique source mạnh nhất cho mỗi query chunk.
    """
    batch_size = len(query_embeddings)
    evidence = [None] * batch_size

    # Chroma lấy metadata qua SQLite sau ANN search. Giới hạn số kết quả mỗi
    # query call để tránh lỗi "too many SQL variables".
    safe_batch_size = max(1, MAX_SQL_RESULTS_PER_QUERY // TOP_K_PER_LABEL)

    for sub_start in range(0, batch_size, safe_batch_size):
        sub_end = min(sub_start + safe_batch_size, batch_size)
        sub_embeddings = np.asarray(
            query_embeddings[sub_start:sub_end], dtype=np.float32
        )

        result = collection.query(
            query_embeddings=sub_embeddings.tolist(),
            n_results=TOP_K_PER_LABEL,
            include=['metadatas', 'distances'],
        )

        for local_i, (metadatas, distances) in enumerate(
            zip(result['metadatas'], result['distances'])
        ):
            label_items = [
                (
                    meta['source_file'],
                    cosine_similarity_from_chroma_distance(distance),
                )
                for meta, distance in zip(metadatas, distances)
            ]
            evidence[sub_start + local_i] = deduplicate_top_unique(label_items)

    return evidence


def retrieve_batch_evidence(benign_collection, phishing_collection, query_embeddings):
    """
    Với mỗi query chunk, retrieve độc lập:
      - top-50 từ html_chunks_benign
      - top-50 từ html_chunks_phishing

    Không còn global top-K, metadata filter theo label hay vòng lặp tăng K.
    """
    benign_evidence = query_collection_evidence(
        benign_collection, query_embeddings
    )
    phishing_evidence = query_collection_evidence(
        phishing_collection, query_embeddings
    )

    return [
        {0: benign_evidence[i], 1: phishing_evidence[i]}
        for i in range(len(query_embeddings))
    ]


def open_split_collections():
    """
    Mở hai collection mới. Nếu ChromaDB local còn là schema cũ, tự refresh
    từ Drive một lần rồi mở lại.
    """
    def open_from_local():
        client = chromadb.PersistentClient(path=str(LOCAL_DB_DIR))
        benign = client.get_collection(BENIGN_COLLECTION)
        phishing = client.get_collection(PHISHING_COLLECTION)
        return client, benign, phishing

    try:
        return open_from_local()
    except Exception as exc:
        print(
            'ChromaDB local không có đủ hai collection mới; '
            'copy lại DB từ Drive...'
        )
        if LOCAL_DB_DIR.exists():
            shutil.rmtree(LOCAL_DB_DIR)
        copytree_with_progress(
            DRIVE_DB_DIR,
            LOCAL_DB_DIR,
            'Copy ChromaDB',
        )
        try:
            return open_from_local()
        except Exception:
            raise RuntimeError(
                f'Không mở được {BENIGN_COLLECTION} và {PHISHING_COLLECTION}. '
                'Hãy chạy build_jina_vector_db.py bản mới trước.'
            ) from exc


def finalize_page(source_file, true_label, num_query_chunks, source_scores):
    benign_score = float(sum(source_scores[0].values()))
    phishing_score = float(sum(source_scores[1].values()))
    total_score = benign_score + phishing_score
    phishing_probability = phishing_score / total_score if total_score > 0 else 0.5
    prediction = 1 if phishing_score > benign_score else 0

    return {
        'source_file': source_file,
        'true_label': true_label,
        'predicted_label': prediction,
        'num_query_chunks': num_query_chunks,
        'benign_score': benign_score,
        'phishing_score': phishing_score,
        'phishing_probability': phishing_probability,
        'unique_benign_sources': len(source_scores[0]),
        'unique_phishing_sources': len(source_scores[1]),
        'tie': int(np.isclose(phishing_score, benign_score)),
    }


def evaluate_rag():
    prepare_local_inputs()

    rows = load_test_metadata()
    embeddings = np.load(LOCAL_TEST_EMBEDDINGS, mmap_mode='r')
    if len(rows) != len(embeddings):
        raise ValueError(
            f'Số test chunk ({len(rows)}) không khớp số embedding ({len(embeddings)})'
        )

    client, benign_collection, phishing_collection = open_split_collections()
    print(f'Benign collection:   {BENIGN_COLLECTION}')
    print(f'Phishing collection: {PHISHING_COLLECTION}')

    checkpoint = load_evaluation_checkpoint(len(rows))
    (
        start_row,
        predictions,
        current_source,
        current_true_label,
        current_num_chunks,
        current_scores,
    ) = restore_evaluation_state(checkpoint)
    finished_pages = len(predictions)

    if checkpoint is not None:
        print(
            f"Resume evaluation từ chunk {start_row:,}/{len(rows):,} "
            f"({100.0 * start_row / len(rows):.2f}%), "
            f"{finished_pages:,} page đã hoàn tất."
        )
    else:
        print('Không có checkpoint evaluation hợp lệ. Chạy từ 0%.')

    checkpoint_rows = {
        percent: min(
            len(rows),
            (len(rows) * percent + 99) // 100,
        )
        for percent in CHECKPOINT_PERCENTS
    }
    pending_percents = [
        percent
        for percent in CHECKPOINT_PERCENTS
        if checkpoint_rows[percent] > start_row
    ]
    checkpoint_cursor = 0

    progress_started_at = time.perf_counter()
    last_progress_len = 0

    def print_progress(current_row: int, final: bool = False) -> None:
        nonlocal last_progress_len

        elapsed = max(time.perf_counter() - progress_started_at, 0.0)
        run_rows = max(current_row - start_row, 0)
        rate = run_rows / elapsed if elapsed > 0 else 0.0
        remaining_rows = max(len(rows) - current_row, 0)
        eta_text = (
            format_duration(remaining_rows / rate)
            if rate > 0
            else 'calculating...'
        )
        elapsed_text = format_duration(elapsed)
        percent = 100.0 * current_row / len(rows) if rows else 100.0

        line = (
            f'RAG retrieval: {current_row:,}/{len(rows):,} ({percent:6.2f}%) | '
            f'{rate:,.1f} chunk/s | pages {finished_pages:,} | '
            f'elapsed {elapsed_text} | ETA {eta_text}'
        )
        padded_line = line.ljust(last_progress_len)
        print('\r' + padded_line, end='\n' if final else '', flush=True)
        last_progress_len = len(line)

    processed_rows = start_row
    print_progress(start_row)

    for start in range(start_row, len(rows), QUERY_BATCH_SIZE):
        end = min(start + QUERY_BATCH_SIZE, len(rows))
        query_embeddings = np.asarray(embeddings[start:end], dtype=np.float32)
        batch_evidence = retrieve_batch_evidence(
            benign_collection, phishing_collection, query_embeddings
        )

        for offset, evidence in enumerate(batch_evidence):
            row_index = start + offset
            row = rows[row_index]
            source_file = row['source_file']
            true_label = row['label']

            if current_source is None:
                current_source = source_file
                current_true_label = true_label

            if source_file != current_source:
                predictions.append(finalize_page(
                    current_source,
                    current_true_label,
                    current_num_chunks,
                    current_scores,
                ))
                finished_pages += 1
                current_source = source_file
                current_true_label = true_label
                current_num_chunks = 0
                current_scores = {0: defaultdict(float), 1: defaultdict(float)}

            if true_label != current_true_label:
                raise ValueError(f'Page {source_file} có nhiều label')

            current_num_chunks += 1
            for label in (0, 1):
                for train_source, similarity in evidence[label]:
                    current_scores[label][train_source] = saturating_add(
                        current_scores[label][train_source],
                        similarity,
                    )

            processed_rows = row_index + 1

            while (
                checkpoint_cursor < len(pending_percents)
                and processed_rows >= checkpoint_rows[
                    pending_percents[checkpoint_cursor]
                ]
            ):
                percent = pending_percents[checkpoint_cursor]
                save_evaluation_checkpoint(
                    completed_percent=percent,
                    next_row_index=processed_rows,
                    total_rows=len(rows),
                    predictions=predictions,
                    current_source=current_source,
                    current_true_label=current_true_label,
                    current_num_chunks=current_num_chunks,
                    current_scores=current_scores,
                    finalized=False,
                )
                checkpoint_cursor += 1
                print_progress(processed_rows)

        print_progress(processed_rows)

    if current_source is not None:
        predictions.append(finalize_page(
            current_source,
            current_true_label,
            current_num_chunks,
            current_scores,
        ))
        finished_pages += 1
        current_source = None
        current_true_label = None
        current_num_chunks = 0
        current_scores = {0: defaultdict(float), 1: defaultdict(float)}

    print_progress(len(rows), final=True)

    y_true = [row['true_label'] for row in predictions]
    y_pred = [row['predicted_label'] for row in predictions]
    y_score = [row['phishing_probability'] for row in predictions]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        'num_pages': len(y_true),
        'num_test_chunks': len(rows),
        'top_k_per_label': TOP_K_PER_LABEL,
        'top_unique_source_per_chunk': TOP_UNIQUE_SOURCE_PER_CHUNK,
        'query_batch_size': QUERY_BATCH_SIZE,
        'benign_collection': BENIGN_COLLECTION,
        'phishing_collection': PHISHING_COLLECTION,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_phishing': precision_score(y_true, y_pred, zero_division=0),
        'recall_phishing': recall_score(y_true, y_pred, zero_division=0),
        'specificity_benign': specificity,
        'f1_phishing': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        },
        'tie_pages': int(sum(row['tie'] for row in predictions)),
        'classification_report': classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=['benign', 'phishing'],
            zero_division=0,
            output_dict=True,
        ),
    }

    with open(DRIVE_PREDICTIONS, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)

    with open(DRIVE_METRICS, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Ghi lại checkpoint 100% ở trạng thái đã finalize toàn bộ page/output.
    save_evaluation_checkpoint(
        completed_percent=100,
        next_row_index=len(rows),
        total_rows=len(rows),
        predictions=predictions,
        current_source=None,
        current_true_label=None,
        current_num_chunks=0,
        current_scores={0: defaultdict(float), 1: defaultdict(float)},
        finalized=True,
    )

    print('\n=== RAG VOTING EVALUATION ===')
    print(f"Pages:               {metrics['num_pages']}")
    print(f"Test chunks:         {metrics['num_test_chunks']}")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:   {metrics['balanced_accuracy']:.4f}")
    print(f"Precision phishing:  {metrics['precision_phishing']:.4f}")
    print(f"Recall phishing:     {metrics['recall_phishing']:.4f}")
    print(f"Specificity benign:  {metrics['specificity_benign']:.4f}")
    print(f"F1 phishing:         {metrics['f1_phishing']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f'Confusion matrix:     TN={tn}, FP={fp}, FN={fn}, TP={tp}')
    print(f"Tie pages:            {metrics['tie_pages']}")
    print(f'Predictions:          {DRIVE_PREDICTIONS}')
    print(f'Metrics:              {DRIVE_METRICS}')
    print(f'Checkpoint:           {DRIVE_CHECKPOINT}')

    return metrics, predictions


if __name__ == '__main__':
    evaluate_rag()
