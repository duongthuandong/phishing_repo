import gzip
import json
import math
import shutil
import time
from itertools import islice
from pathlib import Path

import chromadb
import numpy as np

# Google Drive: nguồn dữ liệu và nơi lưu snapshot ChromaDB để resume.
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/data")
DRIVE_CHUNKS_FILE = DRIVE_DATA_DIR / "chunks" / "train_chunks.jsonl.gz"
DRIVE_EMBEDDINGS_FILE = DRIVE_DATA_DIR / "embeddings" / "train_embeddings.npy"
DRIVE_DB_DIR = DRIVE_DATA_DIR / "chromadb"
DRIVE_DB_TMP_DIR = DRIVE_DATA_DIR / "chromadb_tmp"

# SSD local của Colab: build ChromaDB ở đây để nhanh hơn.
LOCAL_WORK_DIR = Path("/content/jina_vector_db")
LOCAL_CHUNKS_FILE = LOCAL_WORK_DIR / "train_chunks.jsonl.gz"
LOCAL_EMBEDDINGS_FILE = LOCAL_WORK_DIR / "train_embeddings.npy"
LOCAL_DB_DIR = LOCAL_WORK_DIR / "chromadb"
LOCAL_CHECKPOINT_FILE = LOCAL_DB_DIR / "build_checkpoint.json"
DRIVE_CHECKPOINT_FILE = DRIVE_DB_DIR / "build_checkpoint.json"

BENIGN_COLLECTION = "html_chunks_benign"
PHISHING_COLLECTION = "html_chunks_phishing"

BATCH_SIZE = 5000
PROGRESS_EVERY_ROWS = 1000
COPY_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MiB
CHECKPOINT_PERCENTS = (20, 40, 60, 80, 100)

HNSW_CONFIGURATION = {
    "hnsw": {
        "space": "cosine",
        "batch_size": 5000,
        "sync_threshold": 50000,
    }
}


def copy_file_with_progress(src: Path, dst: Path) -> None:
    """Copy tuần tự một file lớn và hiển thị tiến độ trên một dòng."""
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
        eta = remaining / rate if rate > 0 else math.inf
        percent = 100.0 * copied / total if total else 100.0

        line = (
            f"Copy {src.name}: {copied / (1024**2):,.1f}/{total / (1024**2):,.1f} MiB "
            f"({percent:6.2f}%) | {rate / (1024**2):,.1f} MiB/s | "
            f"ETA {format_duration(eta) if math.isfinite(eta) else 'calculating...'}"
        )
        padded_line = line.ljust(last_progress_len)
        print("\r" + padded_line, end="\n" if final else "", flush=True)
        last_progress_len = len(line)

    print_copy_progress()
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        while True:
            chunk = fsrc.read(COPY_BUFFER_SIZE)
            if not chunk:
                break
            fdst.write(chunk)
            copied += len(chunk)
            print_copy_progress()

    print_copy_progress(final=True)
    shutil.copystat(src, dst)


def format_duration(seconds: float) -> str:
    """Định dạng số giây thành MM:SS hoặc H:MM:SS."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def prepare_local_inputs() -> None:
    """Chỉ copy chunks/embeddings từ Drive khi bản local chưa tồn tại."""
    LOCAL_WORK_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_CHUNKS_FILE.exists():
        print(f"Dùng chunks local có sẵn: {LOCAL_CHUNKS_FILE}")
    else:
        print(f"Copy chunks từ Drive xuống local: {LOCAL_CHUNKS_FILE}")
        copy_file_with_progress(DRIVE_CHUNKS_FILE, LOCAL_CHUNKS_FILE)

    if LOCAL_EMBEDDINGS_FILE.exists():
        print(f"Dùng embeddings local có sẵn: {LOCAL_EMBEDDINGS_FILE}")
    else:
        print(f"Copy embeddings từ Drive xuống local: {LOCAL_EMBEDDINGS_FILE}")
        copy_file_with_progress(DRIVE_EMBEDDINGS_FILE, LOCAL_EMBEDDINGS_FILE)


def input_signature() -> dict:
    """Thông tin tối thiểu để tránh resume nhầm sang input khác."""
    return {
        "chunks_size": LOCAL_CHUNKS_FILE.stat().st_size,
        "embeddings_size": LOCAL_EMBEDDINGS_FILE.stat().st_size,
    }


def load_checkpoint(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def checkpoint_is_compatible(checkpoint: dict | None, total_rows: int) -> bool:
    if not checkpoint:
        return False
    return (
        checkpoint.get("total_rows") == total_rows
        and checkpoint.get("input_signature") == input_signature()
        and isinstance(checkpoint.get("next_row_index"), int)
        and 0 <= checkpoint["next_row_index"] <= total_rows
    )


def prepare_local_db_for_resume(total_rows: int) -> int:
    """
    Chọn checkpoint tốt nhất để resume:
      1) local DB nếu có checkpoint hợp lệ;
      2) nếu local không dùng được, tải snapshot gần nhất từ Drive;
      3) nếu không có checkpoint hợp lệ thì build mới từ 0.
    """
    local_checkpoint = load_checkpoint(LOCAL_CHECKPOINT_FILE)
    if LOCAL_DB_DIR.exists() and checkpoint_is_compatible(local_checkpoint, total_rows):
        start_row = local_checkpoint["next_row_index"]
        print(
            f"Resume từ ChromaDB local: row {start_row}/{total_rows} "
            f"({100.0 * start_row / total_rows:.1f}%)"
        )
        return start_row

    drive_checkpoint = load_checkpoint(DRIVE_CHECKPOINT_FILE)
    if DRIVE_DB_DIR.exists() and checkpoint_is_compatible(drive_checkpoint, total_rows):
        if LOCAL_DB_DIR.exists():
            shutil.rmtree(LOCAL_DB_DIR)
        print(
            f"Tải checkpoint ChromaDB từ Drive: row "
            f"{drive_checkpoint['next_row_index']}/{total_rows} "
            f"({100.0 * drive_checkpoint['next_row_index'] / total_rows:.1f}%)"
        )
        shutil.copytree(DRIVE_DB_DIR, LOCAL_DB_DIR)
        return int(drive_checkpoint["next_row_index"])

    if LOCAL_DB_DIR.exists():
        shutil.rmtree(LOCAL_DB_DIR)
    LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
    print("Không có checkpoint hợp lệ. Build ChromaDB mới từ 0%.")
    return 0


def write_checkpoint(next_row_index: int, total_rows: int, completed_percent: int) -> None:
    checkpoint = {
        "next_row_index": int(next_row_index),
        "total_rows": int(total_rows),
        "completed_percent": int(completed_percent),
        "input_signature": input_signature(),
        "benign_collection": BENIGN_COLLECTION,
        "phishing_collection": PHISHING_COLLECTION,
    }
    LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LOCAL_CHECKPOINT_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    tmp.replace(LOCAL_CHECKPOINT_FILE)


def sync_db_snapshot_to_drive(percent: int) -> None:
    """Snapshot DB local lên Drive tại các mốc 20%, 40%, 60%, 80%, 100%."""
    print(f"\n[{percent}%] Đang upload checkpoint ChromaDB lên Google Drive...")

    if DRIVE_DB_TMP_DIR.exists():
        shutil.rmtree(DRIVE_DB_TMP_DIR)

    # Copy sang thư mục tạm trước. Chỉ thay snapshot cũ sau khi copy hoàn tất.
    shutil.copytree(LOCAL_DB_DIR, DRIVE_DB_TMP_DIR)

    if DRIVE_DB_DIR.exists():
        shutil.rmtree(DRIVE_DB_DIR)

    shutil.move(str(DRIVE_DB_TMP_DIR), str(DRIVE_DB_DIR))
    print(f"[{percent}%] Đã lưu checkpoint: {DRIVE_DB_DIR}")


def open_collections(client, resume: bool):
    if resume:
        return {
            0: client.get_collection(BENIGN_COLLECTION),
            1: client.get_collection(PHISHING_COLLECTION),
        }

    return {
        0: client.get_or_create_collection(
            name=BENIGN_COLLECTION,
            metadata={"label": 0},
            configuration=HNSW_CONFIGURATION,
        ),
        1: client.get_or_create_collection(
            name=PHISHING_COLLECTION,
            metadata={"label": 1},
            configuration=HNSW_CONFIGURATION,
        ),
    }


def build_chromadb() -> tuple[int, int]:
    """
    Build hai vector index riêng trong cùng một ChromaDB và checkpoint mỗi 20%.

    Resume dùng `next_row_index` của toàn bộ train_chunks.jsonl.gz. Trước khi ghi
    checkpoint, cả hai buffer đều được flush để mọi row <= checkpoint đã nằm trong DB.
    """
    embeddings = np.load(LOCAL_EMBEDDINGS_FILE, mmap_mode="r")
    total_rows = len(embeddings)
    start_row = prepare_local_db_for_resume(total_rows)

    client = chromadb.PersistentClient(path=str(LOCAL_DB_DIR))
    collections = open_collections(client, resume=start_row > 0)

    if start_row > 0:
        existing_count = collections[0].count() + collections[1].count()
        if existing_count < start_row:
            raise RuntimeError(
                "Checkpoint không nhất quán: số record trong ChromaDB nhỏ hơn "
                f"next_row_index ({existing_count} < {start_row})."
            )
        if existing_count > start_row:
            print(
                f"Local DB đang có {existing_count} record, checkpoint ở row {start_row}. "
                "Các row sau checkpoint sẽ được upsert lại an toàn."
            )

    buffers = {
        0: {"ids": [], "docs": [], "metas": [], "indices": []},
        1: {"ids": [], "docs": [], "metas": [], "indices": []},
    }
    def flush(label: int) -> None:
        buffer = buffers[label]
        if not buffer["ids"]:
            return

        indices = np.asarray(buffer["indices"], dtype=np.int64)
        collections[label].upsert(
            ids=buffer["ids"],
            documents=buffer["docs"],
            embeddings=np.asarray(embeddings[indices], dtype=np.float32),
            metadatas=buffer["metas"],
        )

        buffer["ids"].clear()
        buffer["docs"].clear()
        buffer["metas"].clear()
        buffer["indices"].clear()

    checkpoint_rows = {
        percent: min(total_rows, math.ceil(total_rows * percent / 100.0))
        for percent in CHECKPOINT_PERCENTS
    }
    pending_percents = [
        percent
        for percent in CHECKPOINT_PERCENTS
        if checkpoint_rows[percent] > start_row
    ]
    checkpoint_cursor = 0

    processed_rows = start_row
    progress_started_at = time.perf_counter()
    last_progress_len = 0

    def print_progress(current_row: int, final: bool = False) -> None:
        nonlocal last_progress_len

        elapsed = max(time.perf_counter() - progress_started_at, 0.0)
        run_rows = max(current_row - start_row, 0)
        rate = run_rows / elapsed if elapsed > 0 else 0.0
        remaining_rows = max(total_rows - current_row, 0)
        eta_text = (
            format_duration(remaining_rows / rate)
            if rate > 0
            else "calculating..."
        )
        elapsed_text = format_duration(elapsed)
        percent = 100.0 * current_row / total_rows if total_rows else 100.0

        line = (
            f"Progress: {current_row:,}/{total_rows:,} ({percent:6.2f}%) | "
            f"{rate:,.1f} row/s | elapsed {elapsed_text} | ETA {eta_text}"
        )
        padded_line = line.ljust(last_progress_len)
        print("\r" + padded_line, end="\n" if final else "", flush=True)
        last_progress_len = len(line)

    # In ngay trạng thái ban đầu; ETA sẽ xuất hiện sau khi đã xử lý được một số row.
    print_progress(start_row)

    with gzip.open(LOCAL_CHUNKS_FILE, "rt", encoding="utf-8") as f:
        remaining_lines = islice(f, start_row, None)

        for row_index, line in enumerate(remaining_lines, start=start_row):
            if row_index >= total_rows:
                raise ValueError("Số chunk lớn hơn số embedding")

            row = json.loads(line)
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Label không hợp lệ tại dòng {row_index}: {label}")

            buffer = buffers[label]
            buffer["ids"].append(row["id"])
            buffer["docs"].append(row["document"])
            buffer["metas"].append({
                "source_file": row["source_file"],
                "chunk_index": row["chunk_index"],
                "token_count": row["token_count"],
                "label": label,
            })
            buffer["indices"].append(row_index)

            if len(buffer["ids"]) >= BATCH_SIZE:
                flush(label)

            processed_rows = row_index + 1
            rows_this_run = processed_rows - start_row
            if (
                rows_this_run % PROGRESS_EVERY_ROWS == 0
                or processed_rows == total_rows
            ):
                print_progress(processed_rows)

            # Có thể vượt mốc vài row vì checkpoint không nhất thiết trùng batch.
            while (
                checkpoint_cursor < len(pending_percents)
                and processed_rows >= checkpoint_rows[pending_percents[checkpoint_cursor]]
            ):
                percent = pending_percents[checkpoint_cursor]

                # Đảm bảo toàn bộ row đến checkpoint đều đã persist vào ChromaDB.
                flush(0)
                flush(1)
                write_checkpoint(processed_rows, total_rows, percent)
                sync_db_snapshot_to_drive(percent)
                checkpoint_cursor += 1
                print_progress(processed_rows)

    flush(0)
    flush(1)
    print_progress(processed_rows, final=True)

    if processed_rows != total_rows:
        raise ValueError(
            f"Số chunk đã xử lý ({processed_rows}) không khớp số embedding ({total_rows})"
        )

    benign_count = collections[0].count()
    phishing_count = collections[1].count()

    if benign_count + phishing_count != total_rows:
        raise ValueError(
            "Tổng record của hai collection không khớp số embedding: "
            f"{benign_count} + {phishing_count} != {total_rows}"
        )

    # Trường hợp resume từ checkpoint 100%: không còn pending checkpoint để trigger.
    final_checkpoint = load_checkpoint(LOCAL_CHECKPOINT_FILE)
    if not final_checkpoint or final_checkpoint.get("next_row_index") != total_rows:
        write_checkpoint(total_rows, total_rows, 100)
        sync_db_snapshot_to_drive(100)

    print(f"Benign records:   {benign_count} ({BENIGN_COLLECTION})")
    print(f"Phishing records: {phishing_count} ({PHISHING_COLLECTION})")
    print(f"Total records:    {benign_count + phishing_count}")
    print("Local Vector DB:", LOCAL_DB_DIR)
    print("Drive Vector DB:", DRIVE_DB_DIR)

    return benign_count, phishing_count


def main() -> None:
    prepare_local_inputs()
    build_chromadb()


if __name__ == "__main__":
    main()
