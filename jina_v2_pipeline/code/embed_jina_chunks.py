import gzip
import json
import shutil
import threading
from pathlib import Path
from queue import Queue

import numpy as np
import torch
from transformers import AutoModel
from tqdm.auto import tqdm

DATA_DIR = Path("/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/data")
DRIVE_CHUNKS_FILE = DATA_DIR / "chunks" / "test_chunks.jsonl.gz"
DRIVE_EMBED_DIR = DATA_DIR / "embeddings"
DRIVE_BLOCK_DIR = DRIVE_EMBED_DIR / "test_blocks"
DRIVE_OUT_FILE = DRIVE_EMBED_DIR / "test_embeddings.npy"
DRIVE_TMP_OUT_FILE = DRIVE_EMBED_DIR / "test_embeddings.tmp.npy"

LOCAL_DIR = Path("/content/jina_v2_embedding")
LOCAL_GZ = LOCAL_DIR / "test_chunks.jsonl.gz"
LOCAL_JSONL = LOCAL_DIR / "test_chunks.jsonl"
LOCAL_BLOCK_FILE = LOCAL_DIR / "block.npy"
LOCAL_OUT_FILE = LOCAL_DIR / "test_embeddings.npy"

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
BATCH_SIZE = 256
QUEUE_SIZE = 4
BLOCK_ROWS = 65536

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_EMBED_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_BLOCK_DIR.mkdir(parents=True, exist_ok=True)

if not LOCAL_GZ.exists() or LOCAL_GZ.stat().st_size != DRIVE_CHUNKS_FILE.stat().st_size:
    shutil.copy2(DRIVE_CHUNKS_FILE, LOCAL_GZ)

total = 0
with gzip.open(LOCAL_GZ, "rt", encoding="utf-8") as src, open(LOCAL_JSONL, "wt", encoding="utf-8") as dst:
    for line in src:
        dst.write(line)
        total += 1

if total == 0:
    raise ValueError("Không có chunk")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=dtype
).to(device)
model.eval()
dim = model.config.hidden_size


def valid_array(path, shape):
    try:
        arr = np.load(path, mmap_mode="r")
        ok = arr.shape == shape and arr.dtype == np.float32
        del arr
        return ok
    except Exception:
        return False


if DRIVE_OUT_FILE.exists() and valid_array(DRIVE_OUT_FILE, (total, dim)):
    print(f"Exists: {DRIVE_OUT_FILE}")
    raise SystemExit

for tmp in DRIVE_BLOCK_DIR.glob("*.tmp.npy"):
    tmp.unlink(missing_ok=True)

position = 0
while position < total:
    rows = min(BLOCK_ROWS, total - position)
    path = DRIVE_BLOCK_DIR / f"block_{position:06d}.npy"
    if not path.exists():
        break
    if not valid_array(path, (rows, dim)):
        path.unlink(missing_ok=True)
        break
    position += rows

start_position = position
queue = Queue(maxsize=QUEUE_SIZE)
producer_error = []


def producer():
    try:
        with open(LOCAL_JSONL, "rt", encoding="utf-8") as f:
            for _ in range(start_position):
                next(f)
            batch = []
            for line in f:
                batch.append(json.loads(line)["document"])
                if len(batch) == BATCH_SIZE:
                    queue.put(batch)
                    batch = []
            if batch:
                queue.put(batch)
    except BaseException as e:
        producer_error.append(e)
    finally:
        queue.put(None)


def save_block(block, start):
    final_path = DRIVE_BLOCK_DIR / f"block_{start:06d}.npy"
    tmp_path = DRIVE_BLOCK_DIR / f"block_{start:06d}.tmp.npy"
    np.save(LOCAL_BLOCK_FILE, block, allow_pickle=False)
    tmp_path.unlink(missing_ok=True)
    shutil.copy2(LOCAL_BLOCK_FILE, tmp_path)
    if not valid_array(tmp_path, block.shape):
        raise IOError(f"Checkpoint block lỗi: {tmp_path}")
    tmp_path.replace(final_path)
    LOCAL_BLOCK_FILE.unlink(missing_ok=True)


threading.Thread(target=producer, daemon=True).start()
pbar = tqdm(initial=position, total=total, desc="Embedding", unit="chunk")
buffer = []
buffer_rows = 0

while True:
    batch = queue.get()
    if batch is None:
        break

    with torch.inference_mode():
        emb = np.asarray(model.encode(batch, batch_size=BATCH_SIZE), dtype=np.float32)

    buffer.append(emb)
    buffer_rows += len(emb)
    pbar.update(len(emb))

    if buffer_rows >= BLOCK_ROWS:
        block = np.concatenate(buffer)
        save_block(block, position)
        position += len(block)
        buffer.clear()
        buffer_rows = 0

if producer_error:
    raise producer_error[0]

if buffer:
    block = np.concatenate(buffer)
    save_block(block, position)
    position += len(block)

pbar.close()

if position != total:
    raise ValueError("Số embedding không khớp số chunk")

out = np.lib.format.open_memmap(
    LOCAL_OUT_FILE,
    mode="w+",
    dtype=np.float32,
    shape=(total, dim)
)

position = 0
while position < total:
    rows = min(BLOCK_ROWS, total - position)
    path = DRIVE_BLOCK_DIR / f"block_{position:06d}.npy"
    block = np.load(path, mmap_mode="r")
    out[position:position + rows] = block
    position += rows

a = out
out.flush()
del a
del out

DRIVE_TMP_OUT_FILE.unlink(missing_ok=True)
shutil.copy2(LOCAL_OUT_FILE, DRIVE_TMP_OUT_FILE)
if not valid_array(DRIVE_TMP_OUT_FILE, (total, dim)):
    raise IOError(f"Output merge lỗi: {DRIVE_TMP_OUT_FILE}")
DRIVE_OUT_FILE.unlink(missing_ok=True)
DRIVE_TMP_OUT_FILE.replace(DRIVE_OUT_FILE)
LOCAL_OUT_FILE.unlink(missing_ok=True)
print(f"Saved: {DRIVE_OUT_FILE}")
