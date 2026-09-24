# Distill Jina v2 base-code -> 6-layer student bằng MSE thuần.
# Teacher target: train_embeddings.npy hiện có từ model.encode().
# Không L2-normalize trước loss; mặc định train 1 epoch.
# Có checkpoint/resume trên Google Drive và progress bằng print đè một dòng.
import gzip
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

# ============================================================
# Paths: bám theo jina_v2_pipeline hiện tại trên Google Drive
# ============================================================
ROOT_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline')
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'

DRIVE_CHUNKS = DATA_DIR / 'chunks' / 'train_chunks.jsonl.gz'
DRIVE_TEACHER_EMB = DATA_DIR / 'embeddings' / 'train_embeddings.npy'
DRIVE_OUT_DIR = MODELS_DIR / 'jina_v2_base_code_distill_6l_mse'

# Checkpoint để riêng khỏi final model, tránh bị ghi đè khi save model cuối.
DRIVE_CHECKPOINT_DIR = MODELS_DIR / 'jina_v2_base_code_distill_6l_mse_checkpoint'
DRIVE_CHECKPOINT_TMP_DIR = MODELS_DIR / 'jina_v2_base_code_distill_6l_mse_checkpoint_tmp'

LOCAL_DIR = Path('/content/jina_v2_distill')
LOCAL_CHUNKS = LOCAL_DIR / 'train_chunks.jsonl.gz'
LOCAL_TEACHER_EMB = LOCAL_DIR / 'train_embeddings.npy'
LOCAL_OUT_DIR = LOCAL_DIR / 'model'
LOCAL_CHECKPOINT_DIR = LOCAL_DIR / 'checkpoint'

TEACHER_MODEL = 'jinaai/jina-embeddings-v2-base-code'
NUM_STUDENT_LAYERS = 6
MAX_LENGTH = 512

# Colab-safe defaults. Với GPU lớn hơn có thể tăng BATCH_SIZE.
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0

# Bounded streaming shuffle; giữ alignment với teacher embedding bằng row index.
SHUFFLE_BUFFER = 8192

# Length bucketing: gom chunk có token_count gần nhau để giảm padding.
LENGTH_BUCKET_BUFFER = 8192
DATA_ORDER_VERSION = 2
SEED = 42

# Mỗi 10% của epoch lưu checkpoint lên Drive.
# Checkpoint chỉ được lưu ngay sau optimizer.step(), nên không có gradient dở dang.
CHECKPOINT_EVERY_PERCENT = 10

# Cập nhật dòng progress sau mỗi N batch. Dùng print(..., end='\r'), không dùng tqdm.
PROGRESS_EVERY_BATCHES = 50

# False = tự resume checkpoint nếu có.
# Đặt True khi muốn train lại từ đầu và xoá checkpoint cũ.
RESET_CHECKPOINT = False

# Distillation loss: MSE thuần trên raw teacher embedding.
# Không L2-normalize trước loss.

# Đặt None để dùng toàn bộ train chunks.
# Có thể đặt 1_000_000 cho vòng thử nghiệm đầu tiên.
MAX_TRAIN_ROWS = None

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print(f'Use local: {dst}')
        return
    print(f'Copy {src} -> {dst}')
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    tmp.unlink(missing_ok=True)
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def resolve_encoder_layers(model):
    """Tìm ModuleList chứa transformer blocks, tránh hard-code riêng JinaBert."""
    candidates = [
        ('encoder', 'layer'),
        ('encoder', 'layers'),
        ('bert', 'encoder', 'layer'),
        ('model', 'encoder', 'layer'),
        ('model', 'encoder', 'layers'),
    ]
    for path in candidates:
        parent = model
        ok = True
        for name in path[:-1]:
            if not hasattr(parent, name):
                ok = False
                break
            parent = getattr(parent, name)
        if not ok or not hasattr(parent, path[-1]):
            continue
        layers = getattr(parent, path[-1])
        if isinstance(layers, nn.ModuleList):
            return parent, path[-1], layers
    raise RuntimeError('Không tìm thấy transformer layer list trong model Jina.')


def prune_to_student(model, num_layers: int):
    parent, attr, layers = resolve_encoder_layers(model)
    full_layers = len(layers)
    if not 1 <= num_layers <= full_layers:
        raise ValueError(f'num_layers={num_layers}, teacher has {full_layers}')

    # Với Jina base 12 -> 6: lấy layer cuối của mỗi block 2 layer.
    # Kết quả [1, 3, 5, 7, 9, 11], giữ được final teacher layer.
    if full_layers == 12 and num_layers == 6:
        keep = [1, 3, 5, 7, 9, 11]
    else:
        keep = np.linspace(0, full_layers - 1, num_layers).round().astype(int).tolist()

    keep = list(dict.fromkeys(keep))
    if len(keep) != num_layers:
        raise RuntimeError(f'Layer selection bị trùng: {keep}')

    new_layers = nn.ModuleList([layers[i] for i in keep])
    setattr(parent, attr, new_layers)
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = num_layers

    print(f'Pruned encoder: {full_layers} -> {num_layers} layers, keep={keep}')
    return keep


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def encode_student(model, tokenizer, texts, device):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt',
    )
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    outputs = model(**batch, return_dict=True)

    # Jina model.encode() mặc định dùng mean pooling và
    # normalize_embeddings=False. Teacher .npy hiện tại được tạo theo
    # đường này, nên student cũng trả raw pooled embedding.
    return mean_pool(outputs.last_hidden_state, batch['attention_mask']).float()


def distill_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor):
    # MSE thuần, không L2 normalize trước loss.
    return F.mse_loss(
        student_emb.float(),
        teacher_emb.float(),
        reduction='mean',
    )


def shuffled_stream(path: Path, max_rows: int | None, seed: int):
    """
    Streaming shuffle xác định theo seed.
    Trả (row_idx, document, label, source_file, token_count).
    """
    rng = random.Random(seed)
    buffer = []
    seen = 0

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for row_idx, line in enumerate(f):
            if max_rows is not None and seen >= max_rows:
                break

            row = json.loads(line)
            token_count = min(int(row.get('token_count', MAX_LENGTH)), MAX_LENGTH)
            item = (
                row_idx,
                row['document'],
                int(row.get('label', -1)),
                row.get('source_file'),
                token_count,
            )
            seen += 1

            if len(buffer) < SHUFFLE_BUFFER:
                buffer.append(item)
                continue

            j = rng.randrange(len(buffer))
            yield buffer[j]
            buffer[j] = item

    rng.shuffle(buffer)
    yield from buffer


def _yield_length_bucket(buffer: list, rng: random.Random):
    """
    Sort theo token_count, chia batch BATCH_SIZE rồi shuffle thứ tự batch.
    Như vậy sequence trong cùng batch có chiều dài gần nhau nhưng thứ tự batch
    vẫn ngẫu nhiên.
    """
    if not buffer:
        return

    buffer.sort(key=lambda item: item[4])

    full_count = (len(buffer) // BATCH_SIZE) * BATCH_SIZE
    batches = [
        buffer[i:i + BATCH_SIZE]
        for i in range(0, full_count, BATCH_SIZE)
    ]
    rng.shuffle(batches)

    for batch in batches:
        rng.shuffle(batch)
        yield from batch

    if full_count < len(buffer):
        remainder = buffer[full_count:]
        rng.shuffle(remainder)
        yield from remainder


def length_bucketed_stream(
    path: Path,
    max_rows: int | None,
    seed: int,
    source_skip_items: int = 0,
):
    """
    Bỏ prefix cũ nếu đang migrate checkpoint, sau đó gom buffer 8192 sample,
    sort theo token_count, chia batch 64, và shuffle thứ tự batch.
    """
    source = shuffled_stream(path, max_rows, seed)

    skipped_source = 0
    while skipped_source < source_skip_items:
        try:
            next(source)
        except StopIteration as exc:
            raise ValueError(
                f'Không skip đủ legacy prefix: {skipped_source} != {source_skip_items}'
            ) from exc
        skipped_source += 1

    bucket_rng = random.Random(seed + 1_000_003 + source_skip_items)
    bucket = []

    for item in source:
        bucket.append(item)
        if len(bucket) >= LENGTH_BUCKET_BUFFER:
            yield from _yield_length_bucket(bucket, bucket_rng)
            bucket = []

    if bucket:
        yield from _yield_length_bucket(bucket, bucket_rng)


def batch_stream(
    path: Path,
    max_rows: int | None,
    seed: int,
    skip_items: int = 0,
    legacy_prefix_rows: int = 0,
):
    """
    Tạo batch length-bucketed.

    legacy_prefix_rows là số sample đã train theo data order cũ.
    skip_items là tổng số sample đã train của epoch.
    """
    if legacy_prefix_rows < 0 or legacy_prefix_rows > skip_items:
        raise ValueError(
            f'legacy_prefix_rows không hợp lệ: {legacy_prefix_rows}, skip_items={skip_items}'
        )

    bucket_skip_items = skip_items - legacy_prefix_rows
    ids, texts, labels, sources = [], [], [], []
    skipped = 0

    stream = length_bucketed_stream(
        path,
        max_rows,
        seed,
        source_skip_items=legacy_prefix_rows,
    )

    for row_idx, text, label, source, token_count in stream:
        if skipped < bucket_skip_items:
            skipped += 1
            continue

        ids.append(row_idx)
        texts.append(text)
        labels.append(label)
        sources.append(source)

        if len(ids) == BATCH_SIZE:
            yield ids, texts, labels, sources
            ids, texts, labels, sources = [], [], [], []

    if skipped != bucket_skip_items:
        raise ValueError(
            f'Không skip đủ bucketed resume rows: {skipped} != {bucket_skip_items}'
        )

    if ids:
        yield ids, texts, labels, sources


def save_model_atomic(model, tokenizer, keep_layers, meta):
    if LOCAL_OUT_DIR.exists():
        shutil.rmtree(LOCAL_OUT_DIR)
    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(LOCAL_OUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(LOCAL_OUT_DIR)

    with open(LOCAL_OUT_DIR / 'distill_config.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'teacher_model': TEACHER_MODEL,
                'student_layers': NUM_STUDENT_LAYERS,
                'keep_teacher_layers': keep_layers,
                'max_length': MAX_LENGTH,
                'loss': 'mse_raw_teacher_embedding',
                'l2_normalize_before_loss': False,
                **meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp = DRIVE_OUT_DIR.with_name(DRIVE_OUT_DIR.name + '_tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(LOCAL_OUT_DIR, tmp)
    if DRIVE_OUT_DIR.exists():
        shutil.rmtree(DRIVE_OUT_DIR)
    tmp.replace(DRIVE_OUT_DIR)
    print(f'Saved distilled model: {DRIVE_OUT_DIR}')


def save_checkpoint_atomic(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    keep_layers,
    epoch: int,
    processed_rows: int,
    global_optim_step: int,
    train_rows: int,
    metric_sums: dict,
    metric_batch_count: int,
    legacy_prefix_rows: int,
):
    """Lưu checkpoint đầy đủ lên Drive. Chỉ gọi sau optimizer.step()."""
    if LOCAL_CHECKPOINT_DIR.exists():
        shutil.rmtree(LOCAL_CHECKPOINT_DIR)
    LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(LOCAL_CHECKPOINT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(LOCAL_CHECKPOINT_DIR)

    state = {
        'version': 2,
        'data_order_version': DATA_ORDER_VERSION,
        'length_bucket_buffer': int(LENGTH_BUCKET_BUFFER),
        'legacy_prefix_rows': int(legacy_prefix_rows),
        'teacher_model': TEACHER_MODEL,
        'num_student_layers': NUM_STUDENT_LAYERS,
        'keep_layers': list(keep_layers),
        'epoch': int(epoch),                 # 0-based epoch đang chạy
        'processed_rows': int(processed_rows),
        'global_optim_step': int(global_optim_step),
        'train_rows': int(train_rows),
        'epochs': int(EPOCHS),
        'batch_size': int(BATCH_SIZE),
        'grad_accum_steps': int(GRAD_ACCUM_STEPS),
        'shuffle_buffer': int(SHUFFLE_BUFFER),
        'seed': int(SEED),
        'metric_sums': {k: float(v) for k, v in metric_sums.items()},
        'metric_batch_count': int(metric_batch_count),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, LOCAL_CHECKPOINT_DIR / 'training_state.pt')

    with open(LOCAL_CHECKPOINT_DIR / 'checkpoint_meta.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'epoch': epoch + 1,
                'processed_rows': processed_rows,
                'train_rows': train_rows,
                'progress_percent': 100.0 * processed_rows / train_rows,
                'global_optimizer_step': global_optim_step,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Copy vào thư mục tạm trên Drive trước. Checkpoint cũ chỉ bị thay khi copy xong.
    if DRIVE_CHECKPOINT_TMP_DIR.exists():
        shutil.rmtree(DRIVE_CHECKPOINT_TMP_DIR)
    shutil.copytree(LOCAL_CHECKPOINT_DIR, DRIVE_CHECKPOINT_TMP_DIR)

    if DRIVE_CHECKPOINT_DIR.exists():
        shutil.rmtree(DRIVE_CHECKPOINT_DIR)
    DRIVE_CHECKPOINT_TMP_DIR.replace(DRIVE_CHECKPOINT_DIR)

    print(
        f'Checkpoint saved: epoch {epoch + 1}/{EPOCHS}, '
        f'{processed_rows:,}/{train_rows:,} '
        f'({100.0 * processed_rows / train_rows:.2f}%) -> {DRIVE_CHECKPOINT_DIR}'
    )


def prepare_checkpoint_for_resume():
    """Trả local checkpoint dir nếu có checkpoint hợp lệ để load."""
    if RESET_CHECKPOINT:
        if DRIVE_CHECKPOINT_DIR.exists():
            shutil.rmtree(DRIVE_CHECKPOINT_DIR)
        if DRIVE_CHECKPOINT_TMP_DIR.exists():
            shutil.rmtree(DRIVE_CHECKPOINT_TMP_DIR)
        if LOCAL_CHECKPOINT_DIR.exists():
            shutil.rmtree(LOCAL_CHECKPOINT_DIR)
        print('RESET_CHECKPOINT=True: đã xoá checkpoint cũ.')
        return None

    if DRIVE_CHECKPOINT_DIR.exists():
        if LOCAL_CHECKPOINT_DIR.exists():
            shutil.rmtree(LOCAL_CHECKPOINT_DIR)
        print(f'Copy checkpoint từ Drive xuống local: {DRIVE_CHECKPOINT_DIR}')
        shutil.copytree(DRIVE_CHECKPOINT_DIR, LOCAL_CHECKPOINT_DIR)
        return LOCAL_CHECKPOINT_DIR

    if LOCAL_CHECKPOINT_DIR.exists() and (LOCAL_CHECKPOINT_DIR / 'training_state.pt').exists():
        print(f'Dùng checkpoint local: {LOCAL_CHECKPOINT_DIR}')
        return LOCAL_CHECKPOINT_DIR

    return None


def validate_checkpoint_state(state: dict, train_rows: int) -> None:
    expected = {
        'teacher_model': TEACHER_MODEL,
        'num_student_layers': NUM_STUDENT_LAYERS,
        'train_rows': train_rows,
        'epochs': EPOCHS,
        'shuffle_buffer': SHUFFLE_BUFFER,
        'seed': SEED,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f'Checkpoint không tương thích: {key}={state.get(key)!r}, '
                f'expected {value!r}. Đặt RESET_CHECKPOINT=True nếu muốn train lại.'
            )

    checkpoint_order_version = int(state.get('data_order_version', 1))
    if checkpoint_order_version > DATA_ORDER_VERSION:
        raise ValueError(
            f'Checkpoint data_order_version={checkpoint_order_version} mới hơn code hiện tại '
            f'({DATA_ORDER_VERSION}).'
        )
    if checkpoint_order_version == DATA_ORDER_VERSION:
        checkpoint_bucket = int(state.get('length_bucket_buffer', 0))
        if checkpoint_bucket != LENGTH_BUCKET_BUFFER:
            raise ValueError(
                f'Checkpoint length_bucket_buffer={checkpoint_bucket}, '
                f'expected {LENGTH_BUCKET_BUFFER}. Đặt RESET_CHECKPOINT=True nếu muốn train lại.'
            )

    # Cho phép đổi physical batch / grad accumulation khi effective batch không đổi.
    # Ví dụ checkpoint cũ 16x4 có thể resume bằng cấu hình mới 64x1.
    checkpoint_batch_size = int(state.get('batch_size', 0))
    checkpoint_grad_accum = int(state.get('grad_accum_steps', 0))
    checkpoint_effective_batch = checkpoint_batch_size * checkpoint_grad_accum
    current_effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS

    if checkpoint_effective_batch != current_effective_batch:
        raise ValueError(
            'Checkpoint không tương thích về effective batch: '
            f'{checkpoint_batch_size}x{checkpoint_grad_accum}='
            f'{checkpoint_effective_batch}, expected '
            f'{BATCH_SIZE}x{GRAD_ACCUM_STEPS}={current_effective_batch}. '
            'Đặt RESET_CHECKPOINT=True nếu muốn train lại.'
        )

    epoch = int(state['epoch'])
    processed_rows = int(state['processed_rows'])
    if not 0 <= epoch < EPOCHS:
        raise ValueError(f'Checkpoint epoch không hợp lệ: {epoch}')
    if not 0 <= processed_rows <= train_rows:
        raise ValueError(f'Checkpoint processed_rows không hợp lệ: {processed_rows}')


def restore_rng_state(state: dict) -> None:
    random.setstate(state['python_random_state'])
    np.random.set_state(state['numpy_random_state'])
    torch.set_rng_state(state['torch_rng_state'])
    if torch.cuda.is_available() and state.get('cuda_rng_state_all') is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng_state_all'])


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return '--:--'
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def print_progress_line(
    epoch: int,
    processed_rows: int,
    train_rows: int,
    metric_sums: dict,
    metric_batch_count: int,
    lr: float,
    started_at: float,
    start_rows: int,
    last_len: int,
    final: bool = False,
):
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    rows_this_run = max(processed_rows - start_rows, 0)
    rate = rows_this_run / elapsed
    remaining = max(train_rows - processed_rows, 0)
    eta = remaining / rate if rate > 0 else math.inf
    n = max(metric_batch_count, 1)

    line = (
        f'Epoch {epoch + 1}/{EPOCHS} | '
        f'{processed_rows:,}/{train_rows:,} '
        f'({100.0 * processed_rows / train_rows:6.2f}%) | '
        f'mse={metric_sums["mse"] / n:.7f} | '
        f'lr={lr:.2e} | '
        f'{rate:,.1f} chunk/s | ETA {format_duration(eta)}'
    )
    padded = line.ljust(last_len)
    print('\r' + padded, end='\n' if final else '', flush=True)
    return len(line)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    copy_if_needed(DRIVE_CHUNKS, LOCAL_CHUNKS)
    copy_if_needed(DRIVE_TEACHER_EMB, LOCAL_TEACHER_EMB)

    teacher_embeddings = np.load(LOCAL_TEACHER_EMB, mmap_mode='r')
    if teacher_embeddings.ndim != 2:
        raise ValueError(f'Bad teacher embedding shape: {teacher_embeddings.shape}')
    total_teacher_rows, teacher_dim = teacher_embeddings.shape
    print(f'Teacher embeddings: {teacher_embeddings.shape} {teacher_embeddings.dtype}')

    train_rows = total_teacher_rows if MAX_TRAIN_ROWS is None else min(MAX_TRAIN_ROWS, total_teacher_rows)

    checkpoint_dir = prepare_checkpoint_for_resume()
    checkpoint_state = None

    if checkpoint_dir is not None:
        state_path = checkpoint_dir / 'training_state.pt'
        if not state_path.exists():
            raise FileNotFoundError(f'Checkpoint thiếu training_state.pt: {state_path}')
        checkpoint_state = torch.load(state_path, map_location='cpu', weights_only=False)
        validate_checkpoint_state(checkpoint_state, train_rows)

        print(f'Resume model từ checkpoint: {checkpoint_dir}')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint_dir, trust_remote_code=True)
        keep_layers = list(checkpoint_state['keep_layers'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
        model = AutoModel.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
        keep_layers = prune_to_student(model, NUM_STUDENT_LAYERS)

    hidden_size = int(getattr(model.config, 'hidden_size'))
    if hidden_size != teacher_dim:
        raise ValueError(
            f'Student hidden_size={hidden_size} nhưng teacher embedding dim={teacher_dim}. '
            'Phương án 6-layer drop-in cần cùng dimension.'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    batches_per_epoch = math.ceil(train_rows / BATCH_SIZE)
    optim_steps_per_epoch = math.ceil(batches_per_epoch / GRAD_ACCUM_STEPS)
    total_optim_steps = optim_steps_per_epoch * EPOCHS
    warmup_steps = int(total_optim_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps,
    )

    use_cuda = device.type == 'cuda'
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda and not use_bf16)

    global_optim_step = 0
    start_epoch = 0
    resume_rows = 0
    resume_legacy_prefix_rows = 0
    resume_metric_sums = {
        'mse': 0.0,
    }
    resume_metric_batch_count = 0

    if checkpoint_state is not None:
        optimizer.load_state_dict(checkpoint_state['optimizer'])
        scheduler.load_state_dict(checkpoint_state['scheduler'])
        scaler.load_state_dict(checkpoint_state['scaler'])

        global_optim_step = int(checkpoint_state['global_optim_step'])
        start_epoch = int(checkpoint_state['epoch'])
        resume_rows = int(checkpoint_state['processed_rows'])

        checkpoint_order_version = int(checkpoint_state.get('data_order_version', 1))
        if checkpoint_order_version < DATA_ORDER_VERSION:
            resume_legacy_prefix_rows = resume_rows
            print(
                f'Migrate checkpoint data order v{checkpoint_order_version} -> '
                f'v{DATA_ORDER_VERSION}: giữ {resume_legacy_prefix_rows:,} sample prefix cũ, '
                'length bucketing áp dụng cho phần còn lại.'
            )
        else:
            resume_legacy_prefix_rows = int(
                checkpoint_state.get('legacy_prefix_rows', 0)
            )

        # Tương thích checkpoint cũ: chỉ giữ MSE, bỏ các metric phụ nếu có.
        checkpoint_metric_sums = checkpoint_state.get('metric_sums', {})
        resume_metric_sums = {
            'mse': float(checkpoint_metric_sums.get('mse', 0.0)),
        }
        resume_metric_batch_count = int(checkpoint_state['metric_batch_count'])

        # Nếu checkpoint đúng 100% epoch, epoch đó đã hoàn tất.
        if resume_rows >= train_rows:
            start_epoch += 1
            resume_rows = 0
            resume_legacy_prefix_rows = 0
            resume_metric_sums = {
                'mse': 0.0,
            }
            resume_metric_batch_count = 0

        restore_rng_state(checkpoint_state)

        print(
            f'Resume state: epoch={start_epoch + 1}/{EPOCHS}, '
            f'processed={resume_rows:,}/{train_rows:,}, '
            f'optimizer_step={global_optim_step:,}'
        )

    optimizer.zero_grad(set_to_none=True)

    final_meta = None

    for epoch in range(start_epoch, EPOCHS):
        epoch_resume_rows = resume_rows if epoch == start_epoch else 0
        epoch_legacy_prefix_rows = (
            resume_legacy_prefix_rows if epoch == start_epoch else 0
        )

        if epoch == start_epoch and checkpoint_state is not None and epoch_resume_rows > 0:
            metric_sums = dict(resume_metric_sums)
            metric_batch_count = resume_metric_batch_count
        else:
            metric_sums = {
                'mse': 0.0,
            }
            metric_batch_count = 0

        processed_rows = epoch_resume_rows
        batches_since_resume = 0
        started_at = time.perf_counter()
        last_progress_len = 0

        checkpoint_percents = list(
            range(CHECKPOINT_EVERY_PERCENT, 101, CHECKPOINT_EVERY_PERCENT)
        )
        checkpoint_rows = {
            p: min(train_rows, math.ceil(train_rows * p / 100.0))
            for p in checkpoint_percents
        }
        pending_percents = [
            p for p in checkpoint_percents if checkpoint_rows[p] > processed_rows
        ]
        checkpoint_cursor = 0

        print(
            f'\nLength bucketing: buffer={LENGTH_BUCKET_BUFFER:,}, '
            f'batch={BATCH_SIZE}, legacy_prefix={epoch_legacy_prefix_rows:,}'
        )
        print(
            f'Start epoch {epoch + 1}/{EPOCHS} từ '
            f'{processed_rows:,}/{train_rows:,} '
            f'({100.0 * processed_rows / train_rows:.2f}%)'
        )
        last_progress_len = print_progress_line(
            epoch,
            processed_rows,
            train_rows,
            metric_sums,
            metric_batch_count,
            scheduler.get_last_lr()[0],
            started_at,
            epoch_resume_rows,
            last_progress_len,
        )

        stream = batch_stream(
            LOCAL_CHUNKS,
            train_rows,
            seed=SEED + epoch,
            skip_items=epoch_resume_rows,
            legacy_prefix_rows=epoch_legacy_prefix_rows,
        )

        for batch_idx, (row_ids, texts, labels, sources) in enumerate(stream):
            if max(row_ids) >= total_teacher_rows:
                raise IndexError('Chunk row vượt quá teacher embedding rows')

            teacher_np = np.asarray(teacher_embeddings[row_ids], dtype=np.float32)
            teacher_batch = torch.from_numpy(teacher_np).to(device, non_blocking=True)

            if use_cuda:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    student_batch = encode_student(model, tokenizer, texts, device)
                    loss = distill_loss(student_batch, teacher_batch)
                    scaled_loss = loss / GRAD_ACCUM_STEPS
            else:
                student_batch = encode_student(model, tokenizer, texts, device)
                loss = distill_loss(student_batch, teacher_batch)
                scaled_loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(scaled_loss).backward()

            metric_batch_count += 1
            metric_sums['mse'] += float(loss.detach().cpu())

            batches_since_resume += 1
            processed_rows += len(texts)
            processed_rows = min(processed_rows, train_rows)

            is_accum_boundary = (batches_since_resume % GRAD_ACCUM_STEPS == 0)
            is_last = processed_rows >= train_rows
            did_optimizer_step = False

            if is_accum_boundary or is_last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_optim_step += 1
                did_optimizer_step = True

            if (
                batch_idx % PROGRESS_EVERY_BATCHES == 0
                or is_last
            ):
                last_progress_len = print_progress_line(
                    epoch,
                    processed_rows,
                    train_rows,
                    metric_sums,
                    metric_batch_count,
                    scheduler.get_last_lr()[0],
                    started_at,
                    epoch_resume_rows,
                    last_progress_len,
                )

            # Chỉ checkpoint sau optimizer.step để không mất gradient accumulation dở dang.
            if did_optimizer_step:
                while (
                    checkpoint_cursor < len(pending_percents)
                    and processed_rows >= checkpoint_rows[pending_percents[checkpoint_cursor]]
                ):
                    percent = pending_percents[checkpoint_cursor]

                    # Kết thúc dòng progress trước khi in log checkpoint.
                    print()
                    save_checkpoint_atomic(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        keep_layers=keep_layers,
                        epoch=epoch,
                        processed_rows=processed_rows,
                        global_optim_step=global_optim_step,
                        train_rows=train_rows,
                        metric_sums=metric_sums,
                        metric_batch_count=metric_batch_count,
                        legacy_prefix_rows=epoch_legacy_prefix_rows,
                    )
                    checkpoint_cursor += 1
                    last_progress_len = 0
                    last_progress_len = print_progress_line(
                        epoch,
                        processed_rows,
                        train_rows,
                        metric_sums,
                        metric_batch_count,
                        scheduler.get_last_lr()[0],
                        started_at,
                        epoch_resume_rows,
                        last_progress_len,
                    )

            if is_last:
                break

        print_progress_line(
            epoch,
            processed_rows,
            train_rows,
            metric_sums,
            metric_batch_count,
            scheduler.get_last_lr()[0],
            started_at,
            epoch_resume_rows,
            last_progress_len,
            final=True,
        )

        n = max(metric_batch_count, 1)
        final_meta = {
            'epoch': epoch + 1,
            'global_optimizer_step': global_optim_step,
            'train_rows': train_rows,
            'teacher_embedding_dim': teacher_dim,
            'mean_mse': metric_sums['mse'] / n,
            'l2_normalize_before_loss': False,
            'length_bucketing': True,
            'length_bucket_buffer': LENGTH_BUCKET_BUFFER,
        }

        # Epoch tiếp theo luôn bắt đầu từ 0.
        resume_rows = 0
        resume_legacy_prefix_rows = 0
        checkpoint_state = None

    # Nếu checkpoint cho biết toàn bộ EPOCHS đã hoàn tất nhưng final model chưa kịp save,
    # model đang load từ checkpoint 100%, nên vẫn save lại final model ở đây.
    if final_meta is None:
        final_meta = {
            'epoch': EPOCHS,
            'global_optimizer_step': global_optim_step,
            'train_rows': train_rows,
            'teacher_embedding_dim': teacher_dim,
            'mean_mse': None,
            'l2_normalize_before_loss': False,
            'length_bucketing': True,
            'length_bucket_buffer': LENGTH_BUCKET_BUFFER,
            'restored_from_completed_checkpoint': True,
        }

    save_model_atomic(model, tokenizer, keep_layers, final_meta)
    print('Done.')
    print(f'Final model: {DRIVE_OUT_DIR}')
    print(f'Checkpoint: {DRIVE_CHECKPOINT_DIR}')


if __name__ == '__main__':
    main()
