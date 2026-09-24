import csv
import gzip
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup, Comment, Doctype
from transformers import AutoTokenizer
from tqdm.auto import tqdm

DRIVE_ROOT_DIR = Path("/content/drive/MyDrive/html_token_analysis")
DRIVE_PIPELINE_DIR = DRIVE_ROOT_DIR / "jina_v2_pipeline"
DRIVE_ZIP_PATH = DRIVE_ROOT_DIR / "html_clean.zip"
DRIVE_TRAIN_CSV = DRIVE_PIPELINE_DIR / "data" / "splits" / "test.csv"
DRIVE_OUT_FILE = DRIVE_PIPELINE_DIR / "data" / "chunks" / "test_chunks.jsonl.gz"

LOCAL_DIR = Path("/content/jina_v2_chunking")
ZIP_PATH = LOCAL_DIR / "html_clean.zip"
TRAIN_CSV = LOCAL_DIR / "test.csv"
OUT_FILE = LOCAL_DIR / "test_chunks.jsonl.gz"

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
MAX_TOKENS = 512

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def copy_if_needed(src, dst):
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        print(f"Copy local: {src} -> {dst}")
        shutil.copy2(src, dst)


copy_if_needed(DRIVE_ZIP_PATH, ZIP_PATH)
copy_if_needed(DRIVE_TRAIN_CSV, TRAIN_CSV)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
_tag_factory = BeautifulSoup("", "html.parser")


def token_len(text):
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def render_leaf(name, attrs):
    return str(_tag_factory.new_tag(name, attrs=attrs))


def split_text(text):
    limit = MAX_TOKENS - tokenizer.num_special_tokens_to_add(pair=False)
    if limit <= 0:
        raise ValueError("Giới hạn token không đủ cho văn bản")
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + limit, len(ids))
        while end > start:
            part = tokenizer.decode(
                ids[start:end], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            count = token_len(part)
            if count <= MAX_TOKENS:
                break
            end -= 1
        if end == start:
            raise ValueError("Không thể chia văn bản trong giới hạn token")
        chunks.append((part, count))
        start = end
    return chunks


def split_long_attribute(name, key, value):
    value = " ".join(map(str, value)) if isinstance(value, list) else str(value)
    ids = tokenizer(value, add_special_tokens=False, truncation=False)["input_ids"]
    frame_tokens = token_len(render_leaf(name, {key: ""}))
    limit = MAX_TOKENS - frame_tokens
    if limit <= 0:
        raise ValueError(f"Không thể giữ node <{name}> với attribute {key} trong {MAX_TOKENS} token")

    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + limit, len(ids))
        while end > start:
            part = tokenizer.decode(
                ids[start:end], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            candidate = render_leaf(name, {key: part})
            count = token_len(candidate)
            if count <= MAX_TOKENS:
                break
            end -= 1
        if end == start:
            raise ValueError(f"Không thể giữ node <{name}> với attribute {key} trong {MAX_TOKENS} token")
        chunks.append((candidate, count))
        start = end
    return chunks


def split_leaf_element(node):
    chunks = []
    current = {}
    current_chunk = None
    for key, value in node.attrs.items():
        single = render_leaf(node.name, {key: value})
        single_count = token_len(single)
        if single_count > MAX_TOKENS:
            if current:
                chunks.append(current_chunk)
                current = {}
                current_chunk = None
            chunks.extend(split_long_attribute(node.name, key, value))
            continue

        if not current:
            current = {key: value}
            current_chunk = (single, single_count)
            continue

        candidate_attrs = {**current, key: value}
        candidate = render_leaf(node.name, candidate_attrs)
        count = token_len(candidate)
        if count > MAX_TOKENS:
            chunks.append(current_chunk)
            current = {key: value}
            current_chunk = (single, single_count)
        else:
            current = candidate_attrs
            current_chunk = (candidate, count)

    if current:
        chunks.append(current_chunk)
    return chunks


def preprocess(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "svg"]):
        tag.decompose()
    for item in soup.find_all(string=lambda text: isinstance(text, (Comment, Doctype))):
        item.extract()
    return soup


def merge_chunks_greedy(chunks):
    merged = []
    current_text = ""
    current_count = 0

    for text, count in chunks:
        if not current_text:
            current_text = text
            current_count = count
            continue

        candidate = current_text + text
        candidate_count = token_len(candidate)
        if candidate_count <= MAX_TOKENS:
            current_text = candidate
            current_count = candidate_count
        else:
            merged.append((current_text, current_count))
            current_text = text
            current_count = count

    if current_text:
        merged.append((current_text, current_count))
    return merged


def merge_direct_children(child_results):
    chunks = []
    group = []

    for child_chunks, child_over_limit in child_results:
        if child_over_limit:
            if group:
                chunks.extend(merge_chunks_greedy(group))
                group = []
            chunks.extend(child_chunks)
        else:
            group.extend(child_chunks)

    if group:
        chunks.extend(merge_chunks_greedy(group))
    return chunks


def split_dom(node):
    children = [
        c for c in getattr(node, "contents", [])
        if getattr(c, "name", None) is not None or str(c).strip()
    ]
    child_results = [split_dom(child) for child in children]
    over_limit = any(child_over_limit for _, child_over_limit in child_results)

    if over_limit:
        return merge_direct_children(child_results), True

    text = str(node).strip()
    if not text:
        return [], False

    count = token_len(text)
    if count <= MAX_TOKENS:
        return [(text, count)], False

    if children:
        chunks = [chunk for child_chunks, _ in child_results for chunk in child_chunks]
        return merge_chunks_greedy(chunks), True

    if getattr(node, "name", None) is None:
        return split_text(text), True
    return split_leaf_element(node), True


def resume_from_output(files):
    source = DRIVE_OUT_FILE
    positions = {info.filename: i for i, info in enumerate(files)}
    temporary = OUT_FILE.with_suffix(".tmp")
    last_name = None
    pending = []
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as out:
        if source.exists() and source.stat().st_size:
            with gzip.open(source, "rt", encoding="utf-8") as old:
                try:
                    for line in old:
                        if not line.endswith("\n"):
                            break
                        name = json.loads(line)["source_file"]
                        if name not in positions:
                            raise ValueError(f"Output chứa tệp không có trong đầu vào: {name}")
                        if name != last_name:
                            if last_name is not None and positions[name] <= positions[last_name]:
                                raise ValueError("Thứ tự tệp đầu vào không khớp output cũ")
                            out.writelines(pending)
                            pending = []
                            last_name = name
                        pending.append(line)
                except EOFError:
                    pass
    temporary.replace(OUT_FILE)
    return positions[last_name] if last_name is not None else 0


def save_output():
    shutil.copy2(OUT_FILE, DRIVE_OUT_FILE)
    print(f"Saved: {DRIVE_OUT_FILE}")


with open(TRAIN_CSV, newline="", encoding="utf-8-sig") as f:
    labels = {row["filename"]: int(row["label"]) for row in csv.DictReader(f)}

with zipfile.ZipFile(ZIP_PATH) as zf:
    files = [
        info for info in zf.infolist()
        if info.filename in labels
        and not info.is_dir()
        and "__MACOSX" not in Path(info.filename).parts
        and not Path(info.filename).name.startswith("._")
    ]
    start = resume_from_output(files)
    try:
        for index in tqdm(range(start, len(files)), initial=start, total=len(files), desc="Chunk"):
            info = files[index]
            name = info.filename
            html = zf.read(info).decode("utf-8", errors="ignore")
            chunks, _ = split_dom(preprocess(html))
            file_id = hashlib.sha1(name.encode()).hexdigest()[:16]

            with gzip.open(OUT_FILE, "at", encoding="utf-8", compresslevel=1) as out:
                for i, (chunk, count) in enumerate(chunks):
                    out.write(json.dumps({
                        "id": f"{file_id}_{i}",
                        "source_file": name,
                        "chunk_index": i,
                        "token_count": count,
                        "label": labels[name],
                        "document": chunk
                    }, ensure_ascii=False) + "\n")

            if (index - start + 1) % 2000 == 0:
                save_output()
    finally:
        save_output()

print(f"Done: {DRIVE_OUT_FILE}")
