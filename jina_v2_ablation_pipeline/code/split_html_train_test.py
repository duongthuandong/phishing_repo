import csv
import random
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT_DIR = Path("/content/drive/MyDrive/html_token_analysis")
PIPELINE_DIR = ROOT_DIR / "jina_v2_ablation_pipeline"
DATA_DIR = PIPELINE_DIR / "data" / "splits"
ZIP_PATH = ROOT_DIR / "html_clean.zip"
CSV_PATH = ROOT_DIR / "ml_features.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

TRAIN_PER_CLASS = 1000
TEST_PER_CLASS = 250
RANDOM_STATE = 42

DATA_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    metadata = list(csv.DictReader(f))
    sources = [row["source"] for row in metadata]
    urls = [row["url"] for row in metadata]

with zipfile.ZipFile(ZIP_PATH) as z:
    files = [
        info.filename for info in z.infolist()
        if not info.is_dir()
        and "__MACOSX" not in Path(info.filename).parts
        and not Path(info.filename).name.startswith("._")
        and Path(info.filename).suffix.lower() in {".html", ".htm", ".txt"}
        and Path(info.filename).stem.isdigit()
    ]

benign_files = [
    name for name in files
    if sources[int(Path(name).stem)] != "verified_online"
]
phishing_files = [
    name for name in files
    if sources[int(Path(name).stem)] == "verified_online"
]

need_per_class = TRAIN_PER_CLASS + TEST_PER_CLASS
if len(benign_files) < need_per_class or len(phishing_files) < need_per_class:
    raise ValueError(
        f"Không đủ dữ liệu để lấy {need_per_class} mẫu mỗi lớp: "
        f"benign={len(benign_files)}, phishing={len(phishing_files)}"
    )

benign_train, benign_test = train_test_split(
    benign_files,
    train_size=TRAIN_PER_CLASS,
    test_size=TEST_PER_CLASS,
    random_state=RANDOM_STATE,
    shuffle=True,
)
phishing_train, phishing_test = train_test_split(
    phishing_files,
    train_size=TRAIN_PER_CLASS,
    test_size=TEST_PER_CLASS,
    random_state=RANDOM_STATE,
    shuffle=True,
)

train_rows = [
    (name, 0, urls[int(Path(name).stem)]) for name in benign_train
] + [
    (name, 1, urls[int(Path(name).stem)]) for name in phishing_train
]
test_rows = [
    (name, 0, urls[int(Path(name).stem)]) for name in benign_test
] + [
    (name, 1, urls[int(Path(name).stem)]) for name in phishing_test
]

rng = random.Random(RANDOM_STATE)
rng.shuffle(train_rows)
rng.shuffle(test_rows)

for output, rows in [(TRAIN_CSV, train_rows), (TEST_CSV, test_rows)]:
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "url"])
        writer.writerows(rows)

print(f"Train: {len(train_rows)} = {TRAIN_PER_CLASS} benign + {TRAIN_PER_CLASS} phishing")
print(f"Test:  {len(test_rows)} = {TEST_PER_CLASS} benign + {TEST_PER_CLASS} phishing")
print(f"Saved: {TRAIN_CSV}")
print(f"Saved: {TEST_CSV}")
