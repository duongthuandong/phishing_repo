import csv
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT_DIR = Path("/content/drive/MyDrive/html_token_analysis")
DATA_DIR = ROOT_DIR / "jina_v2_pipeline" / "data" / "splits"
ZIP_PATH = ROOT_DIR / "html_clean.zip"
CSV_PATH = ROOT_DIR / "ml_features.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    sources = [row["source"] for row in csv.DictReader(f)]

with zipfile.ZipFile(ZIP_PATH) as z:
    files = [
        info.filename for info in z.infolist()
        if not info.is_dir()
        and "__MACOSX" not in Path(info.filename).parts
        and not Path(info.filename).name.startswith("._")
        and Path(info.filename).suffix.lower() in {".html", ".htm", ".txt"}
        and Path(info.filename).stem.isdigit()
    ]

labels = [
    1 if sources[int(Path(name).stem)] == "verified_online" else 0
    for name in files
]

train_files, test_files, train_labels, test_labels = train_test_split(
    files, labels, test_size=0.2, random_state=42, stratify=labels
)

for output, split_files, split_labels in [
    (TRAIN_CSV, train_files, train_labels),
    (TEST_CSV, test_files, test_labels),
]:
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(zip(split_files, split_labels))
