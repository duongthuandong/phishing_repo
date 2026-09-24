import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(CODE_DIR / "split_html_train_test.py")], check=True)
