import os
import subprocess
import sys
from pathlib import Path

ABLATIONS = ("form", "input", "a", "iframe", "meta")
CODE_DIR = Path(__file__).resolve().parent

for ablation in ABLATIONS:
    env = os.environ.copy()
    env["ABLATION"] = ablation
    print(f"\n=== BUILD DB: {ablation} ===", flush=True)
    subprocess.run([sys.executable, str(CODE_DIR / "build_jina_vector_db.py")], check=True, env=env)
