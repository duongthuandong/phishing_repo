import os
import subprocess
import sys
from pathlib import Path

ABLATIONS = ("form", "input", "a", "iframe", "meta")
CODE_DIR = Path(__file__).resolve().parent

for ablation in ABLATIONS:
    env = os.environ.copy()
    env["ABLATION"] = ablation
    print(f"\n=== CHUNK: {ablation} / train ===", flush=True)
    subprocess.run([sys.executable, str(CODE_DIR / "chunk_html_jina.py"), "--split", "train"], check=True, env=env)
    print(f"\n=== CHUNK: {ablation} / test ===", flush=True)
    subprocess.run([sys.executable, str(CODE_DIR / "chunk_html_jina.py"), "--split", "test"], check=True, env=env)
