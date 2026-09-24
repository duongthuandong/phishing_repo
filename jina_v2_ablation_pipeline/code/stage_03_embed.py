import os
import subprocess
import sys
from pathlib import Path

ABLATIONS = ("form", "input", "a", "iframe", "meta")
CODE_DIR = Path(__file__).resolve().parent

for ablation in ABLATIONS:
    env = os.environ.copy()
    env["ABLATION"] = ablation
    print(f"\n=== EMBED: {ablation} / train ===", flush=True)
    subprocess.run([sys.executable, str(CODE_DIR / "embed_jina_chunks.py"), "--split", "train"], check=True, env=env)
    print(f"\n=== EMBED: {ablation} / test ===", flush=True)
    subprocess.run([sys.executable, str(CODE_DIR / "embed_jina_chunks.py"), "--split", "test"], check=True, env=env)
