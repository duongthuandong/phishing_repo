import os
import subprocess
import sys
from pathlib import Path

ABLATIONS = ("form", "input", "a", "iframe", "meta")
CODE_DIR = Path(__file__).resolve().parent

for ablation in ABLATIONS:
    env = os.environ.copy()
    env["ABLATION"] = ablation
    print(f"\n=== QWEN INFERENCE: {ablation} ===", flush=True)
    subprocess.run([sys.executable, str(CODE_DIR / "infer_qwen25_coder_rag.py")], check=True, env=env)
