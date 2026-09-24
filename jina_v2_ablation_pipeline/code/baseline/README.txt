BASELINE RUN (same 2,500 pages as ablations)
===============================================

Goal
----
Run the no-tag-removal baseline on exactly the same shared split used by the five
ablations: 2,000 retrieval/train pages (1,000 benign + 1,000 phishing) and 500 test
pages (250 benign + 250 phishing).

Output root
-----------
/content/drive/MyDrive/html_token_analysis/jina_v2_ablation_pipeline/data/ablations/baseline/

Stages
------
1) stage_01_materialize.py   CPU
   Reuses baseline chunks + Jina v2 embeddings already stored in jina_v2_pipeline.
   It scans BOTH the old train and old test partitions because the new 2,500-page
   split is independent of the old 80/20 split. No Jina model / GPU is used.
   It injects the query-page URL from the shared ablation split into baseline chunks.

2) stage_02_build_db.py      CPU
   Builds baseline ChromaDB using ONLY the 2,000 shared train pages.

3) stage_03_prepare_llm.py   CPU
   Runs the exact same RAG logic as the ablation pipeline: nearest benign + nearest
   phishing per test chunk, similarity scores, page-level top-8 evidence.

4) stage_04_infer.py         GPU
   Runs Qwen/Qwen2.5-Coder-7B-Instruct-AWQ with the same v1 prompt:
   query URL + retrieved examples + similarity scores.

Colab commands
---------------
CPU runtime:
  from google.colab import drive
  drive.mount('/content/drive')
  !pip install -q numpy chromadb
  !python /content/drive/MyDrive/html_token_analysis/jina_v2_ablation_pipeline/code/baseline/stage_01_materialize.py
  !python /content/drive/MyDrive/html_token_analysis/jina_v2_ablation_pipeline/code/baseline/stage_02_build_db.py
  !python /content/drive/MyDrive/html_token_analysis/jina_v2_ablation_pipeline/code/baseline/stage_03_prepare_llm.py

GPU runtime:
  from google.colab import drive
  drive.mount('/content/drive')
  !pip install -q -U vllm transformers
  !pip uninstall -y torchaudio   # only if Colab has the CUDA mismatch seen earlier
  !python /content/drive/MyDrive/html_token_analysis/jina_v2_ablation_pipeline/code/baseline/stage_04_infer.py

Notes
-----
- Stage 1 is idempotent and keeps per-source materialization parts/manifests for resume.
- Stage 2/3/4 preserve the checkpoint/resume behavior from the ablation scripts.
- No HTML is re-chunked and no Jina embedding is recomputed for baseline.
