RAG vs NO-RAG K=4 — controlled ablation
=======================================

Purpose
-------
Measure the contribution of retrieved RAG evidence while holding query-page HTML constant.

Existing RAG result is reused; it is NOT rerun.
Source experiment:
/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/topk_1000_balanced/

Test set
--------
Exactly the same 1,000 balanced test pages used in topk_1000_balanced:
500 benign + 500 phishing, seed=42.

Controlled input
----------------
RAG K=4:
  exact selected query chunks + nearest BENIGN example + nearest PHISHING example + similarity scores.

NO-RAG K=4:
  the exact same selected query chunks, in the exact same rank order.
  No retrieved examples, no retrieval labels, no similarity scores, no importance scores.
  No page URL (matching the existing Top-K experiment).

Important methodological note
-----------------------------
The selected query chunks were originally chosen using the RAG retrieval-based importance score.
Therefore this experiment isolates the contribution of RETRIEVED EVIDENCE after chunk selection.
It does not represent a fully retrieval-free chunk-selection pipeline.

Files
-----
stage_01_infer_no_rag_k4.py
  GPU. Reads existing RAG K=4 input shards, strips all retrieved evidence from the prompt,
  and runs only Qwen inference. Supports shard resume.

stage_02_compare_rag_vs_norag.py
  CPU. Reuses the existing RAG K=4 metrics/predictions and compares them with No-RAG.
  Writes metric deltas and an exact paired McNemar test.

Outputs
-------
data/no_rag_k4/inference_query_chunks_only/
  metrics.json
  inference_manifest.json
  prediction_shards/

data/summary/
  comparison.json
  comparison.csv
  prediction_disagreements.jsonl.gz

Colab
------
Stage 1 (GPU):
  from google.colab import drive
  drive.mount('/content/drive')
  !pip install -q -U vllm transformers
  !pip uninstall -y torchaudio
  !python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_01_infer_no_rag_k4.py

Stage 2 (CPU or same GPU runtime):
  !python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_02_compare_rag_vs_norag.py
