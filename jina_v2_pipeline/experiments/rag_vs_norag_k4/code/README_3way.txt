RAG vs No-RAG K=4 — three-way controlled experiment
=====================================================

All three conditions use the exact same 1,000 balanced test pages and the exact same query chunks selected by RAG Top-4.

Conditions
----------
1) RAG with similarity score (existing result; reused, no rerun)
   query chunk + nearest BENIGN example + nearest PHISHING example + both similarity scores

2) RAG without similarity score (new inference)
   exact same query chunk + exact same retrieved BENIGN/PHISHING examples, but similarity values are omitted from the prompt

3) No-RAG Top-4 (existing experiment branch)
   exact same selected query chunks only; retrieved examples and similarity scores are omitted

Code
----
stage_01_infer_no_rag_k4.py          Existing No-RAG inference
stage_02_infer_rag_no_score_k4.py    New RAG-no-score inference
stage_03_compare_three_way.py        Three-way metrics + pairwise exact McNemar tests
stage_02_compare_rag_vs_norag.py     Original two-way comparison retained unchanged

Outputs
-------
data/no_rag_k4/inference_query_chunks_only/
data/rag_no_score_k4/inference_examples_without_similarity/
data/summary_three_way/
  comparison_three_way.json
  comparison_three_way.csv
  page_predictions_three_way.jsonl.gz

Run (GPU)
---------
from google.colab import drive
drive.mount('/content/drive')
!pip install -q -U vllm transformers
!pip uninstall -y torchaudio

# Run this if No-RAG has not been completed yet:
!python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_01_infer_no_rag_k4.py

# New condition: RAG without similarity score
!python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_02_infer_rag_no_score_k4.py

Run comparison (CPU is enough)
------------------------------
!python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_03_compare_three_way.py
