Closer-only Top-4 ablation
==========================

Purpose
-------
Test whether the retrieval-derived class direction (which class is closer) is sufficient without sending retrieved training HTML to Qwen.

Controlled setup
----------------
- Same exact 1,000 balanced test pages as topk_1000_balanced.
- Same exact Top-4 selected query chunks and same rank order.
- Same model: Qwen/Qwen2.5-Coder-7B-Instruct-AWQ.
- CLOSER_TO is computed per query chunk from the existing benign/phishing retrieval similarities.
- The LLM receives only QUERY_HTML + CLOSER_TO.
- Retrieved BENIGN/PHISHING HTML is NOT included.
- Numeric similarity scores are NOT included.
- Importance score, source filename, and page URL are NOT included.

Files
-----
stage_06_infer_closer_only_k4.py
  GPU inference for the closer-only condition. Resume-safe by prediction shard.

stage_07_compare_five_way.py
  CPU comparison of:
    RAG_FULL_SCORE
    RAG_CLOSER_LABEL
    CLOSER_ONLY
    RAG_NO_SCORE
    NO_RAG
  Produces pairwise exact McNemar tests for all 10 pairs.

Outputs
-------
data/closer_only_k4/inference_query_chunks_with_relative_similarity/
  metrics.json
  inference_manifest.json
  prediction_shards/

data/summary_five_way/
  comparison_five_way.json
  comparison_five_way.csv
  mcnemar_pairwise.csv
  page_predictions_five_way.jsonl.gz

Colab
------
from google.colab import drive
drive.mount('/content/drive')
!pip install -q -U vllm transformers
!pip uninstall -y torchaudio
!python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_06_infer_closer_only_k4.py

After inference:
!python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_07_compare_five_way.py
