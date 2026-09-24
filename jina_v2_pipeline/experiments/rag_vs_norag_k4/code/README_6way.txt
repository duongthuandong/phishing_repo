RAG vs NO-RAG K=4 — six-way ablation
=======================================

New condition: CLOSER_SIGNAL_ONLY
---------------------------------
The same 1,000 balanced test pages and the same retrieval-selected Top-K=4 chunk identities are reused.

For each selected chunk, the pipeline still computes the nearest BENIGN and nearest PHISHING training-chunk similarities internally. It then converts them to one categorical signal:
  PHISHING if phishing_similarity > benign_similarity
  BENIGN   if benign_similarity > phishing_similarity
  TIE      otherwise

The LLM receives ONLY the ordered Top-K categorical signals, for example:
  <TOPK_CHUNK rank="1" closer_to="PHISHING" />
  <TOPK_CHUNK rank="2" closer_to="BENIGN" />
  <TOPK_CHUNK rank="3" closer_to="PHISHING" />
  <TOPK_CHUNK rank="4" closer_to="PHISHING" />

The LLM does NOT receive:
- query/test HTML chunks
- retrieved BENIGN HTML
- retrieved PHISHING HTML
- numeric similarity scores
- importance scores
- page URL or source file

This isolates whether the four relative class-retrieval signals alone are sufficient for page classification.

Files
-----
stage_08_infer_closer_signal_only_k4.py
  GPU inference for the signal-only condition. Resume-safe prediction shards.

stage_09_compare_six_way.py
  CPU comparison across:
  RAG_FULL_SCORE
  RAG_CLOSER_LABEL
  CLOSER_ONLY
  CLOSER_SIGNAL_ONLY
  RAG_NO_SCORE
  NO_RAG
  Includes all 15 pairwise exact two-sided McNemar tests.

Colab
------
!python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_08_infer_closer_signal_only_k4.py

Then:
!python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_09_compare_six_way.py

Outputs
-------
data/closer_signal_only_k4/inference_relative_similarity_only/
  metrics.json
  inference_manifest.json
  prediction_shards/

data/summary_six_way/
  comparison_six_way.json
  comparison_six_way.csv
  mcnemar_pairwise.csv
  page_predictions_six_way.jsonl.gz
