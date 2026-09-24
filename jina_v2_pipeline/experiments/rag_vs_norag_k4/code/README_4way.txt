RAG vs No-RAG K=4 — four-way controlled ablation
=================================================

New condition: RAG_CLOSER_LABEL
- Uses the exact same 1,000 balanced test pages as topk_1000_balanced.
- Uses the exact same selected Top-4 query chunks as the existing RAG K=4 condition.
- Uses the exact same nearest BENIGN and nearest PHISHING retrieved examples.
- Does NOT expose numeric similarity values.
- For each evidence group, computes which class-specific retrieved example is closer using the original similarities:
    PHISHING if s_phishing > s_benign
    BENIGN   if s_benign > s_phishing
    TIE      if equal
- Only that relative direction is exposed to the LLM as closer_to="...".
- Importance scores remain hidden.

Four conditions compared:
1. RAG_FULL_SCORE: query + retrieved examples + exact numeric similarities.
2. RAG_CLOSER_LABEL: query + same retrieved examples + closer-to class only, no numeric similarities.
3. RAG_NO_SCORE: query + same retrieved examples, no similarity information.
4. NO_RAG: same selected query chunks only.

Run inference (GPU):
  !python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_04_infer_rag_closer_label_k4.py

Then compare all four (CPU or same runtime):
  !python /content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/code/stage_05_compare_four_way.py

Outputs:
  data/rag_closer_label_k4/inference_examples_with_relative_similarity/
  data/summary_four_way/comparison_four_way.json
  data/summary_four_way/comparison_four_way.csv
  data/summary_four_way/mcnemar_pairwise.csv
  data/summary_four_way/page_predictions_four_way.jsonl.gz
