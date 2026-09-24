TOP-K EXPERIMENT — jina_v2_pipeline
===================================

Purpose
-------
Compare page-level evidence Top-K values K = 1, 2, 4, 8, 16 while holding all other
variables fixed.

Fairness controls
-----------------
- Test set: exactly 1,000 pages sampled once from the EXISTING jina_v2_pipeline test split.
- Balance: 500 benign + 500 phishing.
- Seed: 42.
- The SAME 1,000 pages are reused for every K.
- Train/retrieval corpus: unchanged. Uses the COMPLETE existing ChromaDB built from the
  original jina_v2_pipeline train split; no train resampling and no DB rebuild.
- Retrieval per query chunk: unchanged, nearest-1 BENIGN + nearest-1 PHISHING.
- Importance formula: unchanged, max(s_p,s_b) * abs(s_p-s_b).
- Prompt/model/inference: same original jina_v2_pipeline style: similarity scores included,
  no source filename/path, no URL, Qwen/Qwen2.5-Coder-7B-Instruct-AWQ.
- Only the number of page-level evidence groups changes: K = 1,2,4,8,16.

Optimization
------------
Stage 2 runs Chroma retrieval only ONCE at max K=16 for the sampled pages. Inputs for
K=1/2/4/8 are deterministic prefixes of the same ranked evidence list, so retrieval noise
or repeated work cannot confound the comparison.
Stage 3 loads Qwen/vLLM only ONCE and evaluates all K values sequentially.

Folder layout
-------------
jina_v2_pipeline/experiments/topk_1000_balanced/
  code/
    stage_01_sample_test.py
    stage_02_prepare_topk.py
    stage_03_infer_topk.py
    README.txt
  data/
    sample/
    retrieval/
    inputs/
      k_01/ ... k_16/
    results/
      k_01/ ... k_16/
    summary/
      topk_metrics.csv
      topk_metrics.json

Stages
------
1. stage_01_sample_test.py       CPU
   Fixed random balanced sample: 500 benign + 500 phishing, seed=42.

2. stage_02_prepare_topk.py      CPU
   Reuses existing full-train ChromaDB and existing test chunks/embeddings.
   Retrieves once at max K=16, then materializes K=1/2/4/8/16 inputs.

3. stage_03_infer_topk.py        GPU
   Loads Qwen AWQ with vLLM once, runs all K values, preserves shard resume, and writes
   per-K metrics plus a consolidated summary CSV/JSON.

Interpretation note
-------------------
Here "Top-K" means the number of ranked evidence groups from a query page included in the
LLM input. Each evidence group still contains exactly one nearest BENIGN and one nearest
PHISHING training chunk. This experiment does NOT change n_results inside each class-specific
Chroma query.
