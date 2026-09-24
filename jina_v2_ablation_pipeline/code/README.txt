JINA V2 ABLATION PIPELINE

Ablations: form, input, a, iframe, meta. No baseline.
Fixed data: 2,000 retrieval pages (1,000 benign + 1,000 phishing), 500 test pages (250 + 250).
All ablations use the same split. The query page URL is preserved from ml_features.csv and passed to Qwen.

Independent stages:
1. stage_01_split.py       CPU - creates fixed balanced split with filename,label,url
2. stage_02_chunk.py       CPU - runs all five ablations for train and test
3. stage_03_embed.py       GPU recommended - Jina embeddings for all five ablations
4. stage_04_build_db.py    CPU - builds train ChromaDB for all five ablations
5. stage_05_prepare_llm.py CPU - retrieves nearest BENIGN/PHISHING evidence for all five ablations
6. stage_06_infer.py       GPU - Qwen v1 inference for all five ablations

Qwen inference:
- Uses infer_qwen25_coder_rag.py (v1).
- Similarity scores are included in the prompt.
- Query page URL is included once at page level in the prompt.
- Retrieved train examples contain HTML + label + similarity, not their source filenames/paths.
- infer_qwen25_coder_rag.py is not used.

Each stage runs horizontally in this order:
form -> input -> a -> iframe -> meta

Run each stage separately so the Colab GPU can be enabled only for embedding/inference.
Example:
python /content/drive/MyDrive/html_token_analysis/jina_v2_ablation_pipeline/code/stage_03_embed.py

Inference prompt: Qwen v1, includes Jina similarity scores and the query page URL.

Inference: Qwen v1 with Jina similarity scores + query page URL in prompt. Qwen v2 removed.
