import csv
import gzip
import json
import math
from pathlib import Path

TOPK_DATA_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/topk_1000_balanced/data')
RAG_DIR = TOPK_DATA_DIR / 'results' / 'k_04' / 'inference_scores_free_reasoning'
RAG_METRICS = RAG_DIR / 'metrics.json'
RAG_PRED_DIR = RAG_DIR / 'prediction_shards'

EXP_DATA_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/data')
NO_RAG_DIR = EXP_DATA_DIR / 'no_rag_k4' / 'inference_query_chunks_only'
NO_RAG_METRICS = NO_RAG_DIR / 'metrics.json'
NO_RAG_PRED_DIR = NO_RAG_DIR / 'prediction_shards'
SUMMARY_DIR = EXP_DATA_DIR / 'summary'


def read_json(path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def read_preds(folder):
    rows = []
    for p in sorted(folder.glob('part-*.jsonl.gz')):
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            rows.extend(json.loads(line) for line in f)
    return rows


def exact_mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def flat_metrics(m):
    v = m['valid_only']
    cm = v['confusion_matrix']
    return {
        'accuracy': m['accuracy_all_invalid_as_wrong'],
        'balanced_accuracy': m['balanced_accuracy_all_invalid_as_wrong'],
        'precision_phishing': v['precision_phishing'],
        'recall_phishing': v['recall_phishing'],
        'f1_phishing': v['f1_phishing'],
        'specificity_benign': v['specificity_benign'],
        'tn': cm['tn'], 'fp': cm['fp'], 'fn': cm['fn'], 'tp': cm['tp'],
        'avg_prompt_tokens': m.get('avg_prompt_tokens'),
        'avg_groups_used': m.get('avg_groups_used'),
        'context_truncated_pages': m.get('context_truncated_pages'),
        'invalid_predictions': m.get('invalid_predictions'),
    }


def main():
    if not RAG_METRICS.exists():
        raise FileNotFoundError(f'Missing existing RAG K=4 metrics: {RAG_METRICS}')
    if not NO_RAG_METRICS.exists():
        raise FileNotFoundError(f'Missing No-RAG metrics: {NO_RAG_METRICS}. Run stage_01 first.')

    rag_m = read_json(RAG_METRICS)
    no_m = read_json(NO_RAG_METRICS)
    rag = read_preds(RAG_PRED_DIR)
    no = read_preds(NO_RAG_PRED_DIR)
    if len(rag) != len(no):
        raise RuntimeError(f'Prediction count mismatch: RAG={len(rag)}, No-RAG={len(no)}')

    rag_by_id = {r['page_id']: r for r in rag}
    no_by_id = {r['page_id']: r for r in no}
    if set(rag_by_id) != set(no_by_id):
        raise RuntimeError('RAG và No-RAG không dùng cùng page_id.')

    b = c = both_correct = both_wrong = 0
    disagreements = []
    for page_id in sorted(rag_by_id):
        r, n = rag_by_id[page_id], no_by_id[page_id]
        if int(r['true_label']) != int(n['true_label']):
            raise RuntimeError(f'True label mismatch: {page_id}')
        rc = r['predicted_label'] == r['true_label']
        nc = n['predicted_label'] == n['true_label']
        if rc and nc:
            both_correct += 1
        elif rc and not nc:
            b += 1
        elif not rc and nc:
            c += 1
        else:
            both_wrong += 1
        if r['predicted_label'] != n['predicted_label']:
            disagreements.append({
                'page_id': page_id,
                'true_label': int(r['true_label']),
                'rag_prediction': r['predicted_label'],
                'no_rag_prediction': n['predicted_label'],
                'rag_correct': bool(rc),
                'no_rag_correct': bool(nc),
            })

    rf = flat_metrics(rag_m)
    nf = flat_metrics(no_m)
    delta = {}
    for key in ['accuracy','balanced_accuracy','precision_phishing','recall_phishing','f1_phishing','specificity_benign']:
        delta[key] = rf[key] - nf[key]
    if rf['avg_prompt_tokens'] is not None and nf['avg_prompt_tokens'] is not None:
        delta['avg_prompt_tokens'] = rf['avg_prompt_tokens'] - nf['avg_prompt_tokens']

    summary = {
        'experiment': 'rag_vs_norag_k4',
        'design': {
            'same_test_pages': True,
            'same_selected_query_chunks': True,
            'top_k': 4,
            'rag_condition': 'query chunks + nearest benign/phishing retrieved examples + similarity scores',
            'no_rag_condition': 'same query chunks only; retrieved examples/similarities/importance omitted',
            'rag_result_reused': True,
        },
        'rag_k4': rf,
        'no_rag_k4': nf,
        'delta_rag_minus_no_rag': delta,
        'paired_correctness': {
            'both_correct': both_correct,
            'rag_correct_no_rag_wrong': b,
            'rag_wrong_no_rag_correct': c,
            'both_wrong': both_wrong,
            'mcnemar_exact_two_sided_p': exact_mcnemar_p(b, c),
        },
        'num_prediction_label_disagreements': len(disagreements),
    }

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with (SUMMARY_DIR / 'comparison.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fields = ['condition','accuracy','balanced_accuracy','precision_phishing','recall_phishing','f1_phishing','specificity_benign','tn','fp','fn','tp','avg_prompt_tokens','avg_groups_used','context_truncated_pages','invalid_predictions']
    with (SUMMARY_DIR / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({'condition':'RAG_K4', **rf})
        w.writerow({'condition':'NO_RAG_K4', **nf})

    with gzip.open(SUMMARY_DIR / 'prediction_disagreements.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as f:
        for row in disagreements:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Summary: {SUMMARY_DIR}')


if __name__ == '__main__':
    main()
