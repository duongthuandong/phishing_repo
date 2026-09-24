import csv
import gzip
import json
import math
from pathlib import Path

TOPK_DATA_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/topk_1000_balanced/data')
RAG_SCORE_DIR = TOPK_DATA_DIR / 'results' / 'k_04' / 'inference_scores_free_reasoning'
RAG_SCORE_METRICS = RAG_SCORE_DIR / 'metrics.json'
RAG_SCORE_PRED_DIR = RAG_SCORE_DIR / 'prediction_shards'

EXP_DATA_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/data')
RAG_NO_SCORE_DIR = EXP_DATA_DIR / 'rag_no_score_k4' / 'inference_examples_without_similarity'
RAG_NO_SCORE_METRICS = RAG_NO_SCORE_DIR / 'metrics.json'
RAG_NO_SCORE_PRED_DIR = RAG_NO_SCORE_DIR / 'prediction_shards'

NO_RAG_DIR = EXP_DATA_DIR / 'no_rag_k4' / 'inference_query_chunks_only'
NO_RAG_METRICS = NO_RAG_DIR / 'metrics.json'
NO_RAG_PRED_DIR = NO_RAG_DIR / 'prediction_shards'

SUMMARY_DIR = EXP_DATA_DIR / 'summary_three_way'


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


def metric_delta(a, b):
    out = {}
    for key in ['accuracy','balanced_accuracy','precision_phishing','recall_phishing','f1_phishing','specificity_benign']:
        out[key] = a[key] - b[key]
    if a.get('avg_prompt_tokens') is not None and b.get('avg_prompt_tokens') is not None:
        out['avg_prompt_tokens'] = a['avg_prompt_tokens'] - b['avg_prompt_tokens']
    return out


def paired_stats(name_a, preds_a, name_b, preds_b):
    a_by = {r['page_id']: r for r in preds_a}
    b_by = {r['page_id']: r for r in preds_b}
    if set(a_by) != set(b_by):
        raise RuntimeError(f'{name_a} và {name_b} không dùng cùng page_id.')
    a_correct_b_wrong = b_correct_a_wrong = both_correct = both_wrong = 0
    label_disagreements = 0
    for page_id in a_by:
        a, b = a_by[page_id], b_by[page_id]
        if int(a['true_label']) != int(b['true_label']):
            raise RuntimeError(f'True label mismatch: {page_id}')
        ac = a['predicted_label'] == a['true_label']
        bc = b['predicted_label'] == b['true_label']
        if ac and bc:
            both_correct += 1
        elif ac and not bc:
            a_correct_b_wrong += 1
        elif not ac and bc:
            b_correct_a_wrong += 1
        else:
            both_wrong += 1
        if a['predicted_label'] != b['predicted_label']:
            label_disagreements += 1
    return {
        'both_correct': both_correct,
        f'{name_a}_correct_{name_b}_wrong': a_correct_b_wrong,
        f'{name_a}_wrong_{name_b}_correct': b_correct_a_wrong,
        'both_wrong': both_wrong,
        'prediction_label_disagreements': label_disagreements,
        'mcnemar_exact_two_sided_p': exact_mcnemar_p(a_correct_b_wrong, b_correct_a_wrong),
    }


def main():
    required = [RAG_SCORE_METRICS, RAG_NO_SCORE_METRICS, NO_RAG_METRICS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Thiếu kết quả để compare 3-way:\n' + '\n'.join(missing))

    metrics = {
        'rag_with_score': read_json(RAG_SCORE_METRICS),
        'rag_without_score': read_json(RAG_NO_SCORE_METRICS),
        'no_rag': read_json(NO_RAG_METRICS),
    }
    preds = {
        'rag_with_score': read_preds(RAG_SCORE_PRED_DIR),
        'rag_without_score': read_preds(RAG_NO_SCORE_PRED_DIR),
        'no_rag': read_preds(NO_RAG_PRED_DIR),
    }

    lengths = {k: len(v) for k, v in preds.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f'Prediction count mismatch: {lengths}')

    id_sets = {k: {r['page_id'] for r in v} for k, v in preds.items()}
    first = next(iter(id_sets.values()))
    if any(s != first for s in id_sets.values()):
        raise RuntimeError('Ba condition không dùng đúng cùng tập page_id.')

    flat = {k: flat_metrics(v) for k, v in metrics.items()}
    pairwise = {
        'rag_with_score_vs_rag_without_score': {
            'delta_first_minus_second': metric_delta(flat['rag_with_score'], flat['rag_without_score']),
            'paired_correctness': paired_stats('rag_with_score', preds['rag_with_score'], 'rag_without_score', preds['rag_without_score']),
        },
        'rag_with_score_vs_no_rag': {
            'delta_first_minus_second': metric_delta(flat['rag_with_score'], flat['no_rag']),
            'paired_correctness': paired_stats('rag_with_score', preds['rag_with_score'], 'no_rag', preds['no_rag']),
        },
        'rag_without_score_vs_no_rag': {
            'delta_first_minus_second': metric_delta(flat['rag_without_score'], flat['no_rag']),
            'paired_correctness': paired_stats('rag_without_score', preds['rag_without_score'], 'no_rag', preds['no_rag']),
        },
    }

    by_id = {name: {r['page_id']: r for r in rows} for name, rows in preds.items()}
    page_rows = []
    for page_id in sorted(first):
        rs = by_id['rag_with_score'][page_id]
        rn = by_id['rag_without_score'][page_id]
        nr = by_id['no_rag'][page_id]
        true_label = int(rs['true_label'])
        if int(rn['true_label']) != true_label or int(nr['true_label']) != true_label:
            raise RuntimeError(f'True label mismatch: {page_id}')
        page_rows.append({
            'page_id': page_id,
            'true_label': true_label,
            'rag_with_score_prediction': rs['predicted_label'],
            'rag_without_score_prediction': rn['predicted_label'],
            'no_rag_prediction': nr['predicted_label'],
            'rag_with_score_correct': rs['predicted_label'] == true_label,
            'rag_without_score_correct': rn['predicted_label'] == true_label,
            'no_rag_correct': nr['predicted_label'] == true_label,
        })

    summary = {
        'experiment': 'rag_vs_norag_k4_three_way',
        'design': {
            'same_1000_test_pages': True,
            'same_selected_query_chunks': True,
            'top_k': 4,
            'rag_with_score': 'query chunks + nearest BENIGN/PHISHING examples + similarity scores',
            'rag_without_score': 'same query chunks + same nearest BENIGN/PHISHING examples, similarity scores omitted',
            'no_rag': 'same query chunks only; retrieved examples and similarity scores omitted',
            'existing_rag_with_score_result_reused': True,
        },
        'conditions': flat,
        'pairwise': pairwise,
    }

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with (SUMMARY_DIR / 'comparison_three_way.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fields = ['condition','accuracy','balanced_accuracy','precision_phishing','recall_phishing','f1_phishing','specificity_benign','tn','fp','fn','tp','avg_prompt_tokens','avg_groups_used','context_truncated_pages','invalid_predictions']
    with (SUMMARY_DIR / 'comparison_three_way.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in ['rag_with_score','rag_without_score','no_rag']:
            w.writerow({'condition': name, **flat[name]})

    with gzip.open(SUMMARY_DIR / 'page_predictions_three_way.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as f:
        for row in page_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Summary: {SUMMARY_DIR}')


if __name__ == '__main__':
    main()
