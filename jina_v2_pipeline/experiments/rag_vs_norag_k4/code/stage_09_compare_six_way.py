import csv
import gzip
import json
import math
from itertools import combinations
from pathlib import Path

TOPK_DATA_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/topk_1000_balanced/data')
EXP_DATA_DIR = Path('/content/drive/MyDrive/html_token_analysis/jina_v2_pipeline/experiments/rag_vs_norag_k4/data')

CONDITIONS = {
    'RAG_FULL_SCORE': {
        'dir': TOPK_DATA_DIR / 'results' / 'k_04' / 'inference_scores_free_reasoning',
        'description': 'query chunks + retrieved benign/phishing examples + exact numeric similarity scores',
    },
    'RAG_CLOSER_LABEL': {
        'dir': EXP_DATA_DIR / 'rag_closer_label_k4' / 'inference_examples_with_relative_similarity',
        'description': 'same query chunks/examples; numeric scores removed; only which class is closer is provided',
    },
    'CLOSER_ONLY': {
        'dir': EXP_DATA_DIR / 'closer_only_k4' / 'inference_query_chunks_with_relative_similarity',
        'description': 'same selected query chunks + relative closer-to class label only; no retrieved HTML and no numeric similarity scores',
    },
    'CLOSER_SIGNAL_ONLY': {
        'dir': EXP_DATA_DIR / 'closer_signal_only_k4' / 'inference_relative_similarity_only',
        'description': 'only ordered Top-K closer-to class labels; no query HTML, no retrieved HTML, and no numeric similarity scores',
    },
    'RAG_NO_SCORE': {
        'dir': EXP_DATA_DIR / 'rag_no_score_k4' / 'inference_examples_without_similarity',
        'description': 'same query chunks/examples; no similarity score or relative direction',
    },
    'NO_RAG': {
        'dir': EXP_DATA_DIR / 'no_rag_k4' / 'inference_query_chunks_only',
        'description': 'same selected query chunks only; no retrieved examples or similarity information',
    },
}
SUMMARY_DIR = EXP_DATA_DIR / 'summary_six_way'


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
        'num_pages': m['num_pages'],
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


def pairwise_correctness(name_a, rows_a, name_b, rows_b):
    a = {r['page_id']: r for r in rows_a}
    b = {r['page_id']: r for r in rows_b}
    if set(a) != set(b):
        raise RuntimeError(f'Page-id mismatch: {name_a} vs {name_b}')
    both_correct = a_correct_b_wrong = a_wrong_b_correct = both_wrong = 0
    label_disagreements = 0
    for page_id in a:
        ra, rb = a[page_id], b[page_id]
        if int(ra['true_label']) != int(rb['true_label']):
            raise RuntimeError(f'True-label mismatch for {page_id}: {name_a} vs {name_b}')
        ca = ra['predicted_label'] == ra['true_label']
        cb = rb['predicted_label'] == rb['true_label']
        if ca and cb:
            both_correct += 1
        elif ca and not cb:
            a_correct_b_wrong += 1
        elif not ca and cb:
            a_wrong_b_correct += 1
        else:
            both_wrong += 1
        if ra['predicted_label'] != rb['predicted_label']:
            label_disagreements += 1
    return {
        'condition_a': name_a,
        'condition_b': name_b,
        'both_correct': both_correct,
        'a_correct_b_wrong': a_correct_b_wrong,
        'a_wrong_b_correct': a_wrong_b_correct,
        'both_wrong': both_wrong,
        'prediction_label_disagreements': label_disagreements,
        'mcnemar_exact_two_sided_p': exact_mcnemar_p(a_correct_b_wrong, a_wrong_b_correct),
    }


def main():
    metrics = {}
    preds = {}
    for name, cfg in CONDITIONS.items():
        metrics_path = cfg['dir'] / 'metrics.json'
        pred_dir = cfg['dir'] / 'prediction_shards'
        if not metrics_path.exists():
            raise FileNotFoundError(f'Missing {name} metrics: {metrics_path}')
        if not pred_dir.exists():
            raise FileNotFoundError(f'Missing {name} predictions: {pred_dir}')
        metrics[name] = read_json(metrics_path)
        preds[name] = read_preds(pred_dir)

    sizes = {name: len(rows) for name, rows in preds.items()}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f'Prediction-count mismatch: {sizes}')

    page_sets = {name: set(r['page_id'] for r in rows) for name, rows in preds.items()}
    first = next(iter(page_sets.values()))
    for name, ids in page_sets.items():
        if ids != first:
            raise RuntimeError(f'Page-id set differs for {name}')

    flat = {name: flat_metrics(m) for name, m in metrics.items()}
    pairwise = []
    for a, b in combinations(CONDITIONS.keys(), 2):
        pairwise.append(pairwise_correctness(a, preds[a], b, preds[b]))

    full = flat['RAG_FULL_SCORE']
    deltas_vs_full = {}
    for name, fm in flat.items():
        if name == 'RAG_FULL_SCORE':
            continue
        deltas_vs_full[name] = {
            key: full[key] - fm[key]
            for key in [
                'accuracy', 'balanced_accuracy', 'precision_phishing',
                'recall_phishing', 'f1_phishing', 'specificity_benign'
            ]
        }
        if full['avg_prompt_tokens'] is not None and fm['avg_prompt_tokens'] is not None:
            deltas_vs_full[name]['avg_prompt_tokens'] = full['avg_prompt_tokens'] - fm['avg_prompt_tokens']

    summary = {
        'experiment': 'rag_vs_norag_k4_six_way',
        'design': {
            'same_test_pages': True,
            'same_selected_query_chunks': True,
            'top_k': 4,
            'conditions': {name: cfg['description'] for name, cfg in CONDITIONS.items()},
            'rag_full_score_result_reused': True,
        },
        'metrics': flat,
        'delta_rag_full_score_minus_condition': deltas_vs_full,
        'pairwise_mcnemar': pairwise,
    }

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with (SUMMARY_DIR / 'comparison_six_way.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fields = [
        'condition','num_pages','accuracy','balanced_accuracy','precision_phishing','recall_phishing',
        'f1_phishing','specificity_benign','tn','fp','fn','tp','avg_prompt_tokens',
        'avg_groups_used','context_truncated_pages','invalid_predictions'
    ]
    with (SUMMARY_DIR / 'comparison_six_way.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in CONDITIONS:
            w.writerow({'condition': name, **flat[name]})

    with (SUMMARY_DIR / 'mcnemar_pairwise.csv').open('w', newline='', encoding='utf-8') as f:
        fields2 = [
            'condition_a','condition_b','both_correct','a_correct_b_wrong','a_wrong_b_correct',
            'both_wrong','prediction_label_disagreements','mcnemar_exact_two_sided_p'
        ]
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(pairwise)

    with gzip.open(SUMMARY_DIR / 'page_predictions_six_way.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as f:
        by_condition = {name: {r['page_id']: r for r in rows} for name, rows in preds.items()}
        for page_id in sorted(first):
            truth = int(by_condition['RAG_FULL_SCORE'][page_id]['true_label'])
            row = {'page_id': page_id, 'true_label': truth}
            for name in CONDITIONS:
                r = by_condition[name][page_id]
                row[name] = {
                    'predicted_label': r['predicted_label'],
                    'correct': bool(r['predicted_label'] == truth),
                }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Summary: {SUMMARY_DIR}')


if __name__ == '__main__':
    main()
