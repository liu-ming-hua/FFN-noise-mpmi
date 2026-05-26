#!/usr/bin/env python3
"""
multi-seed evaluation script with better result management.
Evaluates multiple seeds and provides organized output with timestamp and metadata.
"""

import pickle
import os
import sys
import json
import argparse
import re
from collections import Counter
from typing import Tuple
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
# import bayesian_transformer

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    from torch.nn.functional import softmax
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None
    softmax = None

# # Add src path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
# evaluation utils
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from run.utils.math import *
from run.utils.grader import *
from run.utils.qwen_math_parser import *





def load_nli_model(nli_model_path: str):
    if nli_model_path is None:
        return None, None
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise ImportError(
            "transformers and torch are required for NLI evaluation. Install them with `pip install transformers torch`."
        )
    tokenizer = AutoTokenizer.from_pretrained(nli_model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        nli_model_path, trust_remote_code=True
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return tokenizer, model


def _normalize_label(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def nli_predict(premise: str, hypothesis: str, tokenizer, model):
    if tokenizer is None or model is None:
        return None
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    if logits is None:
        return None
    probs = softmax(logits, dim=-1).cpu().numpy()[0]
    if hasattr(model.config, "id2label"):
        label_map = {int(k): _normalize_label(v) for k, v in model.config.id2label.items()}
    elif hasattr(model.config, "label2id"):
        label_map = {v: _normalize_label(k) for k, v in model.config.label2id.items()}
    else:
        label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    scores = {}
    for idx, prob in enumerate(probs):
        label = label_map.get(idx, f"label_{idx}")
        scores[label] = float(prob)
    return scores


def _best_nli_label(scores: dict) -> str:
    if not scores:
        return ""
    return max(scores.items(), key=lambda kv: kv[1])[0]


def nli_bidirectional_entailment(prediction: str, truth: str, question: str, tokenizer, model):
    """Check bidirectional entailment with question context for better matching.
    
    Args:
        prediction: predicted answer
        truth: ground truth answer
        question: question context to add before the answer
        tokenizer: NLI model tokenizer
        model: NLI model
    
    Returns:
        (is_equivalent, forward_entail_score, backward_entail_score)
    """
    if tokenizer is None or model is None:
        return False, 0.0, 0.0
    
    # Add question context to answers for better matching
    prediction_with_context = f"{question} {prediction}" if question else prediction
    truth_with_context = f"{question} {truth}" if question else truth
    
    forward = nli_predict(prediction_with_context, truth_with_context, tokenizer, model)
    backward = nli_predict(truth_with_context, prediction_with_context, tokenizer, model)
    if forward is None or backward is None:
        return False, 0.0, 0.0
    forward_label = _best_nli_label(forward)
    backward_label = _best_nli_label(backward)
    forward_entail = max(
        forward.get("entailment", 0.0),
        forward.get("entail", 0.0),
        forward.get("ENTAILMENT", 0.0),
        forward.get("Entailment", 0.0),
    )
    backward_entail = max(
        backward.get("entailment", 0.0),
        backward.get("entail", 0.0),
        backward.get("ENTAILMENT", 0.0),
        backward.get("Entailment", 0.0),
    )
    is_equivalent = forward_label == "entailment" and backward_label == "entailment"
    return is_equivalent, forward_entail, backward_entail


def flatten_answers(answers):
    flattened = []
    if isinstance(answers, str):
        return [answers]
    if answers is None:
        return []
    for answer in answers:
        if isinstance(answer, (list, tuple)):
            flattened.extend(flatten_answers(answer))
        elif answer is None:
            continue
        else:
            flattened.append(str(answer))
    return flattened


def is_reading_comprehension_dataset(dataset: str) -> bool:
    if dataset is None:
        return False
    dataset_lower = dataset.lower()
    return any(key in dataset_lower for key in ("squad", "nq", "natural_questions", "reading_comprehension", "rc"))

# extract answer from string
def extract(string):
    """Extract answer from string"""
    try:
        return strip_string(extract_answer(string, 'math'))
    except:
        return string

def extract_unc(output):
    """Extract uncertainty scores from output"""
    try:
        unc_list = output.uncertainties
        ll_list = output.logprobs
        
        # au_list = []
        # tu_list = []
        # eu_list = []
        mpmi_list = []
        ll_list_total = []
        
        for d in unc_list:
            v = list(d.values())[0]
            # au_list.append(getattr(v, 'aleatoric_uncertainty', np.nan))
            # tu_list.append(getattr(v, 'total_uncertainty', np.nan))
            # eu_list.append(getattr(v, 'epistemic_uncertainty', np.nan))
            mpmi_list.append(getattr(v, 'mpmi', np.nan))

        for d in ll_list:
            ll_list_total.append(list(d.values())[0].logprob)
        # au = np.array(au_list).sum()
        # tu = np.array(tu_list).sum()
        # eu = np.array(eu_list).sum()
        mpmi = np.array(mpmi_list).sum()

        ll = np.array(ll_list_total).mean()

        return {
            'll': ll,
            # original uncertainty returns commented out while keeping placeholders
            # 'au': au, 'tu': tu, 'eu': eu,
            'au': np.nan, 'tu': np.nan, 'eu': np.nan,
            'mpmi': mpmi,
        }
    except Exception as e:
        return None


def get_top_p_acc(df, p, col):
    """Get accuracy for top p% samples sorted by column"""
    # ignore rows where the metric is NaN when selecting top p%
    df_sorted = df.dropna(subset=[col]).sort_values(by=col, ascending=False)
    top_p = int(len(df_sorted) * p)
    if top_p == 0:
        top_p = 1
    return df_sorted.iloc[:top_p]['label'].mean()


def safe_auc(labels, scores):
    """Safely compute AUC score"""
    try:
        y = np.asarray(labels)
        s = np.asarray(scores)
        # mask out NaNs in scores or labels
        mask = ~np.isnan(s) & ~np.isnan(y)
        if np.sum(mask) == 0:
            return np.nan
        y_masked = y[mask]
        s_masked = s[mask]
        # need at least one positive and one negative label
        if len(np.unique(y_masked)) < 2:
            return np.nan
        return roc_auc_score(y_masked, s_masked)
    except Exception:
        return np.nan


def safe_auprc(labels, scores):
    """Safely compute AUPRC score"""
    try:
        y = np.asarray(labels)
        s = np.asarray(scores)
        mask = ~np.isnan(s) & ~np.isnan(y)
        if np.sum(mask) == 0:
            return np.nan
        y_masked = y[mask]
        s_masked = s[mask]
        # require at least one positive label to compute PR
        if np.sum(y_masked == 1) == 0:
            return np.nan
        precision, recall, _ = precision_recall_curve(y_masked, s_masked)
        return auc(recall, precision)
    except Exception:
        return np.nan


def bootstrap_scalar_statistic(values, statistic=np.mean):
    clean_values = np.asarray([float(value) for value in values if value is not None and not np.isnan(value)], dtype=float)
    
    # 调试信息
    print(f"[DEBUG] clean_values size: {clean_values.size}")
    print(f"[DEBUG] clean_values: {clean_values}")
    
    if clean_values.size == 0:
        return {
            'estimate': np.nan,
            'bootstrap_mean': np.nan,
            'bootstrap_std': np.nan,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'bootstrap_count': 0,
        }

    estimate = float(statistic(clean_values))
    if clean_values.size == 1:
        print(f"[DEBUG] Only 1 value, returning std=0.0")
        return {
            'estimate': estimate,
            'bootstrap_mean': estimate,
            'bootstrap_std': 0.0,
            'ci_low': estimate,
            'ci_high': estimate,
            'bootstrap_count': 1,
        }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = rng.choice(clean_values, size=clean_values.size, replace=True)
        bootstrap_values.append(float(statistic(sample)))

    bootstrap_values = np.asarray(bootstrap_values, dtype=float)
    return {
        'estimate': estimate,
        'bootstrap_mean': float(np.mean(bootstrap_values)),
        'bootstrap_std': float(np.std(bootstrap_values, ddof=0)),
        'ci_low': float(np.percentile(bootstrap_values, 2.5)),
        'ci_high': float(np.percentile(bootstrap_values, 97.5)),
        'bootstrap_count': int(bootstrap_values.size),
    }


def format_metric_with_error(value, bootstrap_std, ci_low, ci_high):
    if np.isnan(value):
        return 'nan'
    if np.isnan(bootstrap_std):
        return f"{value:.4f}"
    if np.isnan(ci_low) or np.isnan(ci_high):
        return f"{value:.4f} ± {bootstrap_std:.4f}"
    return f"{value:.4f} ± {bootstrap_std:.4f} [{ci_low:.4f}, {ci_high:.4f}]"


def build_summary_row(metric, values):
    values = pd.Series(values).dropna()
    bootstrap = bootstrap_scalar_statistic(values)
    if len(values) > 0:
        metric_mean = float(values.mean())
        metric_std = float(values.std())
        metric_min = float(values.min())
        metric_max = float(values.max())
    else:
        metric_mean = np.nan
        metric_std = np.nan
        metric_min = np.nan
        metric_max = np.nan

    return {
        'metric': metric,
        'mean': metric_mean,
        'std': metric_std,
        'min': metric_min,
        'max': metric_max,
        'count': int(len(values)),
        'bootstrap_mean': bootstrap['bootstrap_mean'],
        'bootstrap_std': bootstrap['bootstrap_std'],
        'bootstrap_ci_low': bootstrap['ci_low'],
        'bootstrap_ci_high': bootstrap['ci_high'],
        'bootstrap_count': bootstrap['bootstrap_count'],
    }


def create_results_directory(model, dataset):
    """Create organized results directory structure"""
    results_dir = os.path.join(BASE_DIR, 'results', 'eval', f'{model}_{dataset}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_metadata(results_dir, args, seeds, successful_seeds):
    """Save evaluation metadata"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'model': args.model,
        'results_subdir': args.results_subdir,
        'seeds_requested': seeds,
        'seeds_successful': successful_seeds,
        'seeds_failed': [s for s in seeds if s not in successful_seeds],
        'success_rate': f"{len(successful_seeds)}/{len(seeds)}",
    }
    
    with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def evaluate_single_seed(dataset, model, seed, results_subdir, nli_tokenizer=None, nli_model=None):
    """Evaluate a single seed and return metrics"""
    # Find pickle files
    pattern = os.path.join(BASE_DIR, 'results', f'{model}_results_vllm_pg', dataset, f'seed{seed}', results_subdir, '*.pkl')
    files = glob(pattern)
    if not files:
        print(f"No files found for seed {seed}")
        return None
    
    # Load data
    data = []
    for file in files:
        try:
            with open(file, 'rb') as f:
                data += pickle.load(f)
        except Exception as e:
            print(f"Error loading {os.path.basename(file)}: {e}")
    
    if not data:
        print(f"No data loaded for seed {seed}")
        return None
    
    # Process data
    processed_data = []
    for sample in data:
        try:
            if isinstance(sample['result'], list):
                result = sample['result'][0]
            else:
                result = sample['result']

            answer = extract(result.outputs[0].text)
            answers = sample.get('answers')
            if isinstance(answers, dict):
                answers = answers.get('text', [])
            answers = flatten_answers(answers)

            if not answers:
                raw_answer = sample.get('answer')
                ground_truths = flatten_answers(raw_answer)
            else:
                ground_truths = [str(x) for x in answers if x is not None]

            # Get question for context
            question = sample.get('question', '')

            # Compute label (correctness)
            if is_reading_comprehension_dataset(dataset):
                # Only use NLI bidirectional entailment for reading comprehension
                label = False
                nli_score = 0.0
                if nli_model is not None and nli_tokenizer is not None:
                    for truth in ground_truths:
                        eq, forward_entail, backward_entail = nli_bidirectional_entailment(
                            answer, truth, question, nli_tokenizer, nli_model
                        )
                        if eq:
                            label = True
                            nli_score = max(nli_score, min(forward_entail, backward_entail))
                else:
                    print(f"Warning: NLI model not loaded, cannot evaluate {dataset} dataset")
            elif 'deepscaler' in dataset or 'gsm8k' in dataset or 'math500' in dataset or 'leg-counting' in dataset:
                label = math_equal(memoized_canonical_form(answer), memoized_canonical_form(sample.get('answer')))
                nli_score = None
            else:
                label = (answer.strip().lower() == sample.get('answer', '').strip().lower())
                nli_score = None
            # Extract uncertainty scores
            unc = extract_unc(result.outputs[0])
            if unc is None:
                continue
                
            row = {
                'unique_id': sample['unique_id'],
                'label': label,
                'll': unc['ll'],
                # 'au': -unc['au'],
                # 'tu': -unc['tu'],
                # 'eu': -unc['eu'],
                'mpmi': -unc.get('mpmi', np.nan),
            }
            if is_reading_comprehension_dataset(dataset):
                row['nli_score'] = nli_score
            processed_data.append(row)
            
        except Exception as e:
            continue

    if not processed_data:
        print(f"No samples processed successfully for seed {seed}")
        return None
        
    df = pd.DataFrame(processed_data)
    df.set_index('unique_id', inplace=True)
    
    # Bootstrap on samples within this seed
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed)  # Different seed per bootstrap
    bootstrap_results = {
        'seed': seed,
        'total_samples': len(df),
        'correct_samples': df['label'].sum(),
    }
    
    # Available metrics: only consider 'mpmi'
    eval_metrics = []
    if 'mpmi' in df.columns and not df['mpmi'].isna().all():
        eval_metrics.append('mpmi')

    # Bootstrap sampling on samples
    bootstrap_accuracies = []
    bootstrap_auc_scores = {m: [] for m in eval_metrics}
    bootstrap_auprc_scores = {m: [] for m in eval_metrics}
    bootstrap_top_p_scores = {f'top{int(p*100)}_{m}': [] for p in [0.1, 0.25, 0.5, 0.75] for m in eval_metrics}

    for _ in range(BOOTSTRAP_RESAMPLES):
        # Sample with replacement
        indices = rng.choice(len(df), size=len(df), replace=True)
        df_sample = df.iloc[indices]

        # Accuracy
        acc = df_sample['label'].mean()
        bootstrap_accuracies.append(acc)

        # AUC and AUPRC
        for metric in eval_metrics:
            auc_score = safe_auc(df_sample['label'], df_sample[metric])
            bootstrap_auc_scores[metric].append(auc_score)

            auprc_score = safe_auprc(df_sample['label'], df_sample[metric])
            bootstrap_auprc_scores[metric].append(auprc_score)

            # Top P% scores
            for p in [0.1, 0.25, 0.5, 0.75]:
                try:
                    top_p_acc = get_top_p_acc(df_sample, p, metric)
                    bootstrap_top_p_scores[f'top{int(p*100)}_{metric}'].append(top_p_acc)
                except:
                    bootstrap_top_p_scores[f'top{int(p*100)}_{metric}'].append(np.nan)
    
    # Calculate statistics from bootstrap samples
    metrics = bootstrap_results.copy()
    
    # Accuracy with bootstrap CI
    bootstrap_acc = np.array(bootstrap_accuracies)
    metrics['overall_accuracy'] = float(np.mean(bootstrap_acc))
    metrics['accuracy_bootstrap_std'] = float(np.std(bootstrap_acc, ddof=0))
    metrics['accuracy_ci_low'] = float(np.percentile(bootstrap_acc, 2.5))
    metrics['accuracy_ci_high'] = float(np.percentile(bootstrap_acc, 97.5))
    
    # AUC Scores with bootstrap CI
    for metric in eval_metrics:
        values = np.array([v for v in bootstrap_auc_scores[metric] if not np.isnan(v)])
        if len(values) > 0:
            metrics[f'auc_{metric}'] = float(np.mean(values))
            metrics[f'auc_{metric}_std'] = float(np.std(values, ddof=0))
            metrics[f'auc_{metric}_ci_low'] = float(np.percentile(values, 2.5))
            metrics[f'auc_{metric}_ci_high'] = float(np.percentile(values, 97.5))
    
    # AUPRC Scores with bootstrap CI
    for metric in eval_metrics:
        values = np.array([v for v in bootstrap_auprc_scores[metric] if not np.isnan(v)])
        if len(values) > 0:
            metrics[f'auprc_{metric}'] = float(np.mean(values))
            metrics[f'auprc_{metric}_std'] = float(np.std(values, ddof=0))
            metrics[f'auprc_{metric}_ci_low'] = float(np.percentile(values, 2.5))
            metrics[f'auprc_{metric}_ci_high'] = float(np.percentile(values, 97.5))
    
    # Top P% Accuracies with bootstrap CI
    for key in bootstrap_top_p_scores:
        values = np.array([v for v in bootstrap_top_p_scores[key] if not np.isnan(v)])
        if len(values) > 0:
            metrics[key] = float(np.mean(values))
            metrics[f'{key}_std'] = float(np.std(values, ddof=0))
            metrics[f'{key}_ci_low'] = float(np.percentile(values, 2.5))
            metrics[f'{key}_ci_high'] = float(np.percentile(values, 97.5))
    
    return metrics


def create_summary_report(df_metrics, results_dir, args, successful_seeds):
    """Create a comprehensive summary report"""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("MULTI-SEED EVALUATION SUMMARY REPORT (With Bootstrap Sampling)")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: {args.dataset}")
    report_lines.append(f"Model: {args.model}")
    report_lines.append(f"Results Subdirectory: {args.results_subdir}")
    report_lines.append(f"Successfully Evaluated Seeds: {successful_seeds}")
    report_lines.append(f"Total Seeds: {len(successful_seeds)}")
    report_lines.append(f"Bootstrap Resamples per Seed: {BOOTSTRAP_RESAMPLES}")
    report_lines.append("")
    
    # Overall accuracy with bootstrap CI from samples
    report_lines.append("OVERALL ACCURACY (Bootstrap from samples):")
    for idx, row in df_metrics.iterrows():
        seed = row['seed']
        acc = row['overall_accuracy']
        acc_std = row.get('accuracy_bootstrap_std', np.nan)
        acc_ci_low = row.get('accuracy_ci_low', np.nan)
        acc_ci_high = row.get('accuracy_ci_high', np.nan)
        report_lines.append(
            f"  Seed {seed}: {format_metric_with_error(acc, acc_std, acc_ci_low, acc_ci_high)}"
        )
    report_lines.append("")
    
    # AUC scores summary
    report_lines.append("AUC SCORES (Bootstrap from samples):")
    auc_cols = [col for col in df_metrics.columns if col.startswith('auc_') and not col.endswith('_std') and not col.endswith('_ci_low') and not col.endswith('_ci_high')]
    for col in sorted(auc_cols):
        metric = col.replace('auc_', '').upper()
        report_lines.append(f"  {metric}:")
        for idx, row in df_metrics.iterrows():
            seed = row['seed']
            value = row[col]
            std = row.get(f'{col}_std', np.nan)
            ci_low = row.get(f'{col}_ci_low', np.nan)
            ci_high = row.get(f'{col}_ci_high', np.nan)
            if not np.isnan(value):
                report_lines.append(
                    f"    Seed {seed}: {format_metric_with_error(value * 100, std * 100, ci_low * 100, ci_high * 100)}"
                )
    report_lines.append("")
    
    # AUPRC scores summary
    report_lines.append("AUPRC SCORES (Bootstrap from samples):")
    auprc_cols = [col for col in df_metrics.columns if col.startswith('auprc_') and not col.endswith('_std') and not col.endswith('_ci_low') and not col.endswith('_ci_high')]
    for col in sorted(auprc_cols):
        metric = col.replace('auprc_', '').upper()
        report_lines.append(f"  {metric}:")
        for idx, row in df_metrics.iterrows():
            seed = row['seed']
            value = row[col]
            std = row.get(f'{col}_std', np.nan)
            ci_low = row.get(f'{col}_ci_low', np.nan)
            ci_high = row.get(f'{col}_ci_high', np.nan)
            if not np.isnan(value):
                report_lines.append(
                    f"    Seed {seed}: {format_metric_with_error(value * 100, std * 100, ci_low * 100, ci_high * 100)}"
                )
    report_lines.append("")
    
    # Top 50% accuracy (highlighted)
    report_lines.append("TOP 50% ACCURACY (Bootstrap from samples):")
    top50_cols = [col for col in df_metrics.columns if col.startswith('top50_') and not col.endswith('_std') and not col.endswith('_ci_low') and not col.endswith('_ci_high')]
    for col in sorted(top50_cols):
        metric = col.replace('top50_', '').upper()
        report_lines.append(f"  {metric}:")
        for idx, row in df_metrics.iterrows():
            seed = row['seed']
            value = row[col]
            std = row.get(f'{col}_std', np.nan)
            ci_low = row.get(f'{col}_ci_low', np.nan)
            ci_high = row.get(f'{col}_ci_high', np.nan)
            if not np.isnan(value):
                report_lines.append(
                    f"    Seed {seed}: {format_metric_with_error(value * 100, std * 100, ci_low * 100, ci_high * 100)}"
                )
    report_lines.append("")
    
    # Best performing metrics
    report_lines.append("BEST PERFORMING METRICS ACROSS SEEDS:")
    
    auc_cols = [col for col in df_metrics.columns if col.startswith('auc_') and not col.endswith('_std') and not col.endswith('_ci_low') and not col.endswith('_ci_high')]
    if auc_cols:
        best_auc_col = max(auc_cols, key=lambda x: df_metrics[x].mean())
        best_auc_metric = best_auc_col.replace('auc_', '').upper()
        best_auc_value = df_metrics[best_auc_col].mean()
        report_lines.append(f"  Best AUC: {best_auc_metric} ({best_auc_value*100:.2f})")
    
    top50_cols = [col for col in df_metrics.columns if col.startswith('top50_') and not col.endswith('_std') and not col.endswith('_ci_low') and not col.endswith('_ci_high')]
    if top50_cols:
        best_top50_col = max(top50_cols, key=lambda x: df_metrics[x].mean())
        best_top50_metric = best_top50_col.replace('top50_', '').upper()
        best_top50_value = df_metrics[best_top50_col].mean() * 100
        report_lines.append(f"  Best Top 50%: {best_top50_metric} ({best_top50_value:.2f})")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_file = os.path.join(results_dir, 'evaluation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Print report
    print('\n'.join(report_lines))
    
    return report_file


def main():
    parser = argparse.ArgumentParser(description='multi-seed evaluation for uncertainty results')
    parser.add_argument('--dataset', default='math500', help='Dataset name')
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seeds to evaluate')
    parser.add_argument('--seeds_range', nargs=2, type=int, help='Range of seeds (start, end)')
    parser.add_argument('--model', default='qwen3b', help='Model name')
    parser.add_argument('--results_subdir', default='greedy_unc',
                       help='Results subdirectory')
    
    args = parser.parse_args()
    
    # Load default NLI model for reading comprehension evaluation
    nli_tokenizer, nli_model = None, None
    try:
        nli_tokenizer, nli_model = load_nli_model("microsoft/deberta-large-mnli")
    except Exception as e:
        print(f"Warning: failed to load default NLI model: {e}")

    # Determine seeds to evaluate
    if args.seeds:
        seeds = args.seeds
    elif args.seeds_range:
        seeds = list(range(args.seeds_range[0], args.seeds_range[1] + 1))
    else:
        # Default seeds
        seeds = [96, 89, 64]
    
    # Create results directory
    results_dir = create_results_directory(args.model, args.dataset)
    
    print(f"{'='*80}")
    print(f"MULTI-SEED UNCERTAINTY EVALUATION")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seeds: {seeds}")
    print(f"Results will be saved to: {results_dir}")
    
    # Evaluate all seeds
    all_metrics = []
    successful_seeds = []
    
    for seed in seeds:
        print(f"\nEvaluating seed {seed}...")
        metrics = evaluate_single_seed(
            args.dataset,
            args.model,
            seed,
            args.results_subdir,
            nli_tokenizer=nli_tokenizer,
            nli_model=nli_model,
        )
        if metrics:
            all_metrics.append(metrics)
            successful_seeds.append(seed)
            print(f"Seed {seed}: {metrics['correct_samples']}/{metrics['total_samples']} correct ({metrics['overall_accuracy']:.4f})")
        else:
            print(f"Seed {seed}: Failed to evaluate")
    
    if not all_metrics:
        print("\nNo seeds were successfully evaluated!")
        return
    
    # Convert to DataFrame for easier analysis
    df_metrics = pd.DataFrame(all_metrics)
    
    # Save metadata
    metadata = save_metadata(results_dir, args, seeds, successful_seeds)
    print(f"\nMetadata saved to: {os.path.join(results_dir, 'metadata.json')}")
    
    # Save detailed results
    detailed_file = os.path.join(results_dir, 'detailed_results.csv')
    df_metrics.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    # Create comprehensive report
    report_file = create_summary_report(df_metrics, results_dir, args, successful_seeds)
    print(f"Evaluation report saved to: {report_file}")
    
    print(f"\n{'='*80}")
    print(f"Multi-seed evaluation completed successfully!")
    print(f"All results saved in: {results_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
