"""
Benchmark run_pipeline.py's per-word hallucination_label against TruthfulQA's
own lexical-overlap correctness heuristic (dataset.label_correctness), used
here as a noisy stand-in for ground truth (hallucinated == NOT correct).

This expects a CSV from the *current* run_pipeline.py -- one that stamps
`example_index`, `hallucination_label`, and `is_correct_heuristic` onto every
row. It will refuse to run against an older results CSV that doesn't have
these columns rather than silently computing nonsense.

Per-word labels are rolled up to one prediction per TruthfulQA question by
the fraction of that answer's words flagged `hallucination`. The rollup
threshold (--flag-fraction-threshold) is another knob that needs calibrating,
same as the pipeline's own --dependence-threshold / --entropy-threshold.

Run:
    python evaluate.py --in results_tokens.csv
"""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REQUIRED_COLUMNS = {
    "example_index", "question", "generated_text",
    "hallucination_label", "is_correct_heuristic",
}


def load_rows(path: str) -> List[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and not REQUIRED_COLUMNS.issubset(rows[0]):
        missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
        raise ValueError(
            f"{path} is missing columns {missing}. This script expects output "
            "from the current run_pipeline.py (with example_index, "
            "hallucination_label, is_correct_heuristic, etc.), not an older "
            "or partial results CSV."
        )
    return rows


def group_by_example(rows: List[dict]) -> Dict[str, List[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["example_index"]].append(row)
    return groups


def example_summary(group: List[dict]) -> dict:
    """Roll up one TruthfulQA example's word rows into a single prediction
    (flagged-word fraction) and ground-truth label.
    """
    n_words = len(group)
    n_flagged = sum(1 for row in group if row["hallucination_label"] == "hallucination")
    flagged_fraction = n_flagged / n_words if n_words else 0.0
    # is_correct_heuristic is constant across a group's rows -- it's an
    # answer-level label attached to every word row at generation time.
    is_correct = group[0]["is_correct_heuristic"].strip().lower() == "true"
    return {
        "question": group[0]["question"],
        "generated_text": group[0]["generated_text"],
        "n_words": n_words,
        "n_flagged": n_flagged,
        "flagged_fraction": flagged_fraction,
        "ground_truth_hallucinated": not is_correct,
    }


def report(summaries: List[dict], flag_fraction_threshold: float):
    y_true = np.array([1 if s["ground_truth_hallucinated"] else 0 for s in summaries])
    scores = np.array([s["flagged_fraction"] for s in summaries])
    y_pred = (scores > flag_fraction_threshold).astype(int)

    print(f"\n=== Answer-level hallucination benchmark (n={len(summaries)}) ===")
    print(f"  Ground-truth hallucination rate "
          f"(TruthfulQA lexical-overlap heuristic): {y_true.mean():.2f}")
    print(f"  Predicted hallucination rate "
          f"(>{flag_fraction_threshold:.2f} of an answer's words flagged): "
          f"{y_pred.mean():.2f}")

    if len(set(y_true.tolist())) < 2:
        print("\nOnly one ground-truth class present in this sample -- "
              "TruthfulQA's own heuristic called every answer the same way. "
              "Precision/recall/ROC-AUC aren't meaningful here; try a larger "
              "--n from run_pipeline.py (TruthfulQA is designed to elicit a "
              "mix of correct and incorrect answers, but small samples can "
              "still land on one side).")
        return

    print(f"\n  Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    try:
        print(f"  ROC-AUC (flagged-word fraction as score): "
              f"{roc_auc_score(y_true, scores):.3f}  (0.5 = no signal, 1.0 = perfect)")
    except ValueError:
        print("  ROC-AUC undefined for this sample (too few examples of one class).")

    print("\n  Confusion matrix (rows=ground truth, cols=predicted; "
          "0=not-hallucinated, 1=hallucinated):")
    print(confusion_matrix(y_true, y_pred))

    print("\nRead this skeptically for two independent reasons: (1) TruthfulQA's "
          "own lexical-overlap 'ground truth' is a rough heuristic, not human "
          "judgment -- see dataset.py's label_correctness docstring, and it "
          "will disagree with a careful human reader on plenty of answers. "
          "(2) --flag-fraction-threshold here and --dependence-threshold / "
          "--entropy-threshold in run_pipeline.py all need separate "
          "calibration; a single accuracy number from one threshold "
          "combination is not a finished benchmark.")


def show_disagreements(summaries: List[dict], flag_fraction_threshold: float, n: int):
    if n <= 0:
        return
    disagreements = [
        s for s in summaries
        if (s["flagged_fraction"] > flag_fraction_threshold) != s["ground_truth_hallucinated"]
    ]
    print(f"\n=== {len(disagreements)} answer-level disagreements "
          f"(predicted != TruthfulQA-heuristic label), showing up to {n} ===")
    for s in disagreements[:n]:
        print(f"\nQ: {s['question']}")
        print(f"A: {s['generated_text']}")
        print(f"  ground_truth_hallucinated={s['ground_truth_hallucinated']}  "
              f"flagged={s['n_flagged']}/{s['n_words']} words "
              f"({s['flagged_fraction']:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, default="results_tokens.csv")
    parser.add_argument("--flag-fraction-threshold", type=float, default=0.0,
                         help="predict an answer hallucinated if the fraction of "
                              "its words flagged `hallucination` exceeds this. "
                              "0.0 (default) means 'any flagged word counts'. "
                              "Needs calibration, same as the pipeline's own "
                              "thresholds.")
    parser.add_argument("--show-disagreements", type=int, default=5,
                         help="number of answer-level disagreements to print "
                              "(0 to skip)")
    args = parser.parse_args()

    rows = load_rows(args.in_path)
    groups = group_by_example(rows)
    summaries = [example_summary(group) for group in groups.values()]
    print(f"Loaded {len(rows)} token rows across {len(summaries)} answers "
          f"from {args.in_path}.")
    if not summaries:
        print("No answers to evaluate.")
        return

    report(summaries, args.flag_fraction_threshold)
    show_disagreements(summaries, args.flag_fraction_threshold, args.show_disagreements)


if __name__ == "__main__":
    main()
