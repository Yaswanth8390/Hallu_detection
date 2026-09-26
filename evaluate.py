"""
Evaluate whether the Jacobian trajectory + grounding features from
run_pipeline.py actually separate correct answers from hallucinated ones.

This is the "does it work" script: it doesn't just dump features, it reports
concrete numbers you can look at and be skeptical of --
  - per-feature correlation with the hallucination label (which single
    features carry signal, if any)
  - a simple logistic-regression classifier's held-out accuracy / ROC-AUC
    using ALL features together, vs. a majority-class baseline
  - a confusion matrix so you can see the failure pattern, not just one number

Run against real output:
    python evaluate.py --in results.csv

Run against a synthetic sanity-check CSV (no GPU/model needed) to confirm
this script's own logic is correct before trusting it on real results:
    python evaluate.py --synthetic
"""

import argparse
import csv
import random
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Columns that are inputs to the model, not metadata/text.
FEATURE_COLUMNS = [
    "grad_norm_final_layer", "grad_norm_mean", "grad_norm_max",
    "grad_norm_argmax_layer_frac", "late_to_early_ratio",
    "grad_grounding_fraction", "grad_entity_to_other_ratio",
    "ablate_logit_drop", "ablate_rank_worsened_by",
]
LABEL_COLUMN = "is_correct_heuristic"


def load_rows(path: str) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def rows_to_xy(rows: List[dict]):
    """Keep only rows that have every feature populated (some rows may lack
    grounding features if an entity span wasn't found) and a valid label.
    """
    X, y, kept = [], [], []
    for r in rows:
        if LABEL_COLUMN not in r or r[LABEL_COLUMN] == "":
            continue
        try:
            feats = [float(r[c]) for c in FEATURE_COLUMNS]
        except (KeyError, ValueError):
            continue  # missing grounding features for this row -- skip it
        X.append(feats)
        # label = 1 means HALLUCINATED (i.e. NOT correct), so the classifier
        # target is "detect hallucination", matching the problem statement.
        y.append(0 if r[LABEL_COLUMN].strip().lower() == "true" else 1)
        kept.append(r)
    return np.array(X), np.array(y), kept


def point_biserial_correlations(X: np.ndarray, y: np.ndarray) -> List[tuple]:
    """Correlation of each feature with the binary hallucination label --
    the cheapest possible check of "is there any signal at all here".
    """
    out = []
    for i, name in enumerate(FEATURE_COLUMNS):
        col = X[:, i]
        if np.std(col) < 1e-12:
            out.append((name, 0.0))
            continue
        corr = np.corrcoef(col, y)[0, 1]
        out.append((name, corr))
    return sorted(out, key=lambda t: abs(t[1]), reverse=True)


def evaluate(X: np.ndarray, y: np.ndarray, seed: int = 0):
    if len(set(y.tolist())) < 2:
        print("Only one class present in the labels -- can't evaluate discrimination. "
              "You likely need more/more-varied examples (TruthfulQA is designed to "
              "elicit false answers, so a mix is expected with enough examples).")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y if min(np.bincount(y)) > 1 else None
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)

    print("\n=== Feature correlations with hallucination label (|r|, sorted) ===")
    for name, corr in point_biserial_correlations(X, y):
        print(f"  {name:32s} r={corr:+.3f}")

    print("\n=== Classifier (all features) vs. majority-class baseline ===")
    print(f"  Baseline accuracy (always predict majority class): "
          f"{accuracy_score(y_test, baseline_preds):.3f}")
    print(f"  Logistic regression accuracy:                      "
          f"{accuracy_score(y_test, preds):.3f}")
    try:
        print(f"  Logistic regression ROC-AUC:                       "
              f"{roc_auc_score(y_test, probs):.3f}  (0.5 = no signal, 1.0 = perfect)")
    except ValueError:
        print("  ROC-AUC undefined for this split (likely too few test examples of one class).")

    print("\n=== Confusion matrix (rows=true, cols=predicted; 0=correct, 1=hallucinated) ===")
    print(confusion_matrix(y_test, preds))

    print(f"\n  n_train={len(y_train)}  n_test={len(y_test)}  "
          f"hallucination_rate={y.mean():.2f}")
    print("\nRead this skeptically: with typical pilot sizes (tens of examples), "
          "these numbers have wide error bars. Look at whether the SAME features "
          "come out on top across a few different random seeds / larger n before "
          "believing the ranking.")


def make_synthetic_csv(path: str, n: int = 200, signal_strength: float = 1.2, seed: int = 0):
    """Generate a fake results.csv with a KNOWN, controllable signal, so you can
    confirm evaluate.py's own logic (correlation calc, classifier, metrics) is
    correct before spending GPU time on the real pipeline. `grounding_fraction`
    is made informative by construction; the rest are noise. If evaluate.py
    can't recover that signal here, the bug is in evaluate.py, not in your model.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        hallucinated = rng.random() < 0.5
        # Informative feature: lower grounding fraction for hallucinated examples,
        # with noise -- this is the effect we HOPE the real pipeline shows.
        grounding_fraction = rng.gauss(0.15 if hallucinated else 0.15 + signal_strength * 0.2, 0.15)
        grounding_fraction = min(max(grounding_fraction, 0.0), 1.0)
        row = {
            "question": "synthetic question",
            "generated_text": "synthetic answer",
            "candidate_entity": "Entity",
            "grad_norm_final_layer": rng.gauss(0, 1),
            "grad_norm_mean": rng.gauss(0, 1),
            "grad_norm_max": rng.gauss(0, 1),
            "grad_norm_argmax_layer_frac": rng.random(),
            "late_to_early_ratio": rng.gauss(1, 0.3),
            "grad_grounding_fraction": grounding_fraction,
            "grad_entity_to_other_ratio": rng.gauss(0, 1),
            "ablate_logit_drop": rng.gauss(0, 1),
            "ablate_rank_worsened_by": rng.gauss(0, 1),
            "is_correct_heuristic": str(not hallucinated),
        }
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, default="results.csv")
    parser.add_argument("--synthetic", action="store_true",
                         help="Run against a generated sanity-check CSV instead of real results.")
    parser.add_argument("--synthetic-signal", type=float, default=1.2,
                         help="Signal strength for --synthetic. Set to 0 to verify the script "
                              "correctly reports 'no signal' rather than always finding one.")
    args = parser.parse_args()

    if args.synthetic:
        path = "synthetic_results.csv"
        make_synthetic_csv(path, signal_strength=args.synthetic_signal)
        print(f"Generated {path} with a KNOWN synthetic signal in `grad_grounding_fraction`.\n"
              "If the numbers below don't show that feature near the top of the "
              "correlation ranking and an ROC-AUC well above 0.5, the bug is in "
              "evaluate.py -- fix that before trusting real results.\n")
    else:
        path = args.in_path

    rows = load_rows(path)
    X, y, kept = rows_to_xy(rows)
    print(f"Loaded {len(rows)} rows, {len(kept)} usable after dropping rows with missing "
          f"features/labels.")
    if len(kept) == 0:
        print("No usable rows -- check that run_pipeline.py actually populated "
              "grounding features (needs a found entity span) and labels.")
        return
    evaluate(X, y)


if __name__ == "__main__":
    main()
