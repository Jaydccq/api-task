"""Train a small logistic regression on pair features + cross-validate.

If CV accuracy beats the rule-only predictor, we print the learned
intercept and coefficients for embedding into `app/model.py`. Otherwise
we keep the rule-only predictor.

Usage:
    python3 scripts/train.py [path/to/public.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.features import FEATURE_NAMES, extract_features  # noqa: E402
from app.predictor import predict_case  # noqa: E402


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_matrix(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    truth = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"] for t in data["truth"]}
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []  # case_id — used to keep a case together in CV folds
    ids: list[tuple[str, str]] = []
    for case in data["cases"]:
        n_priors = len(case["prior_studies"])
        cur_desc = case["current_study"]["study_description"]
        for p in case["prior_studies"]:
            lbl = truth.get((case["case_id"], p["study_id"]))
            if lbl is None:
                continue
            rows.append(extract_features(cur_desc, p["study_description"], n_priors))
            labels.append(int(bool(lbl)))
            groups.append(case["case_id"])
            ids.append((case["case_id"], p["study_id"]))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), np.asarray(groups), ids


def rule_accuracy(data: dict) -> float:
    truth = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"] for t in data["truth"]}
    correct = 0
    total = 0
    for case in data["cases"]:
        preds = predict_case(case["current_study"]["study_description"], [p["study_description"] for p in case["prior_studies"]])
        for pred, p in zip(preds, case["prior_studies"]):
            lbl = truth.get((case["case_id"], p["study_id"]))
            if lbl is None:
                continue
            correct += int(bool(pred) == bool(lbl))
            total += 1
    return correct / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=str(REPO_ROOT / "relevant_priors_public.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--model", choices=["lr", "gbm"], default="lr")
    args = parser.parse_args()

    data = load(Path(args.path))
    X, y, groups, _ids = build_matrix(data)
    print(f"Loaded {X.shape[0]} pairs, {X.shape[1]} features, base rate True = {y.mean():.3f}")

    rule_acc = rule_accuracy(data)
    print(f"Rule-only accuracy: {rule_acc:.4f}")

    # Group k-fold: cases are kept together in a fold.
    gkf = GroupKFold(n_splits=args.folds)
    fold_accs = []
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(np.unique(groups)))  # noop, but sets a seed
    def make_clf():
        if args.model == "gbm":
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                random_state=args.seed,
            )
        return LogisticRegression(C=args.C, max_iter=2000, solver="lbfgs")

    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        acc = (preds == y[te]).mean()
        fold_accs.append(acc)
        print(f"  Fold {fold_idx + 1}: train={len(tr)} test={len(te)} acc={acc:.4f}")
    mean = float(np.mean(fold_accs))
    std = float(np.std(fold_accs))
    print(f"CV accuracy: {mean:.4f} ± {std:.4f}")

    # Final model on all data for coefficient inspection / export.
    final = make_clf()
    final.fit(X, y)
    print()
    if args.model == "lr":
        print(f"{'feature':<34} {'coef':>10}")
        for name, coef in zip(FEATURE_NAMES, final.coef_[0]):
            print(f"  {name:<32} {coef:>+10.4f}")
        print(f"  {'intercept':<32} {final.intercept_[0]:>+10.4f}")
    else:
        print("Top feature importances (GBM):")
        imp = final.feature_importances_
        for name, v in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1]):
            print(f"  {name:<32} {v:>+10.4f}")

    print()
    if mean > rule_acc + 0.001:
        print(f"MODEL WINS over rule-only by {(mean - rule_acc) * 100:.2f} pp — embed coefficients")
    else:
        print(f"MODEL DOES NOT IMPROVE vs rule-only (rule={rule_acc:.4f} vs CV={mean:.4f})")
        print("Keep rule-only predictor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
