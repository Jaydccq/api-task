"""Evaluate the heuristic predictor against the labeled public JSON.

Usage:
    python3 scripts/evaluate.py [path/to/public.json]

Default path is `relevant_priors_public.json` next to the repo root.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Make `app` importable when running as a script from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.predictor import predict_case  # noqa: E402


def evaluate(data: dict) -> dict:
    truth = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"] for t in data["truth"]}

    tp = fp = tn = fn = 0
    missed = 0  # priors with no truth label — excluded from accuracy
    # Most common confusion pairs for debugging.
    fp_pairs: Counter[tuple[str, str]] = Counter()
    fn_pairs: Counter[tuple[str, str]] = Counter()

    t0 = time.perf_counter()
    for case in data["cases"]:
        cur_desc = case["current_study"]["study_description"]
        priors = case["prior_studies"]
        preds = predict_case(cur_desc, [p["study_description"] for p in priors])
        for p, pred in zip(priors, preds):
            lbl = truth.get((case["case_id"], p["study_id"]))
            if lbl is None:
                missed += 1
                continue
            if pred and lbl:
                tp += 1
            elif pred and not lbl:
                fp += 1
                fp_pairs[(cur_desc, p["study_description"])] += 1
            elif not pred and lbl:
                fn += 1
                fn_pairs[(cur_desc, p["study_description"])] += 1
            else:
                tn += 1
    elapsed = time.perf_counter() - t0

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "missed": missed,
        "seconds": elapsed,
        "fp_pairs": fp_pairs,
        "fn_pairs": fn_pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=str(REPO_ROOT / "relevant_priors_public.json"),
    )
    parser.add_argument("--top", type=int, default=15, help="Top-N confusion pairs to print")
    args = parser.parse_args()

    with open(args.path) as f:
        data = json.load(f)

    r = evaluate(data)
    print(f"Cases: {len(data['cases'])}  Labeled priors: {r['tp']+r['fp']+r['tn']+r['fn']}")
    print(f"Elapsed: {r['seconds']*1000:.1f} ms")
    print()
    print(f"Accuracy:  {r['accuracy']:.4f}")
    print(f"Precision: {r['precision']:.4f}   Recall: {r['recall']:.4f}")
    print(f"Confusion: TP={r['tp']}  FP={r['fp']}  TN={r['tn']}  FN={r['fn']}   (unlabeled skipped: {r['missed']})")
    print()
    print(f"Top {args.top} FALSE POSITIVE pairs (predicted True, actually False):")
    for (cur, pri), n in r["fp_pairs"].most_common(args.top):
        print(f"  {n:3}  CUR={cur!r}  | PRI={pri!r}")
    print()
    print(f"Top {args.top} FALSE NEGATIVE pairs (predicted False, actually True):")
    for (cur, pri), n in r["fn_pairs"].most_common(args.top):
        print(f"  {n:3}  CUR={cur!r}  | PRI={pri!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
