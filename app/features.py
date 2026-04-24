"""Feature extraction for the (current_study, prior_study) pair.

Features are deliberately abstract (derived from region/modality/family
sets and from structural properties of the description) rather than from
raw tokens, so that a model trained on the public split has a chance to
generalize to a private split with a different vocabulary.

Keep `FEATURE_NAMES` in sync with `extract_features()`. The fixed order
is how the served model's coefficients are indexed.
"""
from __future__ import annotations

from math import log1p

from app.predictor import (
    _bridge_hit,
    _effective_regions,
    _sig,
    families,
    modalities,
    normalize,
    regions,
)


FEATURE_NAMES: list[str] = [
    "exact_match",
    "region_overlap_count",
    "effective_region_overlap_count",
    "region_current_count",
    "region_prior_count",
    "family_overlap_count",
    "modality_overlap_count",
    "both_ct",
    "both_mri",
    "both_xr",
    "both_us",
    "bridge_hit",
    "wholebody_either",
    "token_jaccard",
    "token_overlap_abs",
    "current_short",
    "prior_short",
    "num_priors_log",
    "rule_predict",
]


def extract_features(
    current_desc: str,
    prior_desc: str,
    num_priors_in_case: int = 1,
) -> list[float]:
    """Compute the fixed-order feature vector for one (current, prior) pair."""
    cur = _sig(current_desc or "")
    pri = _sig(prior_desc or "")

    cur_eff = _effective_regions(cur)
    pri_eff = _effective_regions(pri)

    cur_mods = set(cur.modalities)
    pri_mods = set(pri.modalities)

    cur_tokens = set(normalize(current_desc or "").split())
    pri_tokens = set(normalize(prior_desc or "").split())
    union = cur_tokens | pri_tokens
    jaccard = len(cur_tokens & pri_tokens) / max(1, len(union))
    overlap_abs = len(cur_tokens & pri_tokens)

    exact = 1.0 if cur.norm and cur.norm == pri.norm else 0.0

    # Rule-based prediction fold-in: lets the model learn a residual on top
    # of the hand-tuned rules rather than re-learning them.
    rule_pred = 0.0
    if exact:
        rule_pred = 1.0
    elif cur_eff & pri_eff:
        rule_pred = 1.0
    elif cur.families & pri.families:
        rule_pred = 1.0
    elif _bridge_hit(cur.norm, pri.norm):
        rule_pred = 1.0

    return [
        exact,
        float(len(cur.regions & pri.regions)),
        float(len(cur_eff & pri_eff)),
        float(len(cur.regions)),
        float(len(pri.regions)),
        float(len(cur.families & pri.families)),
        float(len(cur_mods & pri_mods)),
        1.0 if "CT" in cur_mods and "CT" in pri_mods else 0.0,
        1.0 if "MR" in cur_mods and "MR" in pri_mods else 0.0,
        1.0 if "XR" in cur_mods and "XR" in pri_mods else 0.0,
        1.0 if "US" in cur_mods and "US" in pri_mods else 0.0,
        1.0 if _bridge_hit(cur.norm, pri.norm) else 0.0,
        1.0 if "WHOLEBODY" in cur.regions or "WHOLEBODY" in pri.regions else 0.0,
        jaccard,
        float(overlap_abs),
        1.0 if len(cur_tokens) <= 2 else 0.0,
        1.0 if len(pri_tokens) <= 2 else 0.0,
        log1p(max(0, num_priors_in_case - 1)),
        rule_pred,
    ]
