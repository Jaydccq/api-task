# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HTTP service for the New Lantern `relevant-priors-v1` challenge. Given a
current radiology examination plus a list of priors, `/predict` returns a
boolean per prior indicating clinical relevance. The predictor is a
deterministic regex heuristic — no external calls at request time, so it
cannot time out against the evaluator's 360 s budget.

Current accuracy on the public split: **0.9665** (TP 5997 / FP 356 / TN
20691 / FN 570) in ~1.1 s across all 996 cases / 27,614 priors. See
`experiments.md` for the full rule-ablation log.

## Commands

```bash
# Local dev (Python 3.12 recommended to match Docker base image)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Tests (httpx is needed for FastAPI's TestClient)
pip install pytest httpx
python -m pytest tests/
python -m pytest tests/test_predictor.py::test_name   # single test

# Accuracy harness — needs relevant_priors_public.json in repo root.
# Prints accuracy/precision/recall plus top FP/FN confusion pairs (the
# primary signal for which rule to adjust next).
python3 scripts/evaluate.py
python3 scripts/evaluate.py --top 30

# Optional ML baseline — only useful to confirm the heuristic is not
# leaving meaningful accuracy on the table. Requires numpy + scikit-learn
# (not in requirements.txt; install separately if used).
python3 scripts/train.py

# Container (respects $PORT for Render/Fly/Cloud Run/Railway)
docker build -t relevant-priors .
docker run --rm -p 8000:8000 relevant-priors
```

## Architecture

The request path is intentionally shallow:

```
app/main.py        FastAPI layer — request logging, calls predict_case per case
app/predictor.py   Pure heuristic — the entire classifier
app/schemas.py     Pydantic models; extra="ignore" everywhere for forward-compat
app/features.py    Feature extraction used ONLY by scripts/train.py (not /predict)
```

`predict_case` pre-computes the current study's signature once per case
and reuses it across all priors. Do not re-introduce per-prior recomputation
of the current signature — it measurably widens the evaluator latency budget
on cases with hundreds of priors.

### Decision pipeline (short-circuit on first match)

`app/predictor.py::predict` / `predict_case` run **negative gates first**,
then fall through to the positive rules.

**Negative (return False):**

0. `_UNINFORMATIVE_BARE_PRIOR` — prior normalizes to a single generic
   token (currently `"PELVIC"` only; 5 % True on public split).
0b. `_NEGATIVE_PAIR_RULES` — `(name, cur_pat, prior_pat)` anti-rules for
   cross-rule false positives we can't fix by narrowing any single rule
   (e.g. CTA HEAD vs. carotid US via the neurovascular family alias;
   brain vs. sinus prior; DXA hip vs. unilateral hip XR).
0c. `_is_laterality_mismatch` — opposite side tokens (R vs. L) on a
   shared **paired region** (BREAST / KNEE / HIP / SHOULDER / ANKLE /
   FOOT / HAND / ELBOW / LEG). Midline regions (CHEST, ABDOMEN, SPINE_*,
   BRAIN, HEART) are excluded because left/right descriptors there still
   refer to the same underlying study. This single gate caught ≈55 FP
   (mammography opposite-side alone was +42 TN / -1 TP).

**Positive (return True):**

1. **Normalized exact match** — `normalize()` uppercases, strips punctuation
   (so `ABD_PEL` matches `ABD PEL`), collapses whitespace.
2. **Region overlap** — 24 patterns in `REGION_PATTERNS`. Whole-body studies
   (PET, bone scan, skull-to-thigh) expand via `_WHOLEBODY_COVERS` to match
   torso priors. A `HEAD AND NECK` post-filter strips BRAIN from soft-tissue
   neck studies (fixes 265/270 such pairs).
3. **Family overlap** — `FAMILY_PATTERNS` groups mutually-relevant studies
   that may not share a region (mammography, cardiac, cancer_workup, etc.).
4. **Directional bridge rules** — `_BRIDGE_RULES` is an ordered list of
   `(name, current_pattern, prior_pattern)` tuples. Each is one-directional:
   add a reverse entry explicitly when the clinical relationship is
   bidirectional. `_BRIDGE_CURRENT_EXCLUSIONS` suppresses a bridge when the
   current description contains a context token (e.g. CHEMO / LUM / DEFINITY
   disable the cardiac → CT chest bridge; `\bMRI?\b` disables T-spine →
   chest XR because MRI T-spine currents hit a coin-flip label rate).

### How rules get added

All rules are data-driven. The workflow is:

1. Run `scripts/evaluate.py` and read the top FP/FN confusion pairs.
2. Identify a pattern in those pairs (a typo, an abbreviation, an asymmetric
   clinical relationship).
3. **Before writing the rule, measure its label rate on the public split**
   with an inline analysis script: `python3 -c "..." ` that counts True /
   False on the specific (cur_pat, prior_pat). Only add the rule when the
   net change (TPs gained − FPs gained, or FPs removed − TPs removed) is
   clearly positive — ideally ≥ +5 on the public split and ≥ 70 % / ≤ 30 %
   label rate so the decision is robust to private-split drift.
4. Prefer the narrowest mechanism:
   - Opposite-side laterality on a paired region → add the region to
     `_LATERALITY_PAIRED_REGIONS`, not a bespoke regex.
   - Cross-rule FP that the offending rule is otherwise earning its
     keep on → `_NEGATIVE_PAIR_RULES`.
   - Asymmetric clinical relationship → one-way `_BRIDGE_RULES` entry.
   - Reverse direction that's also net-positive → a second bridge
     entry, never an inflated pattern.
   - Symmetric region expansion is the last resort — an early attempt
     (HEART → CHEST, BREAST → CHEST) cost ~11 pp of accuracy.
5. Re-run `evaluate.py` and confirm accuracy moved in the predicted
   direction (within ~±5 of the measured net change — bigger deltas
   usually mean the new rule is interacting with an existing rule).
6. Append a row to the rule-ablation table in `experiments.md`.

Region and family additions are blunt instruments — they fire in many
directions at once and readily trade recall for precision losses. Bridge
rules and anti-rules are the default tools for asymmetric findings.

### Schema contract

`schemas.py` uses `extra="ignore"` on every inbound model so evaluator
envelope additions (`schema_version`, `generated_at`, new fields) never
break the service. One prediction is emitted per `(case_id, study_id)` in
`prior_studies`; missing or empty descriptions return `false`.
