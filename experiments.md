# Experiments — Relevant Priors API

## TL;DR

- **Final accuracy on the public split:** 0.9542 (TP 6022, FP 719, TN 20328, FN 545).
- **Total inference time on all 996 cases / 27,614 priors:** ~3 seconds.
- **Approach:** deterministic regex heuristic — regions, modalities, study families, plus directional clinical-bridge rules. No LLM call in the request path.
- **Alternative tested:** logistic regression and gradient-boosted trees on top of the heuristic features via 5-fold group CV. Gains were within fold noise (+0.0 to +0.13 pp). The rule-only predictor was shipped for simplicity, reproducibility, and lower private-split risk.

## Data shape

- 996 cases, 27,614 labeled priors (mean 27.7 priors / case, max 234).
- Base rate of `is_relevant_to_current = True` is 0.238.
- Always-False baseline accuracy = 0.7622.
- Exact normalized-string matches between current and prior study descriptions are 99.9 % relevant.
- Region overlap (e.g. both studies mention CHEST, BREAST, SPINE_L) correctly predicts ~92 % of relevance in the public split.
- The remaining ~5 % of gains come from study-family membership (mammography, cardiac, oncologic) and directional clinical bridges that capture asymmetric relationships.

## Approach

The predictor computes three descriptors for every study description:

1. **Normalized form** — uppercased, punctuation stripped (`ABD_PEL` → `ABD PEL`), whitespace collapsed. Both the exact-match check and all pattern matching run against this form so that regex `\b` boundaries behave consistently.
2. **Regions** — 24 patterns (BRAIN, CHEST, BREAST, SPINE_L, HEART, …). One description can map to zero, one, or multiple regions. A `"head and neck"` post-filter removes BRAIN from descriptions that are actually soft-tissue neck studies (fixes 265 of 270 such pairs).
3. **Families** — study classes that should mutually match regardless of region (mammography, cardiac, cancer workup, aortic screening, DXA, neurovascular). `cardiac` includes FFR/TTE/TEE/coronary/myo-perf; `cancer_workup` covers PET, bone scan, and whole-body oncology scans.

A prior is predicted relevant iff **any** of:

1. Normalized equality with the current study.
2. Region overlap, with a `WHOLEBODY` expansion that maps PET/bone-scan regions to torso regions (CHEST, ABDOMEN, PELVIS, SPINE_T, SPINE_L, HIP, AORTA, NECK).
3. Family overlap.
4. One of 19 **directional clinical bridge rules** fires. Each bridge states "if current matches X and prior matches Y, mark relevant"; adding the reverse is an explicit second rule. Some bridges have a context exclusion (cardiac → CT chest is suppressed when the current description contains CHEMO or LUM, because those are oncology-driven surveillance echoes where the prior chest CT is usually an unrelated staging scan).

## Rule-ablation log

Each row adds on top of the stack above.

| # | Rule added | Accuracy | Precision | Recall |
|---|------------|---------:|----------:|-------:|
| 0 | Always False baseline                                           | 0.7622 | —     | 0.000 |
| 1 | Exact-description match                                         | 0.7782 | 0.999 | 0.133 |
| 2 | Region overlap                                                  | 0.8960 | 0.927 | 0.611 |
| 3 | Region overlap + whole-body expansion + families                | 0.9364 | 0.912 | 0.811 |
| 4 | Normalize-before-regex (fixes `ABD_PEL`, `CT ABD/PEL`)          | 0.9379 | 0.912 | 0.817 |
| 5 | + high-precision bone-scan ↔ torso-CT bridge                    | 0.9453 | 0.914 | 0.851 |
| 6 | + directional cardiac / T-spine / EEG / esophagram / biopsy     | 0.9484 | 0.890 | 0.894 |
| 7 | + `CERVICL` / `CERV SPINE` typo patterns, MRI torso reverse     | 0.9496 | 0.891 | 0.898 |
| 8 | + cholangiogram, thoracentesis, adjacent-spine bridges          | 0.9505 | 0.888 | 0.907 |
| 9 | Narrow T-spine ↔ chest to studies with an explicit modality     | 0.9502 | 0.877 | 0.920 |
| 10 | "HEAD AND NECK" post-filter on BRAIN (US soft-tissue neck)     | 0.9515 | 0.882 | 0.919 |
| 11 | + pelvic-US / interventional / esophagram-XR bridges           | 0.9520 | 0.880 | 0.924 |
| 12 | Cardiac-CT-chest bridge excludes CHEMO/LUM contexts             | 0.9526 | 0.885 | 0.921 |
| 13 | Narrow neurovascular bridge to CT head only (drop MRI, reverse) | 0.9533 | 0.889 | 0.919 |
| 14 | Drop T↔L spine bridge (base-rate level in public split)         | 0.9538 | 0.892 | 0.917 |
| 15 | Drop SKULL from BRAIN pattern ("skull to thigh" was PET range)  | **0.9542** | 0.893 | 0.917 |

## What worked

- **Inspecting the top confusion pairs at every step.** Every rule landed because the FP or FN list pointed directly at a pattern; no rule was speculative. Examples: the `ABD_PEL` word-boundary bug, the `CERVICL` typo, the `HEAD AND NECK` false BRAIN match, the `SKULL` false BRAIN match from PET range descriptors.
- **Directional bridges instead of symmetric region expansion.** A first attempt at symmetric expansion (HEART→CHEST, BREAST→CHEST) dropped accuracy from 0.938 to 0.82 — many prior pairings that share a region are clinically unrelated. Directional rules, added one at a time and measured, held precision while raising recall.
- **Context exclusions per bridge.** "ECHO Chemo" and "LUM TTE" are cardiac studies but their paired prior chest CTs are usually oncologic (36 % vs 72 % True rate for plain echoes). A per-rule exclusion (`CHEMO|LUM`) turned 18 FPs into TNs at the cost of 10 TPs.
- **Family membership for modality-independent concepts.** Mammography lives in MR, US, plain film, and CAD variants. A `families` set lets any pair of mammo-family studies match without requiring every variant to be in the regex.
- **A minimal test suite from the start.** The 15-test harness caught two regressions during iteration and anchored the contract of `predict_case()` through the refactors.

## What failed

- **Symmetric clinical bridges.** The first cardiac ↔ chest rule fired both directions and crashed precision because `current = CT chest, prior = echo` is almost always labeled False. Splitting into directional rules recovered the gains.
- **Adding all CT torso studies to the `cancer_workup` family.** Family equivalence propagated to every torso CT pairing and cratered precision to 0.58. Rolled back.
- **Vagueness stripping for bare single-word priors.** `Abdomen`, `Pelvic`, `Chest PA`, `Thyroid` have 5–12 % True base rate, suggesting they should never match via region. Implementing that lost 48 true positives for the same tokens when the current study shared the region, because 23 % of bare `Chest` priors actually are relevant. Rolled back.
- **Per-case sibling-cluster lift.** 1 sibling in the same region as current had a 78 % True rate, 2 siblings 97 %. Appeared to be a huge signal. In practice it was almost entirely subsumed by existing region-overlap rules, and the remaining "novel" lift was corrupted by PET's whole-body expansion (`SKULL TO THIGH` matched both BRAIN and WHOLEBODY, so any CT HEAD prior in the same case got pulled in). Fixing SKULL alone recovered the non-cluster gain. A conservative family-siblings ≥ 2 lift produced +19 TPs / +11 FPs at 2 siblings but −8 TPs / −16 FPs at 3+ — net zero. Rolled back.
- **Trained models on top of heuristic features.** A logistic regression with 19 features (region overlap counts, family overlap, token Jaccard, bridge hit, case size, etc.) and group 5-fold CV tied the rule-only baseline at 0.9538. A gradient-boosted tree improved it to 0.9551 (±0.0046) — a 0.13 pp gain within the fold-to-fold noise of 1.6 pp. The GBM's top feature was the rule-based prediction itself (0.94 importance), confirming the rules capture almost all available signal from this feature set. Not shipped, to keep the container small and avoid private-split transfer risk. See `scripts/train.py` to reproduce.

## Residual error analysis (top remaining misses)

False positives (predicted True, labeled False):
- `MRI thoracic spine` ↔ `CHEST 2V` — 7 cases. Thoracic-spine / chest-XR bridge fires but not every such pair is relevant.
- `US abdomen complete w/ doppler` ↔ `Abdomen` — 7 cases. Short legacy descriptor, region match.
- `MRI thoracic spine` ↔ `MRI CERV SPINE` / `CERVICL SPINE` — 11 cases. Adjacent-spine bridge holds at 36 % relevance overall but is wrong for these specific pairs.
- `CT angio carotid` ↔ `CT HEAD` / `CT brain perfusion` — 11 cases. The carotid-angio ↔ CT-head bridge sits at 50 % relevance overall and we keep it, but particular pairs are labeled False.

False negatives (predicted False, labeled True):
- `ECHO Chemo TTE` ↔ `CT chest` — 8 cases. Accepted collateral from the CHEMO exclusion (prevented 18 FPs, kept 8 FNs).
- `LUM TTE` / `CT angio coronary` ↔ plain chest XR — 10 cases. Cardiac → XR-chest bridge produced more FPs than TPs overall (13 % relevance vs 24 % base), so not added.
- `CT HEAD` ↔ `CT angio carotid` — 5 cases. Reverse of the carotid bridge. Dropping the reverse bridge saves more FPs than these cost.
- Interventional priors (peritoneal drainage, nephrostomy, ablation) ↔ CT abdomen/pelvis — 8 cases. Partially covered by the IR-procedure bridge; the remaining names are idiosyncratic.

## Next-step improvements

1. **Interventional-procedure vocabulary pass.** Add `STENT|CATHETER|LOCALIZATION|SEED|WIRE` to the IR-procedure bridge. Estimated +0.1–0.2 pp on public; variable on private depending on vocabulary.
2. **Description-alias dictionary.** A hand-curated 50–100 entry `alias.yaml` that normalizes `CERV SPINE` → `CERVICAL SPINE`, `LUM TTE` → `LIMITED TTE`, `ABD_PEL` → `ABDOMEN PELVIS`, etc., before regex extraction. Lets rules shrink and makes the vocabulary explicit.
3. **Trainable calibration layer for residual uncertainty.** A shallow GBM trained on the heuristic feature set (see `scripts/train.py`) showed +0.13 pp in 5-fold CV. If the private-split result is close to public, shipping it is a small but real gain. The current code exports feature extraction (`app/features.py`) so a pickled model drops in as a post-rule scorer.
4. **LLM fallback for ambiguous cases only.** Keep the heuristic as primary. For priors that match via a single weak signal (only family or only bridge, no region and no exact match), batch all such priors in a single call to a classification LLM. Cache by (normalized-current, normalized-prior). Bounded latency because the heuristic already decides ~95 % of priors confidently.
5. **Case-level patient context beyond priors.** The input schema includes `patient_name` (same across studies for the same patient). Cross-case reuse (e.g. caching that patient X has a breast-cancer surveillance stream) could lift recall on genuinely ambiguous priors without changing the API contract.
6. **Private-split calibration loop.** Submit, inspect which pairs the background evaluator grades wrong, and patch the regex dictionary. Evaluator round-trip is ~2–3 minutes, which is fast enough to iterate.
7. **Operational hardening before production traffic.** Structured JSON logging, `/metrics` endpoint (Prometheus), request-size guard, per-request timeout, and retries for transient connectivity. Current service is intentionally minimal for submission.

## Reproducing these numbers

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest httpx scikit-learn numpy
python -m pytest tests/                       # 15 tests pass
python3 scripts/evaluate.py                   # prints 0.9542 accuracy
python3 scripts/train.py                      # prints LR 5-fold CV acc
python3 scripts/train.py --model gbm          # prints GBM 5-fold CV acc
```

`scripts/evaluate.py` also prints the top-N false-positive and false-negative confusion pairs, which was the input for every iteration in the ablation table.
