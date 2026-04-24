# Relevant Priors API — Execution Plan

**Date:** 2026-04-24
**Workstream:** New Lantern "relevant-priors-v1" challenge submission

## Background

Build and host an HTTP POST `/predict` endpoint that, for each prior examination
in a request, predicts whether it is clinically relevant to the current
examination. Evaluator times out at 360 s. Missing predictions count as wrong.

Provided artifact: `relevant_priors_public.json` — 996 cases, 27,614 labeled
priors. Label distribution: 76.22 % False, 23.78 % True. Always-False baseline
accuracy = 0.7622.

## Goal

1. Achieve the highest accuracy we can on the public split using a
   deterministic, fast heuristic (no LLM in the request path → zero timeout risk,
   no API keys, fully reproducible).
2. Ship a hostable service (FastAPI + Docker) and provide the deploy URL.
3. Deliver the three-item submission: endpoint URL, code zip, `experiments.md`.

## Success Criteria

- `scripts/evaluate.py` on the public JSON reports accuracy ≥ 0.92.
- `POST /predict` returns exactly one prediction per `(case_id, study_id)` in the
  request body. No skipped priors.
- End-to-end latency on the full public split < 3 s locally (well under the
  360 s evaluator budget).
- Zero external dependencies at request time (no LLM, no network).

## Assumptions

- `case_id` + `study_id` are unique within a case; request schema matches the
  brief exactly.
- Study descriptions are free text with a long tail of abbreviations. A
  curated regex dictionary of modalities + body regions plus a string-similarity
  fallback will be enough to clear the high-90s without a model.
- The private split comes from the same distribution (same description vocabulary
  style). If it does not, heuristic will still beat the always-False baseline.
- User will handle cloud deploy themselves (or via the included Dockerfile).
  I will not assume any cloud credentials are present.

## Scope

In scope:
- Deterministic predictor (regex + string similarity).
- FastAPI service with structured logging.
- Evaluation harness and test suite.
- Dockerfile, requirements, README, experiments.md, submission zip.

Out of scope (listed in write-up as next steps):
- LLM classifier (needs API key + careful batching to beat 360 s).
- Trainable model (e.g. gradient-boosted classifier over hand features).
- Persistent per-study cache across requests (not required; nothing stateful).

## File Structure

```
task/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, /predict, /healthz
│   ├── predictor.py         # pure heuristic: predict_case(current, priors)
│   └── schemas.py           # pydantic models matching the challenge schema
├── scripts/
│   └── evaluate.py          # load public JSON → run predictor → accuracy
├── tests/
│   ├── test_predictor.py
│   └── test_api.py
├── docs/exec-plans/2026-04-24-relevant-priors-api.md  (this file)
├── experiments.md           # submission write-up
├── Dockerfile
├── requirements.txt
├── README.md
└── relevant_priors_public.json  (data; excluded from zip)
```

## Implementation Steps

1. **Plan + scaffolding** (this file, directories).
2. **Predictor module** — single pure function
   `predict(current_desc, prior_desc) -> bool` plus helpers for modality/region
   extraction. No I/O.
3. **Schemas** — Pydantic models exactly matching the brief so validation
   rejects malformed requests early.
4. **FastAPI app** — POST `/predict`, GET `/healthz`. Log `challenge_id`,
   request ID, case count, and prior count per request.
5. **Evaluator script** — run the public JSON against the predictor, report
   overall accuracy, confusion matrix, and per-case accuracy histogram.
6. **Tests** — predictor unit tests covering the four regime buckets
   (exact-match / same-region / diff-region / diff-modality-diff-region) plus
   an API test that feeds a small request and asserts schema conformance.
7. **Iteration** — use evaluator output to identify top systematic errors,
   patch rules, re-run. Stop when marginal gain < 0.2 pp.
8. **Deployment packaging** — Dockerfile (python:3.12-slim), requirements
   pinned, `README.md` with `uvicorn` + `docker run` + fly.io example.
9. **Write-up** — `experiments.md`: baseline table, rule ablations, failure
   patterns, next steps.
10. **Zip** — `submission.zip` with app/, scripts/, tests/, docs/,
    requirements.txt, Dockerfile, README.md, experiments.md. Exclude the data
    file, `__pycache__`, `.venv`.

## Verification Approach

- `pytest tests/` green.
- `python3 scripts/evaluate.py` prints accuracy ≥ 0.92 with confusion matrix.
- `curl -X POST localhost:8000/predict -d @sample_request.json` returns a
  `predictions` array with one entry per input prior.

## Progress Log

- 2026-04-24 — Plan drafted. Public JSON inspected: 996 cases, 27,614 priors,
  base rate True = 0.238, All-False accuracy = 0.7622, exact-description-match
  accuracy ≈ 99.9 %, simple region-overlap heuristic accuracy = 0.896.
- 2026-04-24 — v2 predictor (normalize + region + family): 0.9379.
- 2026-04-24 — v6 (directional bridges): 0.9484.
- 2026-04-24 — v9 (cholangio/thoracentesis/spine adjacency, narrowed T-spine
  chest rule): **0.9502**. 15/15 tests pass. Local eval in 1.7 s total.
- 2026-04-24 — FastAPI + Docker + README + experiments.md complete. Service
  smoke-tested locally (GET /healthz, POST /predict return expected shape).
- 2026-04-24 — submission.zip produced from repo minus data + venv.
- 2026-04-24 (v2) — Optimization pass: HEAD-AND-NECK BRAIN filter (+0.13),
  IR-procedure and pelvic-US bridges (+0.05), cardiac CHEMO/LUM exclusion
  (+0.06), tighter neurovascular bridge (+0.07), drop T↔L spine (+0.05),
  drop SKULL from BRAIN (+0.04). Final: **0.9542**.
- 2026-04-24 (v2) — Tried and rejected: (a) bare-legacy-prior vagueness
  stripping — lost 48 TPs, reverted; (b) case-level sibling-cluster lift
  — corrupted by PET `SKULL TO THIGH` range; fixing SKULL alone captured
  the gain; (c) logistic regression on heuristic features — ties rule
  baseline; (d) gradient-boosted trees — +0.13 pp within fold noise, not
  worth the sklearn dependency.

## Key Decisions

- **Heuristic, not LLM, in the request path.** The evaluator budget is 360 s;
  at ~28 priors/case × 996 cases, even a batched LLM call has timeout risk and
  requires an API key we don't have. Heuristic is reproducible, auditable,
  and already at 89.6 %.
- **Single regex-per-region dictionary** rather than a larger ontology to keep
  the code readable and one-file-reviewable.

## Risks and Blockers

- **Private split distribution drift.** If descriptions use a different
  vocabulary style, hand-tuned regex loses recall. Mitigation: string-similarity
  fallback (normalized-token Jaccard) catches typos / punctuation variants.
- **Hosting.** Submission needs a public URL; the user must deploy (Docker
  image provided). If that blocks, `ngrok http 8000` over a local run is a
  short-lived workaround.

## Final Outcome

- Predictor accuracy on public split: **0.9542** (precision 0.893, recall 0.917).
- Service runs in a single container, zero external dependencies at request time.
- Deliverables produced:
  - `app/`, `tests/`, `scripts/evaluate.py`, `Dockerfile`, `requirements.txt`,
    `README.md`, `experiments.md`.
  - `submission.zip` at the repo root.
- Remaining for the user: deploy the Docker image to a public URL and paste
  (a) the endpoint URL, (b) `submission.zip`, and (c) `experiments.md` into
  the submission form.
