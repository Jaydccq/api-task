# Relevant Priors API

HTTP service for the New Lantern `relevant-priors-v1` challenge. Given a
current radiology examination and a list of prior examinations, the
service returns a boolean per prior indicating whether it should be
surfaced to the radiologist reading the current study.

The predictor is a deterministic heuristic (regex feature extraction +
directional clinical-link rules). It does not call any external service
at request time, so there is zero risk of LLM-provider timeouts hitting
the 360-second evaluator budget.

## Accuracy

Measured on the public labeled split (996 cases, 27,614 priors):

| Model | Accuracy | Precision | Recall | Elapsed |
|---|---|---|---|---|
| Always False (baseline) | 0.7622 | — | 0.000 | — |
| This predictor | **0.9672** | 0.944 | 0.916 | ~1.2 s total |

See `experiments.md` for the rule-ablation log.

## Layout

```
app/
  main.py       # FastAPI entrypoint (POST /predict, GET /healthz)
  predictor.py  # Pure heuristic — extracted regions/modalities/families + bridges
  schemas.py    # Pydantic request/response models
scripts/
  evaluate.py   # Local accuracy harness against relevant_priors_public.json
  train.py      # Optional: trains an LR/GBM on heuristic features (CV)
tests/
  test_predictor.py
  test_api.py
Dockerfile
requirements.txt
experiments.md
```

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check the endpoint:

```bash
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"cases":[{"case_id":"1","current_study":{"study_id":"a","study_description":"CT chest"},"prior_studies":[{"study_id":"b","study_description":"CT CHEST WITHOUT CNTRST"}]}]}'
```

## Run tests

```bash
pip install pytest httpx
python -m pytest tests/
```

## Local evaluation on the public split

Download `relevant_priors_public.json` into the repo root, then:

```bash
python3 scripts/evaluate.py
```

## Docker

```bash
docker build -t relevant-priors .
docker run --rm -p 8000:8000 relevant-priors
```

The container respects `$PORT` — Render, Fly.io, Railway, and Cloud Run
all work without modification.

## Deploy

Pick any Docker-friendly host; no configuration beyond the port is needed.

### Fly.io

```bash
fly launch --no-deploy           # accept defaults; picks up Dockerfile
fly deploy
fly status                        # shows the public URL
```

### Render

1. New → Web Service, connect the repo (or upload a zip).
2. Environment: `Docker`. No env vars required.
3. Health check path: `/healthz`.

### Cloud Run

```bash
gcloud run deploy relevant-priors \
  --source . --region us-central1 --allow-unauthenticated
```

The deployed URL is the value you paste into the submission form.

## Endpoint contract

`POST /predict` accepts the request schema defined in the challenge brief
and returns exactly one prediction per `(case_id, study_id)` present in
`prior_studies`. Missing or empty study descriptions return `false`.
Unknown top-level fields are ignored for forward compatibility.
