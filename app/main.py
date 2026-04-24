"""FastAPI entrypoint for the Relevant Priors challenge.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Docker image is in the repo root.
"""
from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import FastAPI, Request

from app.predictor import predict_case
from app.schemas import PredictRequest, PredictResponse, Prediction

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("relevant-priors")

app = FastAPI(title="Relevant Priors API", version="1.0.0")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"service": "relevant-priors", "predict": "POST /predict"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request) -> PredictResponse:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    case_count = len(req.cases)
    prior_count = sum(len(c.prior_studies) for c in req.cases)

    log.info(
        "predict rid=%s challenge_id=%s cases=%d priors=%d",
        rid, req.challenge_id, case_count, prior_count,
    )

    t0 = time.perf_counter()
    predictions: list[Prediction] = []
    for case in req.cases:
        prior_descs = [p.study_description or "" for p in case.prior_studies]
        flags = predict_case(case.current_study.study_description or "", prior_descs)
        for prior, is_rel in zip(case.prior_studies, flags):
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=bool(is_rel),
                )
            )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    log.info(
        "predict rid=%s done cases=%d priors=%d predictions=%d elapsed_ms=%.1f",
        rid, case_count, prior_count, len(predictions), elapsed_ms,
    )

    return PredictResponse(predictions=predictions)
