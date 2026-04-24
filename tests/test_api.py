"""HTTP contract tests for /predict."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _request() -> dict:
    return {
        "challenge_id": "relevant-priors-v1",
        "schema_version": 1,
        "cases": [
            {
                "case_id": "1001016",
                "patient_id": "606707",
                "patient_name": "Doe, Jane",
                "current_study": {
                    "study_id": "3100042",
                    "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                    "study_date": "2026-03-08",
                },
                "prior_studies": [
                    {
                        "study_id": "2453245",
                        "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                        "study_date": "2020-03-08",
                    },
                    {
                        "study_id": "992654",
                        "study_description": "CT HEAD WITHOUT CNTRST",
                        "study_date": "2021-03-08",
                    },
                    {
                        "study_id": "555000",
                        "study_description": "MRI KNEE LEFT WO CONTRAST",
                        "study_date": "2015-01-01",
                    },
                ],
            }
        ],
    }


def test_healthz() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_one_per_prior() -> None:
    r = client.post("/predict", json=_request())
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    preds = body["predictions"]
    assert len(preds) == 3
    ids = {(p["case_id"], p["study_id"]) for p in preds}
    assert ids == {
        ("1001016", "2453245"),
        ("1001016", "992654"),
        ("1001016", "555000"),
    }
    # Boolean typing check
    for p in preds:
        assert isinstance(p["predicted_is_relevant"], bool)


def test_predict_empty_priors() -> None:
    req = {
        "challenge_id": "relevant-priors-v1",
        "cases": [
            {
                "case_id": "x",
                "current_study": {"study_id": "a", "study_description": "CT chest"},
                "prior_studies": [],
            }
        ],
    }
    r = client.post("/predict", json=req)
    assert r.status_code == 200
    assert r.json() == {"predictions": []}


def test_predict_tolerates_missing_optional_fields() -> None:
    req = {
        "cases": [
            {
                "case_id": "1",
                "current_study": {"study_id": "s1", "study_description": "CT chest"},
                "prior_studies": [
                    {"study_id": "s2", "study_description": "CT chest"},
                ],
            }
        ]
    }
    r = client.post("/predict", json=req)
    assert r.status_code == 200
    assert r.json()["predictions"][0]["predicted_is_relevant"] is True
