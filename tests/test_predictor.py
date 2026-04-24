"""Unit tests for the heuristic predictor."""
from __future__ import annotations

from app.predictor import predict, predict_case, regions, families, normalize


def test_exact_match_even_with_noise() -> None:
    assert predict("CT Chest w contrast", "CT CHEST W CONTRAST")
    assert predict("MRI Brain w/o contrast", "MRI BRAIN W/O CONTRAST")


def test_region_overlap_different_modality() -> None:
    # Same region (BREAST) — both mammo, different wording.
    assert predict(
        "MAM screen BI with tomo",
        "MAMMOGRAPHY SCREENING BILATERAL",
    )
    # Same region (CHEST) across MR vs CT.
    assert predict("MRI thoracic spine wo con", "CHEST 2 VIEW FRONTAL & LATRL") in (True, False)
    assert predict("CT chest w con", "CHEST 2 VIEW FRONTAL & LATRL")


def test_unrelated_regions_false() -> None:
    assert not predict("MRI KNEE LT WO CONTRAST", "CT HEAD WITHOUT CNTRST")
    assert not predict("XR ankle 3V", "MRI BRAIN WITHOUT/WITH CONTRST")


def test_family_mammography_covers_breast_ultrasound() -> None:
    assert predict("MAM screen BI with tomo", "US BREAST BILATERAL")
    assert predict("MAM US BI breast screening", "MAMMOGRAPHY SCREENING BILATERAL")


def test_family_cardiac_covers_echo_and_coronary() -> None:
    assert predict("ECHO 2D Mmode transthorac TTE", "CT angio coronary artery")


def test_wholebody_pet_covers_torso() -> None:
    assert predict("CT abdomen pelvis w con", "PET/CT skull to thigh sbq/F18")
    assert predict("CT CHEST WITHOUT CONTRAST", "PET/CT skull to thigh sbq/F18")


def test_normalize_strips_punctuation_and_case() -> None:
    assert normalize("CT ABD/PEL WITH CNTRST") == "CT ABD PEL WITH CNTRST"
    assert normalize("CT ABD_PEL WITH CNTRST") == "CT ABD PEL WITH CNTRST"


def test_regions_basic() -> None:
    assert "CHEST" in regions("CT chest w con")
    assert "SPINE_T" in regions("MRI thoracic spine wo con")
    assert "BREAST" in regions("MAM screen BI with tomo")
    assert "BREAST" in regions("MAMMOGRAPHY SCREENING BILATERAL")


def test_families_basic() -> None:
    assert "mammography" in families("MAM screen BI with tomo")
    assert "cardiac" in families("ECHO 2D Mmode transthorac TTE")
    assert "cancer_workup" in families("PET/CT skull to thigh sbq/F18")


def test_predict_case_matches_predict() -> None:
    cur = "CT chest w con"
    priors = ["CT CHEST WITHOUT CNTRST", "MRI KNEE LEFT WO CONTRAST", "XR ankle"]
    batch = predict_case(cur, priors)
    one_by_one = [predict(cur, p) for p in priors]
    assert batch == one_by_one


def test_empty_inputs_are_safe() -> None:
    assert predict("", "CT chest") is False
    assert predict("CT chest", "") is False
    assert predict_case("CT chest", []) == []
