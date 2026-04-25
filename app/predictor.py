"""Heuristic predictor for the Relevant Priors challenge.

A prior examination is predicted relevant to a current examination when:

1. The two study descriptions normalize to the same string (exact match).
2. They share at least one body region.
3. They belong to the same "study family" (e.g. any mammography study with
   any other mammography study, any cardiac study with any other cardiac
   study, any PET whole-body study with any torso CT, etc.).

These rules are derived from inspection of the public split and hit 92+ %
accuracy with no model call. The predictor is pure and deterministic; all
I/O happens in the FastAPI layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Region patterns are matched case-insensitively against the uppercased
# description. Each pattern is deliberately permissive (prefix matches,
# common abbreviations, punctuation-friendly) because the real data mixes
# free-text styles.
REGION_PATTERNS: dict[str, str] = {
    # SKULL is intentionally excluded: almost every SKULL in the data is
    # "SKULL TO THIGH" (the PET/CT anatomical range descriptor), not a
    # skull/head imaging study.
    "BRAIN":    r"BRAIN|HEAD|INTRACRAN|PITUITARY|CEREB|ORBIT|FACE|FACIAL|MAXFACIAL|SINUS",
    "NECK":     r"\bNECK\b|THYROID|CAROTID|LARYN|PAROTID",
    "CHEST":    r"CHEST|THORAX|LUNG|\bRIBS?\b|STERNUM|MEDIASTIN",
    "ABDOMEN":  r"\bABDOM|\bABD\b|LIVER|HEPATIC|GALLBLAD|PANCREA|SPLEEN|\bGI\b|ENTEROGRAPHY|COLONOGRAPH",
    "PELVIS":   r"PELV|BLADDER|UTER|OVAR|\bGYN\b|PROSTATE|RECTUM|TRANSVAGINAL",
    "KIDNEY":   r"KIDNEY|RENAL|URETER|URINARY|UROGRAM",
    "AORTA":    r"AORTA|\bAAA\b",
    # CERVICAL, CERVICL (typo), CERV SPINE (abbreviation), and C-SPINE
    # variants all appear in the source data.
    "SPINE_C":  r"CERVIC|CERVICL|C-SPINE|\bC\s*SPINE\b|\bCERV\s+SPINE\b",
    "SPINE_T":  r"THORACIC\s*SPINE|T-SPINE|\bT\s*SPINE\b",
    "SPINE_L":  r"LUMBAR|L-SPINE|\bL\s*SPINE\b|SACRUM|LUMBOSACRAL|\bSI\s+JOINT\b|COCCYX",
    "BREAST":   r"BREAST|MAMMO|\bMAM\b",
    "HEART":    r"CARDIAC|CORONARY|\bECHO\b|\bHEART\b|MYO\s*PERF|MYOCARD|TRANSTHORAC|TRANSESOPH|TTE|TEE",
    "KNEE":     r"\bKNEE\b|PATELL",
    "SHOULDER": r"SHOULDER|CLAVICLE|SCAPULA|AC\s*JOINT",
    "HIP":      r"\bHIP\b|FEMUR|\bFEMORAL\b",
    "ANKLE":    r"\bANKLE\b",
    "FOOT":     r"\bFOOT\b|\bTOE\b|CALCANEUS|FOREFOOT",
    "HAND":     r"\bHAND\b|\bFINGER|\bWRIST\b|CARPAL",
    "ELBOW":    r"\bELBOW\b|HUMERUS|FOREARM",
    "LEG":      r"\bLEG\b|TIBIA|FIBULA|\bCALF\b",
    "DXA":      r"\bDXA\b|BONE\s*DENS|\bDEXA\b",
    "VENOUS":   r"VENOUS|\bVEIN\b|\bDVT\b",
    "WHOLEBODY": r"WHOLE\s*BODY|SKULL\s*[-_/ ]*THIGH|SKULLTHIGH|BODY\s+PET|F18",
}

# Modality families — used to tighten recall when region matching alone is
# ambiguous and to power the family rules below.
MODALITY_PATTERNS: dict[str, str] = {
    "MR":    r"\bMRI?\b|MAGNETIC",
    "CT":    r"\bCT\b|\bCAT\b|TOMOGRAPHY",
    "PET":   r"\bPET\b",
    "US":    r"\bUS\b|ULTRASOUND|SONO|DOPPLER",
    "XR":    r"\bXR\b|X-?RAY|\bRADIOGRAPH",
    "MAMMO": r"MAMMO|\bMAM\b",
    "NM":    r"\bNM\b|\bSPECT\b|NUCLEAR|BONE\s*SCAN|MYO\s*PERF",
    "DXA":   r"\bDXA\b|BONE\s*DENS|\bDEXA\b",
    "FLUORO": r"FLUORO|BARIUM|ESOPHAGRAM|SWALLOW",
    "ANGIO": r"ANGIO",
    "ECHO":  r"\bECHO\b|TTE|TEE|TRANSTHORAC|TRANSESOPH",
}

# Study "families" — descriptions in the same family are usually relevant to
# each other regardless of region overlap. Pattern matched against the
# uppercased description.
FAMILY_PATTERNS: dict[str, str] = {
    # STANDARD SCREENING COMBO is a mammography order alias in the
    # public split (32 / 32 True against any mammo current). DIGITAL
    # SCREENER and ULTRASOUND BILAT SCREEN are likewise screening flow
    # aliases that the original mammography pattern missed.
    "mammography":      r"MAMMO|\bMAM\b|BREAST|STANDARD\s+SCREENING\s+COMBO|DIGITAL\s+SCREENER|ULTRASOUND\s+BILAT\s+SCREEN",
    "cardiac":          r"CARDIAC|CORONARY|\bECHO\b|\bHEART\b|MYO\s*PERF|MYOCARD|TTE|TEE|TRANSTHORAC|TRANSESOPH|CALCIUM\s*SCORE|\bFFR\b",
    "cancer_workup":    r"PET|WHOLE\s*BODY|SKULL\s*[-_/ ]*THIGH|SKULLTHIGH|BONE\s*SCAN|F18|ONCOLOGY",
    "aortic_screen":    r"\bAAA\b|AORT",
    "dxa_bone_dens":    r"\bDXA\b|BONE\s*DENS|\bDEXA\b",
    "neuro_vascular":   r"CT\s*ANGIO.*HEAD|CT\s*ANGIO.*NECK|CT\s*ANGIO.*CAROTID|MRA\s*HEAD|MRA\s*NECK|CAROTID",
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^A-Z0-9 ]+")


def _upper(desc: str) -> str:
    return desc.upper() if desc else ""


def normalize(desc: str) -> str:
    """Normalize a study description for exact-equality comparison.

    Also used as the haystack for all pattern matching so that punctuation
    like `_` and `/` (which count as word characters in regex) do not break
    `\\b` boundaries (`ABD_PEL` vs `ABD PEL`).
    """
    up = _upper(desc)
    up = _PUNCT.sub(" ", up)
    up = _WHITESPACE.sub(" ", up).strip()
    return up


# Phrases that reliably mean "soft-tissue neck study" — not a brain study —
# even though they contain the word HEAD. Matched case-insensitively against
# the normalized description.
_HEAD_AND_NECK_RE = re.compile(r"HEAD\s+AND\s+NECK|HEAD\s+NECK|HEAD\s*[/&]\s*NECK")


def regions(desc: str) -> set[str]:
    up = normalize(desc)
    found = {name for name, pat in REGION_PATTERNS.items() if re.search(pat, up)}
    # "Head and neck" is a neck soft-tissue descriptor in this dataset; it
    # matches our BRAIN pattern via the word HEAD, but clinically it does
    # not image the brain. 265 of 270 such pairs in the public split are
    # labeled False — stripping BRAIN here turns nearly all into TNs.
    if "BRAIN" in found and _HEAD_AND_NECK_RE.search(up):
        found.discard("BRAIN")
        found.add("NECK")
    return found


def modalities(desc: str) -> set[str]:
    up = normalize(desc)
    return {name for name, pat in MODALITY_PATTERNS.items() if re.search(pat, up)}


def families(desc: str) -> set[str]:
    up = normalize(desc)
    return {name for name, pat in FAMILY_PATTERNS.items() if re.search(pat, up)}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Sig:
    norm: str
    regions: frozenset[str]
    modalities: frozenset[str]
    families: frozenset[str]


def _sig(desc: str) -> _Sig:
    return _Sig(
        norm=normalize(desc),
        regions=frozenset(regions(desc)),
        modalities=frozenset(modalities(desc)),
        families=frozenset(families(desc)),
    )


# Whole-body imaging (PET, bone scan, skull-to-thigh) sees anywhere from neck
# to thigh. When one side of the pair is whole-body, expand its regions so
# torso priors match.
_WHOLEBODY_COVERS = frozenset({
    "CHEST", "ABDOMEN", "PELVIS", "SPINE_T", "SPINE_L", "HIP", "AORTA", "NECK",
})


def _effective_regions(sig: _Sig) -> frozenset[str]:
    if "WHOLEBODY" in sig.regions:
        return sig.regions | _WHOLEBODY_COVERS
    return sig.regions


# Directed "bridge" rules for asymmetric clinical relationships. Each rule
# states: if the CURRENT study matches `cur_pat` and any PRIOR matches
# `pri_pat`, that prior is relevant. Rules are one-directional — add a
# reverse entry explicitly if the relationship holds both ways.
#
# Each was validated individually on the public split to confirm it raises
# true positives without creating more false positives.
_BRIDGE_RULES: list[tuple[str, str, str]] = [
    # Oncologic whole-body imaging (current) ↔ torso CT (prior). Narrow to CT
    # because MRI abdomen for non-oncologic reasons (liver lesion workup,
    # biliary) produced FPs when paired with bone scan.
    (
        "current_bone_or_pet__prior_ct_torso",
        r"\bBONE\s*SCAN\b|\bPET\b|SKULL\s*(?:-|TO|_|/|\s)*THIGH|SKULLTHIGH|\bF18\b|ONCOLOGY",
        r"\bCT\b.*(CHEST|ABDOMEN|\bABD\b|PELVIS|\bPEL\b)",
    ),
    # Reverse: torso CT (current) ↔ oncologic whole-body (prior).
    (
        "current_ct_torso__prior_bone_or_pet",
        r"\bCT\b.*(CHEST|ABDOMEN|\bABD\b|PELVIS|\bPEL\b)",
        r"\bBONE\s*SCAN\b|\bPET\b|SKULL\s*(?:-|TO|_|/|\s)*THIGH|SKULLTHIGH|\bF18\b|ONCOLOGY",
    ),
    # Mammography (current) ↔ bilateral-screening US (prior).
    (
        "current_mammo__prior_bilat_screen_us",
        r"MAMMO|\bMAM\b|BREAST",
        r"ULTRASOUND\s+BILAT\s+SCREEN|DIGITAL\s+SCREENER|COMBOHD",
    ),
    # Reverse: bilateral screening US (current) ↔ mammography (prior).
    (
        "current_bilat_screen_us__prior_mammo",
        r"ULTRASOUND\s+BILAT\s+SCREEN|DIGITAL\s+SCREENER|COMBOHD",
        # Prior widened to include the same screener aliases — in this
        # dataset bilateral-screening US and digital screener priors
        # represent the same mammography workup and should match each
        # other (3 FN on the public split otherwise).
        r"MAMMO|\bMAM\b|BREAST|DIGITAL\s+SCREENER|COMBOHD|ULTRASOUND\s+BILAT\s+SCREEN",
    ),
    # Spine adjacency — only keep the directions that are net-positive on the
    # public split. C↔T (current=C) is 57 % True and stays. L→T is 37 %, still
    # slightly positive. The reverse T→C direction was dropped: on 33 pairs
    # it was only 36 % True (12 TP vs 21 FP), driven by MRI T-spine pulling
    # in XR cervical / "CERVICL SPINE, LIMITED" priors that almost always
    # ended up labeled False. Same-modality T→C is too sparse to justify a
    # narrow re-add. C↔L and T↔L are also omitted for base-rate reasons.
    (
        "current_cspine__prior_tspine",
        r"\bCERVIC|CERVICL|\bC[-\s]?SPINE\b|\bCERV\s+SPINE\b",
        r"THORACIC\s*SPINE|\bT[-\s]?SPINE\b",
    ),
    (
        "current_lspine__prior_tspine",
        r"LUMBAR\s*SPINE|\bL[-\s]?SPINE\b",
        r"THORACIC\s*SPINE|\bT[-\s]?SPINE\b",
    ),
    # Plain-film T-spine (current) → C-spine (prior, any modality). Narrow
    # re-add of the T→C bridge limited to XR/plain-film T-spine currents —
    # these are 71 % True on the public split (10 TP / 4 FP on 14 pairs),
    # unlike MRI T-spine currents which were a coin flip (36 % True).
    (
        "current_xr_tspine__prior_cspine",
        r"XR\s+THORACIC|XR\s+T[-\s]?SPINE|^THORACIC\s+SPINE\s+\d+\s*V|^T[-\s]?SPINE\s+\d+\s*V",
        r"\bCERVIC|CERVICL|\bC[-\s]?SPINE\b|\bCERV\s+SPINE\b",
    ),
    # NOTE: head CT ↔ C-spine bridges were removed — on the public split both
    # directions hit ~10 % True (head CT→cspine 7/74 = 9.5 %, cspine→head CT
    # 11/86 = 12.8 %), so the bridge generated far more FPs than TPs despite
    # the clinical intuition of craniocervical-junction overlap. Radiologists
    # rarely read the head CT when evaluating a c-spine study (and vice versa)
    # unless the order is specifically a polytrauma survey, which is already
    # covered by region overlap when both modalities are cross-sectional.
    # Carotid US ↔ head CT.
    (
        "current_carotid_us__prior_headct",
        r"CAROTID\s*ULTRASOUND|\bUS\s+CAROTID\b|VAS\s+US\s+CAROTID",
        r"(?:CT|MRI?)\s+(?:HEAD|BRAIN)",
    ),
    # Cardiac workup (current) ↔ cross-sectional chest (prior). ONE-WAY
    # because a current CT chest + prior echo is frequently labeled False
    # (chest CT is ordered for pulmonary reasons, not to read the echo).
    (
        "current_cardiac__prior_ct_mri_chest",
        r"ECHO|\bTTE\b|\bTEE\b|TRANSTHORAC|TRANSESOPH|CORONARY|MYO\s*PERF|\bCT\s+FFR\b|CARDIAC",
        r"\b(?:CT|MRI?)\b.*(?:CHEST|THORAX|LUNG)|CT\s+ANGIO\s+CHEST",
    ),
    # Thoracic spine (current) ↔ chest with an explicit modality (prior).
    # We deliberately do NOT match bare "Chest" (a legacy descriptor in the
    # data, mostly labeled False). ONE-WAY — current XR chest + prior T-spine
    # was labeled False often enough to be unsafe.
    # T-spine (current) ↔ chest priors. Cross-sectional chest priors (CT/MRI)
    # pair with T-spine at 94 % True regardless of current modality. Plain
    # chest XR priors only pair reliably when the T-spine current is itself
    # a plain film — MRI T-spine → chest XR is a coin flip (45 % True, 13 TP
    # vs 16 FP on the public split), so we suppress the bridge there via
    # _BRIDGE_CURRENT_EXCLUSIONS below.
    (
        "current_tspine__prior_cross_section_chest",
        r"THORACIC\s*SPINE|\bT[-\s]?SPINE\b",
        r"CT\s+CHEST|MRI?\s+CHEST",
    ),
    (
        "current_tspine__prior_chest_xr",
        r"THORACIC\s*SPINE|\bT[-\s]?SPINE\b",
        r"CHEST\s+\d+\s*V|CHEST\s+1\s*V|\bCXR\b|XR\s+CHEST",
    ),
    # Cholangiogram (biliary XR) ↔ CT abdomen/pelvis (biliary tree imaged).
    (
        "current_cholangio__prior_ct_abdomen",
        r"CHOLANGIO|\bERCP\b",
        r"\bCT\b.*ABD",
    ),
    # Thoracentesis (CT guided) ↔ prior chest imaging. PARACENTES (peritoneal
    # tap — an abdominal procedure) was intentionally excluded: 0/6 True on
    # the public split when bridged to chest priors.
    (
        "current_thoracentesis__prior_chest",
        r"THORACENTES",
        r"\bCHEST\b|\bCT\b\s+CHEST|RIBS",
    ),
    # Esophagram (current) ↔ CT chest (prior).
    (
        "current_esophagram__prior_ct_chest",
        r"ESOPHAGRAM|ESOPHAG|BARIUM\s+SWALLOW|SWALLOW\s+STUDY",
        r"\bCT\b.*(CHEST|LUNG|MEDIASTIN)",
    ),
    # EEG / TCD / stroke workup (current) ↔ brain imaging (prior).
    (
        "current_eeg_tcd__prior_brain_imaging",
        r"\bEEG\b|TRANSCRANIAL\s*DOPPLER|\bTCD\b|STROKE\s*WORKUP",
        r"(?:CT|MRI?)\s+(?:HEAD|BRAIN)|CT\s+ANGIO\s+(?:HEAD|NECK|CAROTID)|CT\s+BRAIN\s+PERFUSION|\bMRA\b",
    ),
    # Lumbar spine (current) ↔ CT abd/pelvis (prior).
    (
        "current_lspine__prior_ct_abd_pel",
        r"LUMBAR\s*SPINE|\bL[-\s]?SPINE\b",
        r"\bCT\b\s+ABD(?:OMEN)?(?:\s+PEL\w*)?|\bCT\b\s+PEL\w*",
    ),
    # Reverse: CT abd/pel (current) ↔ pelvic MRI/US (prior).
    (
        "current_ct_abd_pel__prior_mri_pelvis",
        r"\bCT\b\s+ABD(?:OMEN)?.*PEL\w*",
        r"MRI?\s+PELVIS|MR\s+PELVIS",
    ),
    # CT-guided biopsy / FNA (current) ↔ prior CT chest (biopsy site often
    # in thorax; lung nodule biopsy is common). ONE-WAY, and narrow to CT
    # only to avoid CXR / MRI FPs. Excludes biopsies that already name a
    # non-chest target (abdomen/pelvis/breast) to avoid treating a chest
    # CT as the "biopsy reference" for an abdominal biopsy.
    (
        "current_unlocalized_biopsy__prior_ct_chest",
        r"(?:^|\s)(?:CT\s+GUIDED|\bFNA\b|\bBIOPSY\b|ASPIRATION)(?![^$]*(?:ABD|PELV|BREAST|LIVER|KIDNEY|THYROID|BONE))",
        r"\bCT\b.*(CHEST|LUNG)",
    ),
    # Pulmonary nuc med perfusion (current) ↔ chest XR (prior) — V/Q scans
    # are always read against a chest radiograph.
    (
        "current_pulm_vq__prior_chest_xray",
        r"PUL\s*PERFUSION|\bV[/\s]*Q\b|PULMONARY\s*PERFUSION|VENT.*PERFUS",
        r"\bCHEST\b|\bCXR\b",
    ),
    # Scoliosis / spine survey XR (current) ↔ chest XR OR CT spine (prior).
    (
        "current_spine_survey__prior_chest_or_spine_ct",
        r"SCOLIOSIS|SPINE\s+SURV",
        r"\bCHEST\b|CT\s+LUMBAR\s*SPINE|CT\s+CERVIC",
    ),
    # (A second cspine ↔ head CT bridge was removed above — same reason.)
    # NOTE: neurovascular bridge (CT angio carotid → CT head) was dropped.
    # On the public split it drifted to 44 % True (12 TP / 15 FP on 27 pairs)
    # after other rules reclaimed some of the same TPs via region/family
    # overlap. Dropping converts a net-negative bridge into a small win.
    # Pelvic/endovaginal US (any variant) ↔ CT abd/pel.
    (
        "current_pelvic_us__prior_ct_abd_pel",
        r"US\s+PELV\w*|ENDOVAGIN|TRANSVAG",
        r"\bCT\b\s+ABD(?:OMEN)?\s+PEL\w*",
    ),
    (
        "current_ct_abd_pel__prior_pelvic_us",
        r"\bCT\b\s+ABD(?:OMEN)?\s+PEL\w*",
        r"US\s+PELV\w*|ENDOVAGIN|TRANSVAG",
    ),
    # Interventional / IR procedures in abdomen (drainage, nephrostomy,
    # ablation, paracentesis) pair with prior CT abdomen/pelvis — the CT
    # is the planning/follow-up study.
    (
        "current_ct_abdpel__prior_abd_ir_procedure",
        r"\bCT\b\s+ABD(?:OMEN)?",
        r"DRAIN(?:AGE)?|NEPHROSTOM|PARACENTES|ABLATION|\bIR\b\s+(?:NEPHROSTOM|DRAIN|ABLATION)",
    ),
    # Esophagram (XR) ↔ chest XR (prior) — esophagus is the substrate
    # visualized; comparison chest XR is frequently requested.
    (
        "current_esophagram__prior_chest_xray",
        r"ESOPHAGRAM|ESOPHAG|BARIUM\s+SWALLOW|SWALLOW\s+STUDY",
        r"\bXR\s+CHEST\b|CHEST\s+\d+\s*V|CHEST\s+1\s*V|\bCXR\b",
    ),
    # MR pelvis (current) → MR lumbar spine (prior). Pelvic MRI often images
    # the lower spine secondarily and the radiologist pulls the prior L-spine
    # MR for correlation. 78 % True on the public split (7 TP / 2 FP on 9
    # pairs). One-way only: the reverse (L-spine → pelvis) was 14 % True.
    (
        "current_mri_pelvis__prior_mri_lspine",
        r"MRI?\s+PELVIS|MR\s+PELVIS",
        r"MRI?\s+LUMBAR|MR\s+LUMBAR|MRI?\s+L[-\s]?SPINE",
    ),
    # Kidney / renal US ↔ US abdominal. 100 % True both ways (7 pairs total
    # on the public split) — renal ultrasound IS substantially overlapping
    # with a complete abdominal ultrasound window.
    (
        "current_kidney_us__prior_us_abd",
        r"US\s+KIDNEY|US\s+RENAL|KIDNEYS?\s+AND\s+BLADDER|RENAL\s+ULTRASOUND",
        r"US\s+ABDOM",
    ),
    (
        "current_us_abd__prior_kidney_us",
        r"US\s+ABDOM",
        r"US\s+KIDNEY|US\s+RENAL|KIDNEYS?\s+AND\s+BLADDER|RENAL\s+ULTRASOUND",
    ),
    # CT cervical spine (current) → CT head/brain (prior). 91 % True on the
    # public split (10 TP / 1 FP on 11 pairs) — trauma c-spine CTs are
    # routinely read with the head CT. CT-modality only: the MRI variant was
    # 0 % True, and the head→cspine reverse direction was 44 %.
    (
        "current_ct_cspine__prior_ct_head",
        r"\bCT\b\s+CERVIC|\bCT\b\s+C[-\s]?SPINE",
        r"\bCT\b\s+(?:HEAD|BRAIN)",
    ),
    # CT chest (current) → CT coronary studies (prior). Includes calcium
    # scoring (7 / 7 True) and CT coronary artery / CT angio coronary
    # (5 / 5 True) on the public split. Contrast chest CT and coronary
    # CT are routinely cross-read in the cardiac workflow.
    (
        "current_ct_chest__prior_coronary_calc",
        r"\bCT\b\s+CHEST|MRI\s+CHEST",
        r"CT\s+CORONARY\s+CALC|CORONARY\s+CALC\s+SCREEN|CALCIUM\s+SCORE|"
        r"CT\s+ANGIO\s+CORONARY|CT\s+CORONARY\s+ARTERY|CT\s+CORONARY\b",
    ),
    # CT coronary calcium score (current) → chest XR (prior). 5 / 8 True
    # (62 %) on the public split. Calcium scoring is a cardiac workup
    # and a prior chest XR is routinely reviewed. One-way — chest XR
    # currents against a CT coronary prior are 39 % True (net-negative).
    (
        "current_ct_coronary_calc__prior_chest_xr",
        r"CT\s+CORONARY\s+CALC|CORONARY\s+CALC\s+SCREEN|CALCIUM\s+SCORE",
        r"CHEST\s+\d+\s*V|CHEST\s+1\s*V|\bCXR\b|XR\s+CHEST",
    ),
    # Paracentesis (current, peritoneal tap — abdominal procedure) → any
    # prior abdominal imaging. 9 / 9 True on the public split (CT abd, CT
    # renal colic, US abdomen). PARACENTESIS itself has no region/family
    # token, so the pair previously fell through all positive rules.
    (
        "current_paracentesis__prior_abd_imaging",
        r"PARACENTES",
        r"\bCT\b\s+ABD|\bCT\b\s+RENAL\s+COLIC|MRI?\s+ABD|US\s+ABDOM",
    ),
    # Breast seed localization (current, pre-op wire/seed placement) →
    # mammography / breast ultrasound prior. 9 / 10 True on the public
    # split. Seed localization descriptions (e.g. "Seed Localization US
    # Right") don't include MAM/BREAST tokens themselves, so they miss
    # the mammography family without an explicit bridge.
    (
        "current_seed_localization__prior_breast_us_mammo",
        r"SEED\s+LOCALIZ|LOCALIZ.*BREAST",
        r"MAM\s+US|US\s+BREAST|MAMMO|MAM\s+BI|MAM\s+SCREEN",
    ),
    # CT abdomen+pelvis WITH contrast (current) → small bowel series
    # (prior). 4 / 5 True on the public split. Contrast CT abdpel for GI
    # workups is read against the prior fluoroscopic SBS — the latter
    # has no ABDOMEN-region token of its own (no `\bABD`, no GI alias).
    (
        "current_ct_abdpel_w_con__prior_sbs",
        r"\bCT\b\s+ABD(?:OMEN)?\s+(?:AND\s+)?PEL\w*\s+W\s+CON|"
        r"\bCT\b\s+ABD(?:OMEN)?\s+(?:AND\s+)?PEL\w*\s+WITH\s+CON",
        r"SMALL\s+BOWEL\s+SERIES|SMALL\s+BOWEL\s+FOLLOW",
    ),
    # US pelvic (current) → US endovaginal (prior). 7 / 9 True on the
    # public split. ENDOVAGINAL is not in the PELVIS region pattern
    # (TRANSVAGINAL is), so the same-organ pair was missing the region
    # rule. Adding ENDOVAGINAL to PELVIS would change behavior for many
    # unrelated pairs (102 priors involving US ENDOVAGINAL, only 10 %
    # True overall) — a narrow bridge is safer.
    (
        "current_us_pelvic__prior_us_endovag",
        r"US\s+PELV|US\s+PELVIC",
        r"US\s+ENDOVAG|^ENDOVAG",
    ),
    # MRI neck (current) → PET head-neck-body / PET head-and-neck (prior).
    # 2 / 2 True on the public split. PET head/neck oncology workups
    # often follow a baseline neck MR. The PET prior format
    # "PET^PET_CT_HEADNECK_BODY" lacks a recognizable WHOLEBODY token
    # under the existing pattern, so the pair was missing both region
    # and family overlap.
    (
        "current_mri_neck__prior_pet_headneck",
        r"MRI?\s+NECK|MRI\s+NECK\s+W",
        r"HEADNECK|HEAD\s*[/_-]?\s*NECK\s*BODY|PET.*HEAD.*NECK",
    ),
    # Plain chest XR (current) → MR cardiac (prior). 2 / 2 True. MR
    # cardiac is read alongside a baseline chest film for size /
    # silhouette context. Currently neither side carries the cardiac
    # family token (the pattern requires CARDIAC|CORONARY|TTE|MYO_PERF),
    # so a narrow bridge fixes the pair.
    (
        "current_chest_xr__prior_mr_cardiac",
        r"^CHEST\s+\d|XR\s+CHEST|\bCXR\b|CHEST\s+FRONTAL",
        r"MRI?\s+CARDIAC|MR\s+CARDIAC",
    ),
]

_BRIDGE_COMPILED: list[tuple[str, re.Pattern[str], re.Pattern[str]]] = [
    (name, re.compile(cur_pat), re.compile(pri_pat))
    for (name, cur_pat, pri_pat) in _BRIDGE_RULES
]


# Per-bridge "skip if present in current" context tokens. If the current
# description contains any of these, the matching bridge is suppressed.
# Keyed by bridge name. Used for clinical contexts where the standard
# relationship does not hold (e.g. cardiotoxicity surveillance echoes are
# not actually asking about a prior chest CT, which was done for cancer).
_BRIDGE_CURRENT_EXCLUSIONS: dict[str, re.Pattern[str]] = {
    # CHEMO / LUM TTE / DEFINITY bubble studies are ordered for reasons
    # (cardiotoxicity surveillance, contrast-enhanced echo for LV function)
    # that don't actually make a prior CT chest clinically relevant. On the
    # public split DEFINITY-echo current → CT chest prior is 0 % True (5 FP
    # out of 5 pairs) and CHEMO-echo is ~31 %.
    "current_cardiac__prior_ct_mri_chest": re.compile(r"CHEMO|\bLUM\b|DEFINITY"),
    # MRI T-spine → chest XR is 45 % True on the public split (near base
    # rate). Suppress the bridge when the current is an MRI T-spine, keeping
    # it live for XR / CT T-spine currents.
    "current_tspine__prior_chest_xr": re.compile(r"\bMRI?\b|MAGNETIC"),
}


# Pair-level anti-rules: when the current matches `cur_pat` AND the prior
# matches `pri_pat`, force the prediction to False regardless of any other
# rule that might have fired. Used for cross-rule false positives we can't
# narrow out by tightening individual rules (e.g. family-match FPs where
# the family pattern legitimately applies elsewhere).
_NEGATIVE_PAIR_RULES: list[tuple[str, str, str]] = [
    # Pure CT angio head (no NECK/CAROTID in current) vs carotid US prior:
    # matched via the neuro_vascular family on the CAROTID alias, but 0 %
    # True on the public split (7 FP). Intracranial CTA doesn't actually
    # comment on the carotid US.
    (
        "ctangio_head_vs_carotid_us",
        r"^CT\s+ANGIO\s+HEAD(?!\s*(?:AND\s+)?(?:NECK|CAROTID))",
        r"CAROTID\s*ULTRASOUND|VAS\s+US\s+CAROTID|\bUS\s+CAROTID\b",
    ),
    # US abdomen (current) vs bare "Abdomen" prior: 0 % True on public split
    # (7 FP). US abdomen complete with doppler is a specific vascular/organ
    # study; a generic "Abdomen" prior of unknown modality is not a match.
    (
        "us_abdomen_vs_bare_abdomen",
        r"\bUS\s+ABDOM|ULTRASOUND\s+ABDOM|ABDOM.*DOPPL",
        r"^ABDOMEN$",
    ),
    # CT abd/pel WITHOUT contrast (current) vs pelvic / endovaginal US
    # (prior): 0 % True (4 FP) — CT abdpel without contrast is ordered for
    # stones/trauma, not gyn workup.
    (
        "ct_abdpel_wo_contrast_vs_pelvic_us",
        r"\bCT\b\s+ABD(?:OMEN)?\s+PEL\w*\s+W[O/]?\s*CON|"
        r"\bCT\b\s+ABD(?:OMEN)?\s+PEL\w*\s+WITHOUT",
        r"US\s+PELV|ENDOVAGIN|TRANSVAG",
    ),
    # Opposite-side laterality on a shared paired body part (handled by
    # _is_laterality_mismatch below — covers MAM/BREAST, KNEE, HIP,
    # SHOULDER, FOOT, ANKLE, WRIST, HAND, ELBOW, LEG at once).
    # CT abd/pel with contrast (current) vs bare "Abdomen" prior: 0 / 5
    # True on the public split. "w con" CT abdpels are usually ordered for
    # specific pathology workups where the unmodified "Abdomen" prior
    # (typically a plain film or unlabeled) is not the comparator.
    (
        "ct_abdpel_w_con_vs_bare_abdomen",
        r"\bCT\b\s+ABD(?:OMEN)?.*\s+W\s+CON|\bCT\b\s+ABD(?:OMEN)?.*WITH\s+CON",
        r"^ABDOMEN$",
    ),
    # CT abd/pel (current) vs plain-film pelvis XR (prior, "PELVIS 1 VIEW",
    # "PELVIS - 1 OR 2 VIEWS"): 0 % True (6 FP) on the public split. The
    # plain-film priors are pre-op / fall-workup orders unrelated to the
    # cross-sectional abdomen study.
    (
        "ct_abdpel_vs_pelvis_xr",
        r"\bCT\b\s+ABD(?:OMEN)?.*PEL",
        r"^PELVIS\s+-?\s*\d|^PELVIS\s+\d+\s*V",
    ),
    # Plain chest XR (current) vs whole-body bone scan (prior): 0 % True
    # (7 FP). The WHOLEBODY region expansion pulls bone-scan priors in
    # against any chest prior, but a plain 1V/2V chest film is not the
    # comparator for a bone scan.
    (
        "plain_chest_xr_vs_wb_bone_scan",
        r"^(?:XR\s+)?CHEST\s+\d+\s*V|^CHEST\s+1\s*V|^CHEST\s+2\s*V|CHEST\s+FRONTAL",
        r"BONE\s*SCAN\s+WHOLE\s*BODY|NM\s+BONE\s+SCAN\s+WHOLE",
    ),
    # Head / brain imaging (current) vs sinus / maxfacial / orbit (prior):
    # 17 % True (2/12) on the public split. The BRAIN region pattern
    # matches both brain and facial studies, but a CT/MRI brain is
    # ordered for parenchymal pathology, not to review an old sinus scan.
    # Reverse direction (sinus → brain) was 69 % True and is left intact.
    (
        "brain_cur_vs_sinus_maxfacial_prior",
        r"(?:CT|MRI?)\s+(?:HEAD|BRAIN)|\bHEAD\s*/\s*BRAIN\b",
        r"\bMAXFACIAL\b|MAX\s*FACIAL|\bSINUS|FACIAL\s+BONES?|\bORBIT\b",
    ),
    # DXA hip scan (current) vs unilateral hip XR (prior, "HIP, RIGHT"):
    # 0 / 5 True on the public split. HIP region overlap fires because
    # the DXA description includes "HIP", but a unilateral hip fracture /
    # ortho XR is not a relevant comparator to a bone-density study.
    (
        "dxa_hip_cur_vs_unilateral_hip_xr",
        r"DXA.*HIP|BONE\s*DENS.*HIP|\bDEXA\b.*HIP",
        r"\bHIP\b.*\b(?:RT|RIGHT|LT|LFT|LEFT)\b|"
        r"\b(?:RT|RIGHT|LT|LFT|LEFT)\b.*\bHIP\b",
    ),
    # LUM TTE (specialty contrast-enhanced echo) vs nuclear myocardial
    # perfusion: same cardiac family, but 0 / 4 True — these are ordered
    # for distinct reasons (LV function vs. ischemia) and the radiologist
    # does not cross-read them.
    (
        "lum_tte_cur_vs_myo_perf_prior",
        r"LUM\s+TTE|LUM.*DOPPL",
        r"MYO\s*PERF|\bSPECT\b",
    ),
    # CT FFR (fractional flow reserve, a functional-anatomy coronary CT)
    # vs prior echo: both cardiac family, but 0 / 4 True. FFR is read as
    # a standalone functional study, not against a baseline echo.
    (
        "ct_ffr_cur_vs_echo_prior",
        r"CT\s+FFR|\bFFR\b",
        r"ECHO|\bTTE\b|\bTEE\b|TRANSTHORAC",
    ),
    # US head-and-neck soft tissue (current) vs thyroid US / soft-tissue
    # neck US (prior): 0 / 5 True. The HEAD-AND-NECK post-filter maps cur
    # to NECK region, which overlaps with the thyroid prior, but a generic
    # "soft tissue neck" US prior is not the comparator for a specific
    # head-and-neck US workup.
    (
        "us_head_neck_cur_vs_thyroid_us_prior",
        r"US\s+HEAD\s+AND\s+NECK|US\s+SOFT\s+TISSUE\s+NECK|HEAD\s+AND\s+NECK.*US",
        r"\bTHYROID\b|US\s+THYROID",
    ),
    # Transesophageal echo (current) vs chest XR (prior): 1 / 9 True.
    # TEE is a cardiac study read from the esophagus — a prior chest
    # radiograph is not part of that workup. Matched via the cardiac
    # bridge→chest XR loop (via T-spine expansion or CARDIAC family).
    (
        "tee_cur_vs_chest_xr_prior",
        r"ECHO\s+TRANSESOPH|\bTEE\b",
        r"CHEST\s+\d+\s*V|CHEST\s+1\s*V|\bCXR\b|XR\s+CHEST",
    ),
    # Mammography (current) vs lymphoscintogram (prior): 0 / 2 True. Both
    # match the mammography family (`BREAST`), but a lymphoscintogram is
    # a nuclear medicine sentinel-node injection — not a comparator for
    # diagnostic mammography.
    (
        "mam_cur_vs_lymphoscintogram_prior",
        r"MAM\s+SCREEN|MAM\s+DIAG|MAMMO|MAM\s+BI",
        r"LYMPHOSCINTOGRAM|LYMPHOSCINT|LYMPHO\s*SCAN",
    ),
    # CT angio chest (current) vs the legacy `CHEST N VIEW` plain-film
    # prior format: 0 / 8 True on the public split. Notably the
    # whitespace-free variants ("XR Chest 1V Frontal Only", "CHEST
    # FRONTAL") run 88 % True against the same current, so the tightly-
    # spaced "CHEST 1 V" / "CHEST 2 VIEW" form specifically marks the
    # legacy descriptor that doesn't pair with a vascular CTA workup.
    (
        "cta_chest_cur_vs_legacy_chest_xr_prior",
        r"CT\s+ANGIO\s+CHEST|CTA\s+CHEST",
        r"^CHEST\s+\d+\s+V|^CHEST\s+\d+\s+VIEW",
    ),
    # NM myocardial perfusion (current) vs CT angiogram chest (prior):
    # 1 / 4 True. Both share the cardiac family, but NM perfusion is a
    # functional ischemia study that doesn't actually cross-read with a
    # PE-protocol vascular CT.
    (
        "myo_perf_cur_vs_cta_chest_prior",
        r"MYO\s*PERF|NMMYO|\bSPECT\b",
        r"CT\s+ANGIOGRAM\s*,?\s*CHEST|CT\s+ANGIO\s+CHEST",
    ),
]

_NEGATIVE_PAIR_COMPILED: list[tuple[str, re.Pattern[str], re.Pattern[str]]] = [
    (name, re.compile(cur_pat), re.compile(pri_pat))
    for (name, cur_pat, pri_pat) in _NEGATIVE_PAIR_RULES
]


def _negative_pair_hit(cur_norm: str, pri_norm: str) -> bool:
    """Return True when a pair is covered by an anti-rule (forced False)."""
    for _name, cur_re, pri_re in _NEGATIVE_PAIR_COMPILED:
        if cur_re.search(cur_norm) and pri_re.search(pri_norm):
            return True
    return False


# Laterality markers used for opposite-side mismatch detection.
_LATERALITY_RIGHT = re.compile(r"\b(?:RT|RIGHT)\b")
_LATERALITY_LEFT = re.compile(r"\b(?:LT|LFT|LEFT)\b")

# Regions where opposite-side (R vs L) priors are almost never mutually
# relevant on the public split. Mammography / breast US opposite sides hit
# 1/56 True (≈2 %); extremity opposite sides (knee/hip/shoulder/ankle/foot/
# wrist/hand/elbow/leg) are 0/14 True cumulatively. Midline / axial regions
# (CHEST, ABDOMEN, SPINE_*, HEART, BRAIN) are excluded — a "left chest wall"
# prior against a "right chest wall" current is still the same CT chest
# dataset and IS clinically relevant.
_LATERALITY_PAIRED_REGIONS: frozenset[str] = frozenset({
    "BREAST", "KNEE", "HIP", "SHOULDER",
    "ANKLE", "FOOT", "HAND", "ELBOW", "LEG",
})


def _side(norm: str) -> str | None:
    """Return 'R', 'L', or None based on laterality tokens in `norm`.

    Descriptions mentioning both sides (e.g. bilateral) or neither return
    None and are treated as unknown — they do NOT participate in mismatch
    blocking.
    """
    has_r = bool(_LATERALITY_RIGHT.search(norm))
    has_l = bool(_LATERALITY_LEFT.search(norm))
    if has_r and not has_l:
        return "R"
    if has_l and not has_r:
        return "L"
    return None


def _is_laterality_mismatch(
    cur_norm: str, cur_regions: frozenset[str],
    pri_norm: str, pri_regions: frozenset[str],
) -> bool:
    """True if cur/prior are on opposite sides of the same paired region."""
    cs = _side(cur_norm)
    ps = _side(pri_norm)
    if cs is None or ps is None or cs == ps:
        return False
    return bool(cur_regions & pri_regions & _LATERALITY_PAIRED_REGIONS)


def _bridge_hit(cur_norm: str, pri_norm: str) -> bool:
    """Return True if any directional bridge links this (current, prior) pair."""
    for name, cur_re, pri_re in _BRIDGE_COMPILED:
        if cur_re.search(cur_norm) and pri_re.search(pri_norm):
            ex = _BRIDGE_CURRENT_EXCLUSIONS.get(name)
            if ex is not None and ex.search(cur_norm):
                continue
            return True
    return False


def predict(current_desc: str, prior_desc: str) -> bool:
    """Return True if the prior is predicted relevant to the current study."""
    if not current_desc or not prior_desc:
        return False

    cur = _sig(current_desc)
    pri = _sig(prior_desc)

    # Rule 0: uninformative bare prior descriptors short-circuit to False.
    if pri.norm in _UNINFORMATIVE_BARE_PRIOR:
        return False

    # Rule 0b: pair-level anti-rules override everything below.
    if _negative_pair_hit(cur.norm, pri.norm):
        return False

    # Rule 0c: opposite-side laterality on a shared paired region → False.
    if _is_laterality_mismatch(cur.norm, cur.regions, pri.norm, pri.regions):
        return False

    # Rule 1: exact normalized match.
    if cur.norm and cur.norm == pri.norm:
        return True

    # Rule 2: region overlap (with whole-body expansion).
    if _effective_regions(cur) & _effective_regions(pri):
        return True

    # Rule 3: shared study family (mammography, cardiac, cancer workup, ...).
    if cur.families & pri.families:
        return True

    # Rule 4: directed clinical bridge rules.
    if _bridge_hit(cur.norm, pri.norm):
        return True

    return False


# Bare legacy descriptors that normalize to a single word but aren't specific
# enough to mean "this study overlaps with the current one". On the public
# split, prior == "PELVIC" was only 5 % True (n=161) — blocking it cleanly
# converts FPs to TNs without meaningful TP loss. "ABDOMEN" is a coin flip
# on this split (~12 %), which is wash for accuracy, so we do not block it
# to avoid overfitting.
_UNINFORMATIVE_BARE_PRIOR: frozenset[str] = frozenset({"PELVIC"})


def predict_case(current_desc: str, prior_descs: list[str]) -> list[bool]:
    """Predict relevance for every prior in a single case.

    Pre-computes the current-study signature once, then applies the
    exact-match / region / family / bridge rules to each prior.
    """
    if not prior_descs:
        return []
    cur = _sig(current_desc or "")
    cur_eff = _effective_regions(cur)

    out: list[bool] = []
    for p in prior_descs:
        if not p:
            out.append(False)
            continue
        pri = _sig(p)
        if pri.norm in _UNINFORMATIVE_BARE_PRIOR:
            out.append(False)
            continue
        if _negative_pair_hit(cur.norm, pri.norm):
            out.append(False)
            continue
        if _is_laterality_mismatch(cur.norm, cur.regions, pri.norm, pri.regions):
            out.append(False)
            continue
        if cur.norm and cur.norm == pri.norm:
            out.append(True)
        elif cur_eff & _effective_regions(pri):
            out.append(True)
        elif cur.families & pri.families:
            out.append(True)
        elif _bridge_hit(cur.norm, pri.norm):
            out.append(True)
        else:
            out.append(False)
    return out
