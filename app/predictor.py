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
    "mammography":      r"MAMMO|\bMAM\b|BREAST",
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
        r"MAMMO|\bMAM\b|BREAST",
    ),
    # Spine adjacency (only between neighbouring levels — C↔T and T↔L):
    # radiologists commonly read adjacent-level spine priors because
    # pathology crosses levels. C↔L is NOT included (too far apart — high FP).
    (
        "current_cspine__prior_tspine",
        r"\bCERVIC|CERVICL|\bC[-\s]?SPINE\b|\bCERV\s+SPINE\b",
        r"THORACIC\s*SPINE|\bT[-\s]?SPINE\b",
    ),
    (
        "current_tspine__prior_cspine",
        r"THORACIC\s*SPINE|\bT[-\s]?SPINE\b",
        r"\bCERVIC|CERVICL|\bC[-\s]?SPINE\b|\bCERV\s+SPINE\b",
    ),
    (
        "current_lspine__prior_tspine",
        r"LUMBAR\s*SPINE|\bL[-\s]?SPINE\b",
        r"THORACIC\s*SPINE|\bT[-\s]?SPINE\b",
    ),
    # T↔L current_tspine__prior_lspine is omitted: label rate 27 % on the
    # public split, only marginally above the 24 % base rate, and adds more
    # FPs than TPs.
    # C-spine / head CT overlap — craniocervical junction imaging.
    (
        "current_headct__prior_cspine",
        r"(?:CT|MRI?)\s+(?:HEAD|BRAIN)",
        r"CERVIC|CERVICL|\bC\s*SPINE\b",
    ),
    (
        "current_cspine__prior_headct",
        r"CERVIC|CERVICL|\bC\s*SPINE\b",
        r"(?:CT|MRI?)\s+(?:HEAD|BRAIN)",
    ),
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
    (
        "current_tspine__prior_chest_with_modality",
        r"THORACIC\s*SPINE|\bT[-\s]?SPINE\b",
        r"CHEST\s+\d+\s*V|CHEST\s+1\s*V|\bCXR\b|CT\s+CHEST|MRI?\s+CHEST|XR\s+CHEST|RIBS",
    ),
    # Cholangiogram (biliary XR) ↔ CT abdomen/pelvis (biliary tree imaged).
    (
        "current_cholangio__prior_ct_abdomen",
        r"CHOLANGIO|\bERCP\b",
        r"\bCT\b.*ABD",
    ),
    # Thoracentesis (CT guided) ↔ prior chest imaging.
    (
        "current_thoracentesis__prior_chest",
        r"THORACENTES|PARACENTES",
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
    # Cervical spine (current) ↔ CT head/brain (prior) — cervical fractures
    # commonly evaluated with head CT overlap.
    (
        "current_cspine__prior_head_ct",
        r"CERVICAL\s*SPINE|\bC[-\s]?SPINE\b|CERVICL\s*SPINE",
        r"(?:CT|MRI?)\s+(?:HEAD|BRAIN)",
    ),
    # Neurovascular crosswalk: CT angio carotid ↔ CT head (CT only, not MRI).
    # Public-split stats: carotid angio ↔ MRI brain is 10% True, while ↔
    # CT head is ~50% True. Reverse direction (head CT → prior carotid
    # angio) is only 34% True so we do not bridge it.
    (
        "current_carotid_angio__prior_ct_head",
        r"ANGIO\s+CAROTID|CT\s+ANGIO\s+NECK",
        r"\bCT\b\s+(?:HEAD|BRAIN)|CT\s+BRAIN\s+PERFUSION",
    ),
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
    "current_cardiac__prior_ct_mri_chest": re.compile(r"CHEMO|\bLUM\b"),
}


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
