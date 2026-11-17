"""Phenotype vocabularies and label normalization.

Every mapping in this module is a *string normalization* of BacDive's controlled
vocabulary onto the MSS V1 categories named in the project specification. No taxon is
assigned a phenotype here. Raw strings that are not listed are kept as "unmapped" (never
guessed); users can extend or override the oxygen mapping with ``--oxygen-map``.
"""
from __future__ import annotations

import csv
from pathlib import Path

# MSS V1 oxygen-tolerance categories (order fixes the column order of every output).
OXYGEN_CLASSES = ["anaerobe", "facultative", "aerobe", "microaerophile", "aerotolerant"]

# BacDive raw oxygen-tolerance string (lower-cased, whitespace-collapsed) -> MSS category.
# Only strings that are lexically the specification's categories, or those categories with an
# explicit obligate/strict qualifier, are mapped. Everything else (e.g. "facultative aerobe",
# "microaerotolerant") is reported as unmapped so the user decides, rather than the code.
DEFAULT_OXYGEN_MAP = {
    "anaerobe": "anaerobe",
    "obligate anaerobe": "anaerobe",
    "strict anaerobe": "anaerobe",
    "strictly anaerobe": "anaerobe",
    "aerobe": "aerobe",
    "obligate aerobe": "aerobe",
    "strict aerobe": "aerobe",
    "strictly aerobe": "aerobe",
    "facultative anaerobe": "facultative",
    "microaerophile": "microaerophile",
    "aerotolerant": "aerotolerant",
    "aerotolerant anaerobe": "aerotolerant",
}

# Binary / categorical traits: BacDive value -> MSS label. Values not listed are unmapped.
BINARY_MAP = {"yes": "yes", "no": "no", "positive": "yes", "negative": "no", "+": "yes", "-": "no"}
GRAM_MAP = {"positive": "positive", "negative": "negative", "variable": "variable"}

# Trainable / aggregatable categorical traits. ``None`` = open vocabulary: the categories are
# BacDive's own strings (lower-cased), e.g. cell shape "rod-shaped", "coccus-shaped"; MSS does
# not merge or rename them.
TASK_LABELS = {
    "oxygen": OXYGEN_CLASSES,
    "motility": ["yes", "no"],
    "sporulation": ["yes", "no"],
    "gram_stain": ["positive", "negative", "variable"],
    "cell_shape": None,
    "ph_range": None,
}
CATEGORICAL_TRAITS = list(TASK_LABELS)

# Numeric traits: BacDive measurements, summarized as abundance-weighted means (no binning).
NUMERIC_TRAITS = [
    "optimal_temperature_c", "growth_temperature_min_c", "growth_temperature_max_c",
    "optimal_ph", "growth_ph_min", "growth_ph_max",
    "optimal_nacl_pct", "growth_nacl_max_pct",
]


def valid_label(task: str, value) -> bool:
    if is_missing(value):
        return False
    allowed = TASK_LABELS.get(task)
    return True if allowed is None else norm_text(value) in allowed

MISSING = {"", "na", "nan", "none", "null", "unknown"}


def norm_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def is_missing(value) -> bool:
    return norm_text(value) in MISSING


def oxygen_strictness(raw: str) -> str:
    """Return 'obligate' only when the raw string states it explicitly, else 'unspecified'."""
    t = norm_text(raw)
    return "obligate" if ("obligate" in t or "strict" in t) else "unspecified"


def load_oxygen_map(path: str | Path | None) -> dict[str, str]:
    """Default map, optionally updated from a two-column TSV ``raw<TAB>category``.

    A category of ``unmapped`` removes a default entry.
    """
    mapping = dict(DEFAULT_OXYGEN_MAP)
    if path is None:
        return mapping
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            if not row or row[0].startswith("#"):
                continue
            if i == 0 and norm_text(row[0]) in {"raw", "oxygen_raw"}:
                continue
            if len(row) < 2:
                raise ValueError(f"{path}: line {i + 1} needs two tab-separated columns")
            raw, cat = norm_text(row[0]), norm_text(row[1])
            if cat == "unmapped":
                mapping.pop(raw, None)
            elif cat in OXYGEN_CLASSES:
                mapping[raw] = cat
            else:
                raise ValueError(f"{path}: line {i + 1}: category {cat!r} not in {OXYGEN_CLASSES}")
    return mapping


def normalize_species(name) -> str:
    """Normalized exact-match key for species names.

    Strips a MetaPhlAn/GTDB ``s__`` rank prefix, converts underscores between words to spaces,
    lower-cases and collapses whitespace. GTDB suffixes such as ``Blautia_A`` are *kept* in
    meaning because only the separator between genus and epithet is converted when the name
    is written ``Genus_species``; a trailing ``_A`` on the genus is preserved as ``_a``.
    """
    if name is None:
        return ""
    s = str(name).strip()
    if "|" in s:  # full MetaPhlAn lineage
        s = s.split("|")[-1]
    if s.startswith("s__"):
        s = s[3:]
    parts = s.replace(" ", "_").split("_")
    # Re-join: single upper-case letters directly after a token are GTDB suffixes ("Blautia_A").
    tokens: list[str] = []
    for p in parts:
        if not p:
            continue
        if tokens and len(p) == 1 and p.isupper():
            tokens[-1] = tokens[-1] + "_" + p
        else:
            tokens.append(p)
    return " ".join(tokens).lower()
