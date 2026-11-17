"""BacDive API v2 client and phenotype extraction.

API: https://api.bacdive.dsmz.de/ (no registration required since Feb 2026).
Endpoints used: /v2/fetch/{ids}, /v2/taxon/{genus}/{species?}, /v2/culturecollectionno/{no}.

Extraction rules (see README "Evidence rules"):
* Only experimentally recorded sections are read. The ``Genome-based predictions`` section is
  ignored, so computational predictions never enter the reference as ground truth.
* Missing measurements stay missing; they are never treated as negative.
* Within a strain, disagreeing measurements give an empty label plus ``<trait>_conflict=True``;
  the raw strings and reference IDs are kept. There is no majority vote.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .ontology import (
    BINARY_MAP,
    GRAM_MAP,
    load_oxygen_map,
    norm_text,
    oxygen_strictness,
)

log = logging.getLogger(__name__)

API_ROOT = "https://api.bacdive.dsmz.de/v2"

REFERENCE_COLUMNS = [
    "bacdive_id", "species", "genus", "family",
    "oxygen", "oxygen_detail", "oxygen_raw", "oxygen_conflict", "oxygen_unmapped",
    "evidence_references",
    "motility", "motility_conflict", "sporulation", "sporulation_conflict",
    "gram_stain", "gram_stain_conflict",
    "cell_shape", "cell_shape_conflict", "ph_range", "ph_range_conflict",
    "optimal_temperature_c", "optimal_temperature_raw",
    "growth_temperature_min_c", "growth_temperature_max_c", "growth_temperature_conflict",
    "optimal_ph", "optimal_ph_raw", "growth_ph_min", "growth_ph_max", "growth_ph_conflict",
    "optimal_nacl_pct", "optimal_nacl_raw", "growth_nacl_max_pct", "growth_nacl_conflict",
    "nacl_growth_raw",
    "genome_accession", "all_genome_accessions", "type_strain", "strain_designation",
    "bacdive_doi", "source",
]


# --------------------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------------------
class _Identity:
    """Mapping that keeps BacDive's own (lower-cased) string as the category."""

    def get(self, key, default=None):
        return key or default


class BacDiveClient:
    def __init__(self, cache_dir: str | Path | None = None, max_retries: int = 5,
                 backoff: float = 2.0, timeout: float = 60.0, pause: float = 0.2,
                 session=None):
        import requests  # imported lazily so offline modules do not need it

        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "mss-phenotype-client/0.1")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self.pause = pause

    def _get(self, url: str) -> dict:
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.get(url, timeout=self.timeout)
            except Exception as exc:  # network error
                last = exc
                wait = self.backoff ** attempt
                log.warning("GET %s failed (%s); retry in %.1fs", url, exc, wait)
                time.sleep(wait)
                continue
            if r.status_code == 200:
                time.sleep(self.pause)
                return r.json()
            if r.status_code == 404:
                return {"count": 0, "next": None, "results": {}}
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else self.backoff ** attempt
                log.warning("GET %s -> HTTP %s; retry in %.1fs", url, r.status_code, wait)
                last = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(wait)
                continue
            raise RuntimeError(f"BacDive GET {url} returned HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError(f"BacDive GET {url} failed after {self.max_retries} retries: {last}")

    def _paginate_ids(self, url: str) -> list[int]:
        ids: list[int] = []
        while url:
            page = self._get(url)
            res = page.get("results", [])
            if isinstance(res, dict):
                res = list(res.keys())
            ids.extend(int(x) for x in res)
            url = page.get("next")
        return ids

    def search_taxon(self, genus: str, species_epithet: str | None = None,
                     subspecies_epithet: str | None = None) -> list[int]:
        parts = [genus] + [p for p in (species_epithet, subspecies_epithet) if p]
        return self._paginate_ids(f"{API_ROOT}/taxon/" + "/".join(parts))

    def search_name(self, name: str) -> list[int]:
        """Search by a taxonomic name: 'Genus' or 'Genus species [subspecies]'."""
        toks = name.replace("_", " ").split()
        if not toks:
            raise ValueError("empty taxon name")
        return self.search_taxon(*toks[:3])

    def search_culture_collection(self, number: str) -> list[int]:
        return self._paginate_ids(f"{API_ROOT}/culturecollectionno/{number}")

    def fetch(self, ids: Iterable[int], chunk: int = 100) -> Iterator[tuple[int, dict]]:
        """Yield (bacdive_id, raw_record). Uses / fills the JSON cache."""
        todo = []
        for i in dict.fromkeys(int(x) for x in ids):
            cached = self._cache_path(i)
            if cached and cached.exists():
                yield i, json.loads(cached.read_text())
            else:
                todo.append(i)
        for s in range(0, len(todo), chunk):
            block = todo[s:s + chunk]
            url = f"{API_ROOT}/fetch/" + ";".join(map(str, block))
            while url:
                page = self._get(url)
                for k, rec in (page.get("results") or {}).items():
                    bid = int(k)
                    if self.cache_dir:
                        self._cache_path(bid).write_text(json.dumps(rec))
                    yield bid, rec
                url = page.get("next")

    def _cache_path(self, bid: int) -> Path | None:
        return self.cache_dir / f"{bid}.json" if self.cache_dir else None


# --------------------------------------------------------------------------------------
# Record parsing
# --------------------------------------------------------------------------------------
def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return [e for e in x if isinstance(e, dict)]
    if isinstance(x, dict):
        return [x]
    return []


def _resolve(values: list[str]) -> tuple[str, bool]:
    """Collapse per-measurement labels for one strain: (label, conflict)."""
    uniq = sorted(set(values))
    if not uniq:
        return "", False
    if len(uniq) == 1:
        return uniq[0], False
    return "", True


_NUM = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")


def _optimum(entries: list[dict], key: str) -> tuple[float | None, str]:
    raws = [str(e.get(key, "")).strip() for e in entries
            if norm_text(e.get("type")) == "optimum" and str(e.get(key, "")).strip()]
    raw = "; ".join(dict.fromkeys(raws))
    vals = sorted(set(raws))
    if len(vals) == 1 and _NUM.match(vals[0]):
        return float(vals[0]), raw
    return None, raw  # ranges or disagreeing optima stay unresolved; raw kept


_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:-\s*(-?\d+(?:\.\d+)?))?\s*$")


def _num_range(text, unit: str | None = None) -> tuple[float, float] | None:
    """'37' -> (37, 37); '5-30' -> (5, 30); with unit='%', '3.5 %' -> (3.5, 3.5).
    Anything else (other units, '<', '>') -> None, so it is left out rather than guessed."""
    t = str(text).strip()
    if unit:
        if not t.endswith(unit):
            return None
        t = t[: -len(unit)]
    m = _RANGE.match(t)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) is not None else lo
    return (min(lo, hi), max(lo, hi))


def _observed_growth_range(entries, value_key, growth_key, unit=None):
    """(min, max, conflict): span of values with positive growth (type/relation 'growth').

    If any negative-growth test falls inside that span (e.g. growth at 10 % NaCl in one
    reference, no growth at 10 % in another), the span is contradicted: (None, None, True).
    """
    lo, hi, neg = [], [], []
    for e in entries:
        kind = norm_text(e.get("type", e.get("tested relation")))
        if kind != "growth":
            continue
        r = _num_range(e.get(value_key, ""), unit)
        if not r:
            continue
        g = norm_text(e.get(growth_key))
        if g in {"positive", "yes"}:
            lo.append(r[0])
            hi.append(r[1])
        elif g in {"negative", "no"}:
            neg.append(r)
    if not lo:
        return None, None, False
    a, b = min(lo), max(hi)
    if any(n_lo <= b and n_hi >= a for n_lo, n_hi in neg):
        return None, None, True
    return a, b, False


def parse_record(bacdive_id: int, rec: dict, oxygen_map: dict[str, str] | None = None) -> dict:
    oxygen_map = oxygen_map or load_oxygen_map(None)
    tax = rec.get("Name and taxonomic classification", {}) or {}
    gen = rec.get("General", {}) or {}
    morph = rec.get("Morphology", {}) or {}
    phys = rec.get("Physiology and metabolism", {}) or {}
    cult = rec.get("Culture and growth conditions", {}) or {}
    seqi = rec.get("Sequence information", {}) or {}

    refs: dict[str, list[str]] = {}

    def add_ref(trait, e):
        r = e.get("@ref")
        if r is not None:
            refs.setdefault(trait, [])
            if str(r) not in refs[trait]:
                refs[trait].append(str(r))

    # Oxygen tolerance
    ox_raw, ox_cat, ox_detail, ox_unmapped = [], [], [], []
    for e in _as_list(phys.get("oxygen tolerance")):
        raw = e.get("oxygen tolerance")
        if raw is None:
            continue
        ox_raw.append(str(raw))
        add_ref("oxygen", e)
        cat = oxygen_map.get(norm_text(raw))
        if cat is None:
            ox_unmapped.append(str(raw))
        else:
            ox_cat.append(cat)
            ox_detail.append(f"{cat}:{oxygen_strictness(raw)}")
    oxygen, ox_conflict = _resolve(ox_cat)

    def binary_trait(entries, key, mapping, trait):
        vals = []
        for e in entries:
            v = e.get(key)
            if v is None:
                continue
            add_ref(trait, e)
            m = mapping.get(norm_text(v))
            if m is not None:
                vals.append(m)
        return _resolve(vals)

    cellm = _as_list(morph.get("cell morphology"))
    motility, mot_c = binary_trait(cellm, "motility", BINARY_MAP, "motility")
    gram, gram_c = binary_trait(cellm, "gram stain", GRAM_MAP, "gram_stain")
    spor, spor_c = binary_trait(_as_list(phys.get("spore formation")), "spore formation",
                                BINARY_MAP, "sporulation")

    temp_entries = _as_list(cult.get("culture temp"))
    ph_entries = _as_list(cult.get("culture pH"))
    t_opt, t_raw = _optimum(temp_entries, "temperature")
    ph_opt, ph_raw = _optimum(ph_entries, "pH")
    t_min, t_max, t_gc = _observed_growth_range(temp_entries, "temperature", "growth")
    ph_min, ph_max, ph_gc = _observed_growth_range(ph_entries, "pH", "ability")

    # Cell shape and pH category: BacDive's own strings, not renamed or merged.
    shape, shape_c = binary_trait(cellm, "cell shape", _Identity(), "cell_shape")
    ph_label, ph_label_c = binary_trait(ph_entries, "PH range", _Identity(), "ph_range")

    # Salt (NaCl only; concentrations parsed only when given in %).
    nacl = [e for e in _as_list(phys.get("halophily")) if norm_text(e.get("salt")) == "nacl"]
    for e in nacl:
        add_ref("nacl", e)
    opt_nacl_raw = [str(e.get("concentration", "")).strip() for e in nacl
                    if norm_text(e.get("tested relation")) == "optimum"]
    opt_vals = {_num_range(v, "%") for v in opt_nacl_raw}
    nacl_opt = (next(iter(opt_vals))[0] if len(opt_vals) == 1 and None not in opt_vals
                and next(iter(opt_vals))[0] == next(iter(opt_vals))[1] else None)
    _, nacl_max, nacl_gc = _observed_growth_range(nacl, "concentration", "growth", unit="%")
    nacl_growth_raw = "; ".join(f"{e.get('concentration')}:{e.get('growth')}" for e in nacl
                                if norm_text(e.get("tested relation")) == "growth")
    for e in temp_entries:
        if norm_text(e.get("type")) == "optimum":
            add_ref("optimal_temperature", e)
    for e in ph_entries:
        if norm_text(e.get("type")) == "optimum":
            add_ref("optimal_ph", e)

    genomes = []
    for e in _as_list(seqi.get("Genome sequences")):
        acc = e.get("INSDC accession") or e.get("accession")
        if acc:
            score = e.get("score")
            genomes.append((float(score) if isinstance(score, (int, float)) else float("-inf"),
                            str(acc)))
    # Primary genome = highest BacDive assembly score (stable sort keeps record order on ties).
    primary = sorted(genomes, key=lambda t: -t[0])[0][1] if genomes else ""

    return {
        "bacdive_id": int(bacdive_id),
        "species": tax.get("species", "") or "",
        "genus": tax.get("genus", "") or "",
        "family": tax.get("family", "") or "",
        "oxygen": oxygen,
        "oxygen_detail": "; ".join(dict.fromkeys(ox_detail)),
        "oxygen_raw": "; ".join(ox_raw),
        "oxygen_conflict": bool(ox_conflict),
        "oxygen_unmapped": "; ".join(dict.fromkeys(ox_unmapped)),
        "evidence_references": "|".join(f"{k}:{','.join(v)}" for k, v in refs.items()),
        "motility": motility, "motility_conflict": bool(mot_c),
        "sporulation": spor, "sporulation_conflict": bool(spor_c),
        "gram_stain": gram, "gram_stain_conflict": bool(gram_c),
        "cell_shape": shape, "cell_shape_conflict": bool(shape_c),
        "ph_range": ph_label, "ph_range_conflict": bool(ph_label_c),
        "optimal_temperature_c": t_opt, "optimal_temperature_raw": t_raw,
        "growth_temperature_min_c": t_min, "growth_temperature_max_c": t_max,
        "growth_temperature_conflict": bool(t_gc),
        "optimal_ph": ph_opt, "optimal_ph_raw": ph_raw,
        "growth_ph_min": ph_min, "growth_ph_max": ph_max, "growth_ph_conflict": bool(ph_gc),
        "optimal_nacl_pct": nacl_opt, "optimal_nacl_raw": "; ".join(opt_nacl_raw),
        "growth_nacl_max_pct": nacl_max, "growth_nacl_conflict": bool(nacl_gc),
        "nacl_growth_raw": nacl_growth_raw,
        "genome_accession": primary,
        "all_genome_accessions": ";".join(dict.fromkeys(a for _, a in genomes)),
        "type_strain": tax.get("type strain", "") or "",
        "strain_designation": tax.get("strain designation", "") or "",
        "bacdive_doi": gen.get("doi", "") or "",
        "source": "BacDive API v2",
    }


def build_reference(records: Iterable[tuple[int, dict]],
                    oxygen_map: dict[str, str] | None = None) -> pd.DataFrame:
    rows = [parse_record(i, r, oxygen_map) for i, r in records]
    df = pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
    return df.sort_values("bacdive_id").reset_index(drop=True)


def summarize_reference(df: pd.DataFrame) -> dict:
    out = {"n_strains": int(len(df)),
           "n_with_genome": int((df["genome_accession"] != "").sum())}
    for t in ["oxygen", "motility", "sporulation", "gram_stain", "cell_shape", "ph_range"]:
        out[t] = {k: int(v) for k, v in df[t].replace("", "missing").value_counts().items()}
        out[f"{t}_conflicts"] = int(df[f"{t}_conflict"].sum())
    for t in ["optimal_temperature_c", "growth_temperature_max_c", "optimal_ph",
              "optimal_nacl_pct", "growth_nacl_max_pct"]:
        out[f"n_with_{t}"] = int(pd.to_numeric(df[t], errors="coerce").notna().sum())
    unm = df["oxygen_unmapped"][df["oxygen_unmapped"] != ""]
    out["oxygen_unmapped_strings"] = sorted({s for x in unm for s in x.split("; ")})
    return out
