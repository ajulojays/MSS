"""BacDive test results beyond the core phenotypes: metabolite utilization, enzymes, metabolite production
and halophily.

``parse_tests`` turns one cached BacDive record into long rows (one per strain x test entry) that keep
BacDive's own wording: no metabolite, enzyme, habitat or result is renamed, merged or re-coded.

``trait_labels`` pivots those rows into one categorical trait per test (a manifest-style column per
test, values = BacDive's result strings), so each can be trained like the core traits:
  * metabolite utilization: one trait per (metabolite, kind of utilization tested); values '+', '-', '+/-'...
  * enzymes: one trait per enzyme name (EC kept in the label table); values '+', '-', ...
  * metabolite production: one trait per metabolite; values 'yes', 'no', ...
  * halophily: one trait per (salt, tested relation, concentration string); values 'positive', 'no', ...
A strain whose entries for one test disagree gets no label for it and a conflict flag.
Classes with fewer than ``min_class`` genomes are left out of a trait (not merged), and traits left with
fewer than two classes are dropped; both are reported.
"""
from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .ontology import norm_text

SECTIONS = {
    "metabolite_utilization": ("Physiology and metabolism", "metabolite utilization"),
    "enzymes": ("Physiology and metabolism", "enzymes"),
    "metabolite_production": ("Physiology and metabolism", "metabolite production"),
    "halophily": ("Physiology and metabolism", "halophily"),
}
PREFIX = {"metabolite_utilization": "util", "enzymes": "enz",
          "metabolite_production": "prod", "halophily": "salt"}
TEST_COLUMNS = ["bacdive_id", "section", "item", "item_id", "detail", "result", "ref"]


def _as_list(x) -> list:
    if isinstance(x, list):
        return [e for e in x if isinstance(e, dict)]
    return [x] if isinstance(x, dict) else []


def _s(x) -> str:
    return "" if x is None else str(x).strip()


def parse_tests(bacdive_id: int, rec: dict) -> list[dict]:
    """Long rows for the four sections of one BacDive record (BacDive's wording kept)."""
    if isinstance(rec, dict) and isinstance(rec.get("results"), dict):  # API envelope
        rec = next(iter(rec["results"].values()), {}) if rec["results"] else {}
    rows = []

    def add(section, item, item_id, detail, result, ref):
        if _s(item) and _s(result):
            rows.append({"bacdive_id": int(bacdive_id), "section": section, "item": _s(item),
                         "item_id": _s(item_id), "detail": _s(detail), "result": _s(result), "ref": _s(ref)})

    def entries(key):
        sec, sub = SECTIONS[key]
        return _as_list((rec.get(sec) or {}).get(sub))

    for e in entries("metabolite_utilization"):
        add("metabolite_utilization", e.get("metabolite"), e.get("Chebi-ID"), e.get("kind of utilization tested"),
            e.get("utilization activity"), e.get("@ref"))
    for e in entries("enzymes"):
        add("enzymes", e.get("value"), e.get("ec"), "", e.get("activity"), e.get("@ref"))
    for e in entries("metabolite_production"):
        add("metabolite_production", e.get("metabolite"), e.get("Chebi-ID"), "", e.get("production"), e.get("@ref"))
    for e in entries("halophily"):
        add("halophily", e.get("salt"), "", f"{_s(e.get('tested relation'))}; {_s(e.get('concentration'))}",
            e.get("growth"), e.get("@ref"))
    return rows


def _load(path: Path):
    try:
        with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _job(args):
    bid, path = args
    rec = _load(path)
    return parse_tests(bid, rec) if rec is not None else None


def tests_from_cache(cache_dir, bacdive_ids, threads: int = 1) -> tuple[pd.DataFrame, list[int]]:
    """(long test table, ids with no readable cached record) for the given BacDive ids."""
    cache = Path(cache_dir)
    jobs = []
    for b in bacdive_ids:
        p = cache / f"{int(b)}.json.gz"
        jobs.append((int(b), p if p.exists() else cache / f"{int(b)}.json"))
    rows, missing = [], []
    if threads > 1:
        with ProcessPoolExecutor(threads) as ex:
            results = list(ex.map(_job, jobs, chunksize=200))
    else:
        results = [_job(j) for j in jobs]
    for (b, _), r in zip(jobs, results):
        if r is None:
            missing.append(b)
        else:
            rows.extend(r)
    return pd.DataFrame(rows, columns=TEST_COLUMNS), missing


def _slug(text: str) -> str:
    s = re.sub(r"[^0-9a-z]+", "_", norm_text(text)).strip("_")
    return s or "x"


def _trait_key(r) -> tuple[str, str]:
    """(column name, human-readable description) for one test row."""
    p = PREFIX[r.section]
    if r.section == "metabolite_utilization":
        cid = f"chebi{r.item_id}" if r.item_id else _slug(r.item)
        return f"{p}__{cid}__{_slug(r.detail)}", f"{r.item} ({r.detail or 'kind not given'})"
    if r.section == "metabolite_production":
        cid = f"chebi{r.item_id}" if r.item_id else _slug(r.item)
        return f"{p}__{cid}", r.item
    if r.section == "enzymes":
        return f"{p}__{_slug(r.item)}", r.item + (f" (EC {r.item_id})" if r.item_id else "")
    return f"{p}__{_slug(r.item)}__{_slug(r.detail)}", f"{r.item}, {r.detail}"  # halophily


def trait_labels(tests: pd.DataFrame, bacdive_ids, min_class: int = 50,
                 sections=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(wide label table: bacdive_id + one column per kept trait + <trait>_conflict, trait catalogue).

    Labels use BacDive's result strings (lower-cased, whitespace collapsed). Only strains in
    ``bacdive_ids`` count toward class sizes.
    """
    ids = {int(b) for b in bacdive_ids}
    t = tests[tests.bacdive_id.astype(int).isin(ids)].copy()
    if sections:
        t = t[t.section.isin(sections)]
    if t.empty:
        return pd.DataFrame({"bacdive_id": sorted(ids)}), pd.DataFrame()
    keys = [_trait_key(r) for r in t.itertuples()]
    t["trait"] = [k[0] for k in keys]
    t["description"] = [k[1] for k in keys]
    t["value"] = t.result.map(norm_text)
    desc = t.drop_duplicates("trait").set_index("trait")[["section", "description", "item_id"]]

    per = t.groupby(["trait", "bacdive_id"]).value.agg(lambda v: sorted(set(v)))
    lab = per.map(lambda v: v[0] if len(v) == 1 else "")
    conflict = per.map(len) > 1

    wide = lab.unstack("trait").fillna("")
    wide_c = conflict.unstack("trait").fillna(False)

    kept, cat = [], []
    for col in wide.columns:
        counts = wide[col][wide[col] != ""].value_counts()
        big = counts[counts >= min_class]
        dropped = counts[counts < min_class]
        rec = {"trait": col, "section": desc.loc[col, "section"], "description": desc.loc[col, "description"],
               "item_id": desc.loc[col, "item_id"], "n_conflict": int(wide_c[col].sum()),
               "classes_kept": "; ".join(f"{k}={v}" for k, v in big.items()),
               "classes_left_out": "; ".join(f"{k}={v}" for k, v in dropped.items())}
        if len(big) >= 2:
            wide.loc[~wide[col].isin(big.index), col] = ""
            kept.append(col)
            rec["status"] = "kept"
        else:
            rec["status"] = f"dropped: fewer than 2 classes with >= {min_class} genomes"
        cat.append(rec)
    out = wide[kept].copy()
    for col in kept:
        out[f"{col}_conflict"] = wide_c[col].reindex(out.index).fillna(False).astype(bool)
    out.index = out.index.astype(int)
    out = out.reindex(sorted(ids)).fillna({c: "" for c in kept})
    for col in kept:
        out[f"{col}_conflict"] = out[f"{col}_conflict"].fillna(False).astype(bool)
    out.index.name = "bacdive_id"
    catalogue = pd.DataFrame(cat).sort_values(["status", "section", "trait"])
    return out.reset_index(), catalogue
