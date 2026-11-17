"""MSS-Taxa: abundance-weighted community phenotype composition from taxonomic profiles.

Categorical traits:   C_sk = sum_i w_si * p_ik       (oxygen: anaerobic index = 100 * C_s,anaerobe)
Numeric traits:       abundance-weighted mean over taxa with a resolved value, plus coverage.

Assignment per species and trait (auditable in the per-taxon table):
1. reference, unambiguous measured value        -> used, source "reference"
2. reference, strains disagree / strain conflict -> unresolved; never overwritten by a model
3. no measured value (absent or empty)           -> model probabilities if supplied (categorical
                                                    traits only, source "model"), else unresolved
Numeric traits are never binned into categories here; no thresholds are applied.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .ontology import (
    CATEGORICAL_TRAITS,
    NUMERIC_TRAITS,
    OXYGEN_CLASSES,
    TASK_LABELS,
    is_missing,
    norm_text,
    normalize_species,
)

log = logging.getLogger(__name__)

FRACTION_COLS = [f"{c}_pct" for c in OXYGEN_CLASSES] + ["unresolved_pct"]


def read_profile(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#")
    need = {"sample_id", "species", "relative_abundance"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path}: missing required columns {sorted(miss)}")
    df = df[["sample_id", "species", "relative_abundance"]].copy()
    df["relative_abundance"] = pd.to_numeric(df["relative_abundance"], errors="coerce")
    if df["relative_abundance"].isna().any():
        bad = df[df["relative_abundance"].isna()].head(3)
        raise ValueError(f"{path}: non-numeric relative_abundance, e.g.\n{bad}")
    if (df["relative_abundance"] < 0).any():
        raise ValueError(f"{path}: negative relative_abundance values")
    df["sample_id"] = df["sample_id"].astype(str)
    df["species"] = df["species"].astype(str)
    return df


def _normalize_profile(profile: pd.DataFrame) -> pd.DataFrame:
    prof = profile.copy()
    prof["key"] = prof["species"].map(normalize_species)
    prof = prof[prof["relative_abundance"] > 0].copy()
    prof["w"] = prof["relative_abundance"] / prof.groupby("sample_id")["relative_abundance"].transform("sum")
    return prof


def _truthy(v) -> bool:
    return norm_text(v) in {"true", "1", "yes"}


# ----------------------------------------------------------------------- species collapse
def species_reference(ref: pd.DataFrame, trait: str = "oxygen") -> pd.DataFrame:
    """Collapse strain rows to species for one categorical trait, without majority voting.

    status: resolved (all labeled strains agree, no strain-level conflict) / conflict / unannotated.
    """
    if "species" not in ref.columns or trait not in ref.columns:
        raise ValueError(f"reference needs 'species' and {trait!r} columns")
    allowed = TASK_LABELS.get(trait)
    r = ref[["species", trait] + ([f"{trait}_conflict"] if f"{trait}_conflict" in ref.columns else [])].copy()
    r["key"] = r["species"].map(normalize_species)
    r = r[r["key"] != ""]
    r["lab"] = r[trait].map(lambda v: "" if is_missing(v) else norm_text(v))
    if allowed is not None:
        bad = set(r["lab"]) - set(allowed) - {""}
        if bad:
            raise ValueError(f"reference {trait} labels outside {allowed}: {sorted(bad)}")
    r["conf"] = r[f"{trait}_conflict"].map(_truthy) if f"{trait}_conflict" in r.columns else False
    rows = []
    for key, g in r.groupby("key", sort=True):
        labels = sorted(set(g.loc[g["lab"] != "", "lab"]))
        strain_conflict = bool(g["conf"].any())
        if strain_conflict or len(labels) > 1:
            status, lab = "conflict", ""
        elif len(labels) == 1:
            status, lab = "resolved", labels[0]
        else:
            status, lab = "unannotated", ""
        rows.append({"key": key, "reference_species": g["species"].iloc[0], "status": status,
                     trait: lab, "n_strains": len(g), "n_strains_labeled": int((g["lab"] != "").sum()),
                     "labels_seen": ";".join(labels), "strain_conflict": strain_conflict})
    out = pd.DataFrame(rows, columns=["key", "reference_species", "status", trait, "n_strains",
                                      "n_strains_labeled", "labels_seen", "strain_conflict"])
    return out.set_index("key")


def species_numeric(ref: pd.DataFrame, col: str) -> pd.DataFrame:
    """Species value for a numeric trait: resolved only if every strain with a value agrees."""
    r = ref[["species", col]].copy()
    r["key"] = r["species"].map(normalize_species)
    r["v"] = pd.to_numeric(r[col], errors="coerce")
    rows = []
    for key, g in r[r["key"] != ""].groupby("key", sort=True):
        vals = sorted(set(g["v"].dropna()))
        if len(vals) == 1:
            rows.append({"key": key, "status": "resolved", "value": vals[0], "values_seen": str(vals[0])})
        elif vals:
            rows.append({"key": key, "status": "conflict", "value": np.nan,
                         "values_seen": ";".join(f"{v:g}" for v in vals)})
        else:
            rows.append({"key": key, "status": "unannotated", "value": np.nan, "values_seen": ""})
    return pd.DataFrame(rows, columns=["key", "status", "value", "values_seen"]).set_index("key")


def species_predictions(pred: pd.DataFrame, trait: str = "oxygen") -> pd.DataFrame | None:
    """Mean genome-level class probabilities per normalized species, or None if absent."""
    pcols = [c for c in pred.columns if c.startswith(f"{trait}_prob_")]
    if "species" not in pred.columns or not pcols:
        return None
    p = pred[["species"] + pcols].copy()
    p["key"] = p["species"].map(normalize_species)
    p = p.groupby("key")[pcols].mean()
    s = p.sum(axis=1).replace(0, np.nan)
    p = p.div(s, axis=0).fillna(0.0)
    p.columns = [c[len(f"{trait}_prob_"):] for c in pcols]
    return p


# ----------------------------------------------------------------------- aggregation
def _categorical(prof: pd.DataFrame, ref: pd.DataFrame, trait: str, pred: pd.DataFrame | None,
                 classes: list[str] | None = None):
    sref = species_reference(ref, trait)
    spred = species_predictions(pred, trait) if pred is not None else None
    if classes is None:
        classes = list(TASK_LABELS.get(trait) or [])
        seen = set(sref.loc[sref["status"] == "resolved", trait])
        if spred is not None:
            seen |= set(spred.columns)
        classes = classes or sorted(seen)
        classes += sorted(seen - set(classes))
    K = len(classes)
    assign = {}
    for k in prof["key"].unique():
        probs = np.zeros(K)
        if k in sref.index:
            st = sref.at[k, "status"]
            if st == "resolved":
                probs[classes.index(sref.at[k, trait])] = 1.0
                assign[k] = ("reference", "matched_resolved", probs)
                continue
            if st == "conflict":
                assign[k] = ("unresolved", "matched_conflict", probs)
                continue
            match = "matched_unannotated"
        else:
            match = "unmatched"
        if spred is not None and k in spred.index:
            row = spred.loc[k]
            for j, c in enumerate(classes):
                probs[j] = float(row[c]) if c in row.index else 0.0
            assign[k] = ("model", match, probs)
        else:
            assign[k] = ("unresolved", match, probs)
    src = prof["key"].map(lambda k: assign[k][0]).to_numpy()
    P = np.vstack([assign[k][2] for k in prof["key"]]) if len(prof) else np.zeros((0, K))
    audit = pd.DataFrame({f"{trait}_source": src,
                          f"{trait}_match": prof["key"].map(lambda k: assign[k][1]).to_numpy()},
                         index=prof.index)
    for j, c in enumerate(classes):
        audit[f"{trait}_p_{c}"] = P[:, j]
    rows = []
    for sid, idx in prof.groupby("sample_id", sort=True).groups.items():
        loc = prof.index.get_indexer(idx)
        w = prof["w"].to_numpy()[loc]
        comp = (w[:, None] * P[loc]).sum(0)
        s = src[loc]
        row = {"sample_id": sid}
        for j, c in enumerate(classes):
            row[f"{c}_pct"] = 100 * comp[j]
        row["unresolved_pct"] = 100 * max(1.0 - comp.sum(), 0.0)
        row["reference_supported_pct"] = 100 * w[s == "reference"].sum()
        row["model_inferred_pct"] = 100 * w[s == "model"].sum()
        row["n_taxa"] = int(len(loc))
        row["n_taxa_unresolved"] = int((s == "unresolved").sum())
        row["method"] = "reference+model" if row["model_inferred_pct"] > 0 else "reference"
        rows.append(row)
    return pd.DataFrame(rows), audit, classes


def _numeric(prof: pd.DataFrame, ref: pd.DataFrame, col: str):
    sn = species_numeric(ref, col)
    val = prof["key"].map(lambda k: sn.at[k, "value"] if k in sn.index else np.nan).astype(float)
    status = prof["key"].map(lambda k: sn.at[k, "status"] if k in sn.index else "unmatched")
    audit = pd.DataFrame({col: val, f"{col}_status": status}, index=prof.index)
    rows = []
    for sid, g in prof.assign(_v=val).groupby("sample_id", sort=True):
        has = g["_v"].notna()
        cov = g.loc[has, "w"].sum()
        mean = (g.loc[has, "w"] * g.loc[has, "_v"]).sum() / cov if cov > 0 else np.nan
        rows.append({"sample_id": sid, f"{col}__weighted_mean": mean, f"{col}__coverage_pct": 100 * cov,
                     f"{col}__n_taxa_with_value": int(has.sum())})
    return pd.DataFrame(rows), audit


def compute_taxa(profile: pd.DataFrame, reference: pd.DataFrame,
                 predictions: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oxygen composition: (per-sample table, per-taxon audit table)."""
    prof = _normalize_profile(profile)
    samp, audit, _ = _categorical(prof, reference, "oxygen", predictions, list(OXYGEN_CLASSES))
    samp.insert(1, "mss_anaerobic_index", samp["anaerobe_pct"])
    audit = pd.concat([prof.drop(columns=["key"]).rename(columns={"w": "normalized_abundance"}),
                       prof[["key"]].rename(columns={"key": "species_key"}),
                       audit.rename(columns={"oxygen_source": "source", "oxygen_match": "match_status"})
                       .rename(columns=lambda c: c.replace("oxygen_p_", "p_"))], axis=1)
    return samp, audit


def compute_traits(profile: pd.DataFrame, reference: pd.DataFrame,
                   predictions: pd.DataFrame | None = None,
                   traits: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """All available traits in one wide per-sample table + per-taxon audit.

    Categorical trait columns: ``<trait>__<category>_pct``, ``<trait>__unresolved_pct``,
    ``<trait>__reference_supported_pct``, ``<trait>__model_inferred_pct``.
    Numeric: ``<trait>__weighted_mean``, ``<trait>__coverage_pct``, ``<trait>__n_taxa_with_value``.
    Traits absent from the reference are skipped (returned list = traits computed).
    """
    prof = _normalize_profile(profile)
    wanted = traits or (CATEGORICAL_TRAITS + NUMERIC_TRAITS)
    out = pd.DataFrame({"sample_id": sorted(prof["sample_id"].unique())})
    audits = [prof[["sample_id", "species", "key", "w"]].rename(
        columns={"key": "species_key", "w": "normalized_abundance"})]
    done = []
    for t in wanted:
        if t not in reference.columns:
            continue
        if t in CATEGORICAL_TRAITS:
            s, a, _ = _categorical(prof, reference, t, predictions)
            keep = [c for c in s.columns if c.endswith("_pct")]
            s = s[["sample_id"] + keep].rename(columns={c: f"{t}__{c}" for c in keep})
        elif t in NUMERIC_TRAITS:
            s, a = _numeric(prof, reference, t)
        else:
            raise ValueError(f"unknown trait {t!r}")
        out = out.merge(s, on="sample_id", how="left")
        audits.append(a)
        done.append(t)
    return out, pd.concat(audits, axis=1), done
