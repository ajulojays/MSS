"""Phenotype prediction heads (one scikit-learn classifier per trait on a shared feature set).

These are independent heads on the same fixed representation, not a jointly trained
neural multi-task model.
"""
from __future__ import annotations

import datetime as _dt
import logging
import uuid

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .ontology import TASK_LABELS, norm_text, valid_label

log = logging.getLogger(__name__)

MIN_GENOMES = 6
MIN_CLASSES = 2
MIN_CV_GROUPS = 3


def _truthy(v) -> bool:
    return norm_text(v) in {"true", "1", "yes"}


def task_labels(manifest: pd.DataFrame, task: str) -> pd.Series:
    """Valid labels for one task, indexed by genome_id. Missing/conflicting/unknown dropped."""
    if task not in manifest.columns:
        return pd.Series(dtype=object)
    lab = manifest.set_index("genome_id")[task].map(norm_text)
    keep = lab.map(lambda v: valid_label(task, v))
    ccol = f"{task}_conflict"
    if ccol in manifest.columns:
        keep &= ~manifest.set_index("genome_id")[ccol].map(_truthy)
    return lab[keep]


def make_classifier(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed, C=1.0),
    )


def group_cv(clf, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    """Leave-groups-out CV: a held-out genus/family never contributes training examples."""
    ug = np.unique(groups)
    if len(ug) < MIN_CV_GROUPS:
        return {"status": "skipped", "reason": f"only {len(ug)} distinct groups (<{MIN_CV_GROUPS})"}
    n_splits = min(5, len(ug))
    oof = np.full(len(y), None, dtype=object)
    folds = []
    for i, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        assert not set(groups[tr]) & set(groups[te])
        if len(np.unique(y[tr])) < MIN_CLASSES:
            folds.append({"fold": i, "status": "skipped", "reason": "training fold has <2 classes",
                          "n_test": int(len(te))})
            continue
        m = clone(clf).fit(X[tr], y[tr])
        p = m.predict(X[te])
        oof[te] = p
        folds.append({"fold": i, "n_train": int(len(tr)), "n_test": int(len(te)),
                      "held_out_groups": sorted(map(str, np.unique(groups[te]))),
                      "balanced_accuracy": float(balanced_accuracy_score(y[te], p))
                      if len(np.unique(y[te])) > 1 else None})
    ev = np.array([o is not None for o in oof])
    if ev.sum() == 0:
        return {"status": "skipped", "reason": "no evaluable folds", "folds": folds}
    yt, yp = y[ev], oof[ev].astype(str)
    labels = sorted(np.unique(np.concatenate([yt, yp])))
    return {
        "status": "done",
        "n_splits": n_splits,
        "n_evaluated": int(ev.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
        "folds": folds,
        "_oof": oof,  # removed by train_heads before reporting
    }


DEFAULT_TASKS = ("oxygen", "motility", "sporulation", "gram_stain", "cell_shape")


def _per_stratum(y, oof, strata) -> dict:
    out = {}
    ev = np.array([o is not None for o in oof])
    for s_ in sorted(set(strata[ev])):
        m = ev & (strata == s_)
        yt, yp = y[m], oof[m].astype(str)
        out[str(s_)] = {"n": int(m.sum()), "accuracy": float((yt == yp).mean()),
                        "balanced_accuracy": float(balanced_accuracy_score(yt, yp))
                        if len(np.unique(yt)) > 1 else None,
                        "classes_present": sorted(map(str, np.unique(yt)))}
    return out


def train_heads(X: pd.DataFrame, manifest: pd.DataFrame, feature_type: str,
                tasks=DEFAULT_TASKS,
                cv_group: str = "genus", seed: int = 0,
                class_filter: dict | None = None,
                report_by: str | None = None) -> tuple[dict, dict]:
    """class_filter: {task: [classes]} trains that head only on those classes (e.g. OxyMetaG-style
    oxygen head on clear aerobes vs anaerobes). report_by: manifest column (e.g. gtdb_phylum) for
    per-stratum out-of-fold accuracy."""
    heads, report = {}, {"tasks": {}}
    meta = manifest.drop_duplicates("genome_id").set_index("genome_id")
    class_filter = class_filter or {}
    for task in tasks:
        lab = task_labels(manifest, task)
        if task in class_filter:
            lab = lab[lab.isin(class_filter[task])]
        lab = lab[lab.index.isin(X.index)]
        lab = lab[~lab.index.duplicated()]
        counts = {k: int(v) for k, v in lab.value_counts().items()}
        entry = {"n_labeled": int(len(lab)), "class_counts": counts}
        if len(lab) < MIN_GENOMES or len(counts) < MIN_CLASSES:
            entry["status"] = "skipped"
            entry["reason"] = (f"needs >= {MIN_GENOMES} labeled genomes across >= {MIN_CLASSES} "
                               f"classes; have {len(lab)} genomes, {len(counts)} classes")
            log.warning("task %s skipped: %s", task, entry["reason"])
            report["tasks"][task] = entry
            continue
        Xt = X.loc[lab.index].to_numpy(dtype=float)
        y = lab.to_numpy(dtype=object).astype(str)
        clf = make_classifier(seed)
        if cv_group in meta.columns:
            groups = meta.loc[lab.index, cv_group].fillna("").astype(str).to_numpy()
            groups = np.where(groups == "", "unknown_" + lab.index.to_numpy(), groups)
            entry["validation"] = group_cv(clf, Xt, y, groups, seed)
            entry["validation"]["group_column"] = cv_group
            oof = entry["validation"].pop("_oof", None)
            if report_by and report_by in meta.columns and oof is not None:
                strata = meta.loc[lab.index, report_by].fillna("").astype(str).to_numpy()
                entry["validation"][f"by_{report_by}"] = _per_stratum(y, oof, strata)
        else:
            entry["validation"] = {"status": "skipped", "reason": f"no {cv_group!r} column"}
        model = clone(clf).fit(Xt, y)
        entry["status"] = "trained"
        entry["classes"] = list(map(str, model.classes_))
        if task in class_filter:
            entry["class_filter"] = list(class_filter[task])
        heads[task] = model
        log.info("task %s: trained on %d genomes %s; CV %s", task, len(y), counts,
                 entry["validation"].get("balanced_accuracy", entry["validation"].get("reason")))
        report["tasks"][task] = entry
    if not heads:
        raise RuntimeError("no phenotype task had enough labeled genomes to train; see log")
    bundle = {
        "kind": "mss.phenotype_heads",
        "mss_version": __version__,
        "model_id": f"pheno-{uuid.uuid4().hex[:12]}",
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "feature_type": feature_type,
        "feature_columns": list(X.columns),
        "heads": heads,
        "seed": seed,
        # fixed vocabularies keep all columns; open ones (cell_shape, ph_range) use fitted classes
        "task_labels": {t: (class_filter.get(t) or TASK_LABELS.get(t) or list(map(str, heads[t].classes_)))
                        for t in heads},
        "class_filter": class_filter,
    }
    report.update({k: bundle[k] for k in ("model_id", "mss_version", "created", "feature_type", "seed")})
    report["n_features"] = X.shape[1]
    report["note"] = ("Training-set fit is not reported as performance. 'validation' metrics come "
                      f"only from leave-{cv_group}-out folds.")
    return bundle, report


def predict_heads(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    cols = bundle["feature_columns"]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"feature table lacks {len(missing)} model columns, e.g. {missing[:3]}")
    A = X[cols].to_numpy(dtype=float)
    out = pd.DataFrame(index=X.index)
    for task, m in bundle["heads"].items():
        proba = m.predict_proba(A)
        classes = list(map(str, m.classes_))
        for lab in bundle["task_labels"][task]:
            out[f"{task}_prob_{lab}"] = proba[:, classes.index(lab)] if lab in classes else 0.0
        out[f"{task}_pred"] = np.array(classes)[proba.argmax(1)]
    out["model_id"] = bundle["model_id"]
    out["feature_type"] = bundle["feature_type"]
    return out
