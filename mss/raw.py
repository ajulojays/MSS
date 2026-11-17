"""MSS-Raw (EXPERIMENTAL, UNVALIDATED): read k-mer profile -> oxygen-category composition.

Training uses synthetic mixtures of labeled genome k-mer profiles. Mixture weights are DNA
(sequence) fractions, so targets are the DNA-fraction-weighted category composition, not
cell counts. Finite sampling is simulated by multinomial draws of k-mers. This is NOT a read
simulator: no sequencing error, GC bias, coverage bias, or read-length effects are modelled.
"""
from __future__ import annotations

import datetime as _dt
import logging
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .ontology import OXYGEN_CLASSES

log = logging.getLogger(__name__)

STATUS = "EXPERIMENTAL_UNVALIDATED"


def project_simplex(Y: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row onto the probability simplex."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    n, d = Y.shape
    U = -np.sort(-Y, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, d + 1)
    cond = U - css / ind > 0
    rho = d - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(n), rho] / (rho + 1)
    return np.maximum(Y - theta[:, None], 0.0)


def simulate_mixtures(F: np.ndarray, T: np.ndarray, n: int, rng: np.random.Generator,
                      min_members: int = 2, max_members: int = 10, alpha: float = 1.0,
                      depth_range=(2e4, 2e6)) -> tuple[np.ndarray, np.ndarray]:
    """F: genome k-mer frequencies (g x f); T: one-hot oxygen class (g x 5)."""
    g = F.shape[0]
    hi = min(max_members, g)
    lo = min(min_members, hi)
    Xs, Ys = np.zeros((n, F.shape[1])), np.zeros((n, T.shape[1]))
    for i in range(n):
        m = rng.integers(lo, hi + 1)
        idx = rng.choice(g, size=m, replace=False)
        w = rng.dirichlet(np.full(m, alpha))
        p = w @ F[idx]
        depth = int(np.exp(rng.uniform(np.log(depth_range[0]), np.log(depth_range[1]))))
        Xs[i] = rng.multinomial(depth, p / p.sum()) / depth
        Ys[i] = w @ T[idx]
    return Xs, Ys


def _fit(X, Y):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15))).fit(X, Y)


def train_raw(X: pd.DataFrame, labels: pd.Series, groups: pd.Series | None, n_mixtures: int,
              seed: int = 0, feature_columns=None, k: int = 5, **sim) -> tuple[dict, dict]:
    labels = labels[labels.isin(OXYGEN_CLASSES)]
    ids = [i for i in labels.index if i in X.index]
    if len(ids) < 4 or labels.loc[ids].nunique() < 2:
        raise RuntimeError(f"train-raw needs >= 4 oxygen-labeled genomes in >= 2 classes; "
                           f"have {len(ids)} genomes, {labels.loc[ids].nunique()} classes")
    F = X.loc[ids].to_numpy(dtype=float)
    T = np.zeros((len(ids), len(OXYGEN_CLASSES)))
    for r, i in enumerate(ids):
        T[r, OXYGEN_CLASSES.index(labels[i])] = 1
    rng = np.random.default_rng(seed)
    report = {"n_genomes": len(ids), "n_mixtures": n_mixtures,
              "class_counts": {k_: int(v) for k_, v in labels.loc[ids].value_counts().items()},
              "simulation": {"mixture_weights": "Dirichlet DNA fractions", **{k_: str(v) for k_, v in sim.items()}},
              "status": STATUS}

    # Held-out-genus estimate: mixtures built only from genera never used in training.
    if groups is not None:
        g = groups.reindex(ids).fillna("unknown").astype(str).to_numpy()
        ug = np.unique(g)
        if len(ug) >= 3:
            test_g = set(rng.choice(ug, size=max(1, len(ug) // 4), replace=False))
            te = np.array([x in test_g for x in g])
            if te.sum() >= 2 and (~te).sum() >= 2 and len(np.unique(T[~te].argmax(1))) >= 2:
                Xtr, Ytr = simulate_mixtures(F[~te], T[~te], n_mixtures, rng, **sim)
                Xte, Yte = simulate_mixtures(F[te], T[te], max(50, n_mixtures // 4), rng, **sim)
                P = project_simplex(_fit(Xtr, Ytr).predict(Xte))
                mae = np.abs(P - Yte).mean(0)
                report["held_out_genus_validation"] = {
                    "held_out_groups": sorted(test_g), "n_test_genomes": int(te.sum()),
                    "n_test_mixtures": int(len(Xte)),
                    "mae_per_class": dict(zip(OXYGEN_CLASSES, map(float, mae))),
                    "mae_anaerobic_index_points": float(100 * mae[0]),
                    "note": "synthetic mixtures of k-mer profiles; not a validation on real metagenomes",
                }
            else:
                report["held_out_genus_validation"] = {"status": "skipped", "reason": "split too small"}
        else:
            report["held_out_genus_validation"] = {"status": "skipped", "reason": "<3 genera"}

    Xs, Ys = simulate_mixtures(F, T, n_mixtures, rng, **sim)
    model = _fit(Xs, Ys)
    bundle = {
        "kind": "mss.raw_mixture",
        "mss_version": __version__,
        "model_id": f"raw-kmer{k}-ridge-{uuid.uuid4().hex[:10]}",
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "k": k,
        "feature_columns": list(feature_columns if feature_columns is not None else X.columns),
        "classes": OXYGEN_CLASSES,
        "regressor": model,
        "status": STATUS,
        "seed": seed,
    }
    report["model_id"] = bundle["model_id"]
    return bundle, report


def predict_raw(bundle: dict, freqs: np.ndarray) -> np.ndarray:
    return project_simplex(bundle["regressor"].predict(np.atleast_2d(freqs)))
