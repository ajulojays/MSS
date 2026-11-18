"""Pfam domain features for genomes: pyrodigal gene calls + pyhmmer hmmsearch against Pfam-A.

Each genome gets one cache file (<cache_dir>/<genome_id>.tsv) listing, per Pfam family, the number of
predicted proteins with a hit above Pfam's own gathering (GA) thresholds. The work can be split across a
SLURM array (``task_id``/``n_tasks``); finished genomes are skipped, so jobs can be resubmitted safely.
``pfam_table`` then joins the caches into a genome x Pfam table (presence/absence or counts) with
``f_PFxxxxx`` columns, which ``mss train --embeddings`` reads like any other external feature table.

Requires the optional packages pyrodigal and pyhmmer.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

from .features import open_text

log = logging.getLogger(__name__)
MIN_TRAIN_BASES = 100_000  # pyrodigal recommends >= 100 kb for single-genome training; shorter -> meta mode


def _read_fasta(path) -> list[bytes]:
    seqs, cur = [], []
    with open_text(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).encode())
                cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur).encode())
    return seqs


def call_proteins(path) -> list[str]:
    """Predicted protein sequences of one genome (single-genome mode; meta mode if too short to train)."""
    import pyrodigal

    contigs = _read_fasta(path)
    if not contigs:
        raise ValueError("empty FASTA")
    if sum(len(c) for c in contigs) >= MIN_TRAIN_BASES:
        finder = pyrodigal.GeneFinder(meta=False)
        finder.train(*contigs)
    else:
        finder = pyrodigal.GeneFinder(meta=True)
    prots = []
    for c in contigs:
        for g in finder.find_genes(c):
            p = g.translate().rstrip("*")
            if p:
                prots.append(p)
    return prots


def load_pfam(hmm_path):
    """All HMMs of a Pfam-A.hmm file (plain or .gz); every model must carry GA cutoffs."""
    import pyhmmer

    t0 = time.time()
    with pyhmmer.plan7.HMMFile(str(hmm_path)) as fh:
        hmms = list(fh)
    missing = [h.name for h in hmms if h.cutoffs.gathering is None]
    if missing:
        raise ValueError(f"{len(missing)} HMMs lack gathering cutoffs (e.g. {missing[:3]}); use Pfam-A.hmm")
    log.info("loaded %d Pfam HMMs in %.0fs", len(hmms), time.time() - t0)
    return hmms


def _acc(x) -> str:
    x = x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    return x.split(".")[0]


def annotate_batch(genomes: list[tuple[str, str]], hmms, cpus: int = 1) -> dict[str, tuple[int, dict, str]]:
    """{genome_id: (n_proteins, {PFxxxxx: n_proteins_with_hit}, error)} for a batch searched together."""
    import pyhmmer

    alphabet = pyhmmer.easel.Alphabet.amino()
    out, seqs = {}, []
    for i, (gid, path) in enumerate(genomes):
        try:
            prots = call_proteins(path)
        except Exception as exc:  # recorded per genome, not fatal
            out[gid] = (0, {}, f"{type(exc).__name__}: {exc}")
            continue
        out[gid] = (len(prots), {}, "")
        for j, p in enumerate(prots):
            seqs.append(pyhmmer.easel.TextSequence(name=f"{i}|{j}".encode(), sequence=p).digitize(alphabet))
    if not seqs:
        return out
    block = pyhmmer.easel.DigitalSequenceBlock(alphabet, seqs)
    ids = [g for g, _ in genomes]
    for hits in pyhmmer.hmmsearch(hmms, block, cpus=cpus, bit_cutoffs="gathering"):
        acc = _acc(hits.query.accession or hits.query.name)
        per = {}
        for h in hits:
            if h.included:
                name = h.name.decode() if isinstance(h.name, (bytes, bytearray)) else str(h.name)
                gi = int(name.split("|")[0])
                per[gi] = per.get(gi, 0) + 1
        for gi, n in per.items():
            out[ids[gi]][1][acc] = n
    return out


def _write_cache(path: Path, n_prot: int, counts: dict, err: str):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as fh:
        fh.write(f"# n_proteins={n_prot}\terror={err}\n")
        for acc in sorted(counts):
            fh.write(f"{acc}\t{counts[acc]}\n")
    os.replace(tmp, path)


def read_cache(path) -> tuple[int, dict, str]:
    counts, n_prot, err = {}, 0, ""
    with open(path) as fh:
        head = fh.readline().lstrip("# ").rstrip("\n").split("\t")
        meta = dict(f.split("=", 1) for f in head if "=" in f)
        n_prot, err = int(meta.get("n_proteins", 0)), meta.get("error", "")
        for line in fh:
            acc, n = line.rstrip("\n").split("\t")
            counts[acc] = int(n)
    return n_prot, counts, err


def annotate_genomes(manifest: pd.DataFrame, hmm_path, cache_dir, cpus: int = 1, task_id: int = 0,
                     n_tasks: int = 1, batch: int = 20) -> dict:
    """Annotate this task's share of manifest genomes (every n_tasks-th, offset task_id) into cache_dir."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    rows = manifest.drop_duplicates("genome_id").sort_values("genome_id")
    mine = rows.iloc[task_id::n_tasks]
    todo = [(r.genome_id, r.genome_fasta) for r in mine.itertuples()
            if not (cache / f"{r.genome_id}.tsv").exists()]
    log.info("task %d/%d: %d genomes, %d already cached, %d to annotate", task_id, n_tasks, len(mine),
             len(mine) - len(todo), len(todo))
    if not todo:
        return {"assigned": len(mine), "annotated": 0, "errors": 0}
    hmms = load_pfam(hmm_path)
    n_err, t0 = 0, time.time()
    for s in range(0, len(todo), batch):
        res = annotate_batch(todo[s:s + batch], hmms, cpus=cpus)
        for gid, (n_prot, counts, err) in res.items():
            _write_cache(cache / f"{gid}.tsv", n_prot, counts, err)
            n_err += bool(err)
        done = min(s + batch, len(todo))
        rate = done / max(time.time() - t0, 1e-9) * 60
        log.info("annotated %d/%d genomes (%.1f/min, %d errors)", done, len(todo), rate, n_err)
    return {"assigned": len(mine), "annotated": len(todo), "errors": n_err}


def pfam_table(manifest: pd.DataFrame, cache_dir, value: str = "presence",
               min_genomes: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(genome x f_PFxxxxx table with a genome_id column, per-genome stats). Uncached genomes are listed
    in the stats with error 'not annotated' and left out of the table."""
    if value not in ("presence", "count"):
        raise ValueError("value must be 'presence' or 'count'")
    cache = Path(cache_dir)
    rows, stats = {}, []
    for gid in manifest.drop_duplicates("genome_id").genome_id:
        f = cache / f"{gid}.tsv"
        if not f.exists():
            stats.append({"genome_id": gid, "n_proteins": 0, "n_pfams": 0, "error": "not annotated"})
            continue
        n_prot, counts, err = read_cache(f)
        if not err and n_prot == 0:
            err = "no proteins predicted"
        stats.append({"genome_id": gid, "n_proteins": n_prot, "n_pfams": len(counts), "error": err})
        if not err:
            rows[gid] = counts
    X = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    if value == "presence":
        X = (X > 0).astype(int)
    X = X.loc[:, (X > 0).sum(axis=0) >= min_genomes]
    X = X[sorted(X.columns)]
    X.columns = [f"f_{c}" for c in X.columns]
    X.index.name = "genome_id"
    return X.reset_index(), pd.DataFrame(stats)
