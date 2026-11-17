"""Genome manifest construction and NCBI Datasets retrieval.

Requires the NCBI Datasets CLI (``datasets``) on PATH for downloads (internet access).
Failures are recorded per accession and never stop the batch.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from .features import open_text

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "genome_id", "accession", "bacdive_id", "species", "genus", "family", "genome_fasta",
    "oxygen", "oxygen_conflict", "oxygen_detail", "motility", "motility_conflict",
    "sporulation", "sporulation_conflict", "gram_stain", "gram_stain_conflict",
    "cell_shape", "cell_shape_conflict", "ph_range", "ph_range_conflict",
    "type_strain", "strain_designation",
]
LABEL_TRAITS = ["oxygen", "motility", "sporulation", "gram_stain", "cell_shape", "ph_range"]
GTDB_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
GTDB_COLUMNS = ["gtdb_accession", "gtdb_representative", "gtdb_taxonomy"] + \
    [f"gtdb_{r}" for r in GTDB_RANKS] + ["checkm2_completeness", "checkm2_contamination"]


def acc_key(acc: str) -> str:
    """'GCA_000511385.1' / 'GCF_000511385' / 'GB_GCA_000511385.1' -> '000511385' (assembly number)."""
    a = str(acc).strip()
    for pre in ("RS_", "GB_"):
        if a.startswith(pre):
            a = a[3:]
    a = a.split(".")[0]
    return a[4:] if a[:4] in ("GCA_", "GCF_") else a


def load_gtdb(paths) -> pd.DataFrame:
    """GTDB metadata TSV(s) (e.g. bac120_metadata_r226.tsv.gz, ar53_metadata_r226.tsv.gz),
    indexed by assembly number. Only the needed columns are read."""
    want = ["accession", "gtdb_representative", "gtdb_taxonomy", "checkm2_completeness",
            "checkm2_contamination", "ncbi_genbank_assembly_accession"]
    frames = []
    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        df = pd.read_csv(p, sep="\t", usecols=lambda c: c in want, dtype=str, low_memory=False)
        frames.append(df)
        log.info("GTDB metadata %s: %d genomes", p, len(df))
    g = pd.concat(frames, ignore_index=True)
    g["key"] = [acc_key(x) for x in g["accession"]]
    g["gtdb_representative"] = g["gtdb_representative"].str.lower().isin(["t", "true", "1"])
    tax = g["gtdb_taxonomy"].fillna("").str.split(";", expand=True).reindex(columns=range(7)).fillna("")
    for i, r in enumerate(GTDB_RANKS):
        g[f"gtdb_{r}"] = tax[i].str.replace(r"^[a-z]__", "", regex=True)
    g = g.rename(columns={"accession": "gtdb_accession"})
    # A GenBank/RefSeq pair share one assembly number; keep the representative row first.
    g = g.sort_values("gtdb_representative", ascending=False).drop_duplicates("key")
    return g.set_index("key")[GTDB_COLUMNS]


def _merge_duplicate_genomes(r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Several BacDive strains -> one genome: merge labels without voting.

    Agreeing labels are kept; disagreeing labels become empty with ``<trait>_conflict=True``.
    """
    dup_mask = r.duplicated("genome_accession", keep=False)
    if not dup_mask.any():
        return r, r.iloc[0:0]
    dup = r[dup_mask].copy()
    keep = r.drop_duplicates("genome_accession", keep="first").set_index("genome_accession").astype(object)
    for acc, g in dup.groupby("genome_accession"):
        for t in LABEL_TRAITS:
            if t not in g.columns:
                continue
            vals = {str(v).strip() for v in g[t].fillna("") if str(v).strip()}
            conf = f"{t}_conflict"
            prior = g[conf].astype(str).str.lower().isin(["true", "1"]).any() if conf in g.columns else False
            if len(vals) > 1 or prior:
                keep.at[acc, t] = ""
                if conf in keep.columns:
                    keep.at[acc, conf] = True
            elif len(vals) == 1:
                keep.at[acc, t] = vals.pop()
        keep.at[acc, "bacdive_id"] = ";".join(map(str, g["bacdive_id"]))
    return keep.reset_index(), dup


def make_manifest(reference: pd.DataFrame, genome_dir: str = "genomes",
                  require_label: list[str] | None = None, gtdb: pd.DataFrame | None = None,
                  gtdb_reps_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per unique genome; returns (manifest, duplicate log).

    require_label: keep strains with a measured value for at least one of these traits.
    gtdb: output of :func:`load_gtdb`. When given, a strain's accession that is a GTDB species
      representative is preferred over BacDive's primary accession, and GTDB taxonomy and
      CheckM2 columns are added.
    gtdb_reps_only: keep only GTDB species-representative genomes (one genome per GTDB species
      cluster), the dereplication strategy of OxyMetaG (Bueno de Mesquita et al. 2026).
    """
    r = reference.copy()
    r["genome_accession"] = r["genome_accession"].fillna("").astype(str)
    r = r[r["genome_accession"] != ""]
    if require_label:
        has = pd.Series(False, index=r.index)
        for t in require_label:
            if t in r.columns:
                has |= r[t].fillna("").astype(str).str.strip() != ""
        log.info("%d of %d strains with a genome have a measured %s label",
                 int(has.sum()), len(r), "/".join(require_label))
        r = r[has].copy()
    if gtdb is not None:
        alls = r["all_genome_accessions"] if "all_genome_accessions" in r.columns else r["genome_accession"]
        chosen = []
        for primary, allacc in zip(r["genome_accession"], alls.fillna("")):
            cands = [a for a in str(allacc).split(";") if a] or [primary]
            reps = [a for a in cands if acc_key(a) in gtdb.index and gtdb.at[acc_key(a), "gtdb_representative"]]
            chosen.append(reps[0] if reps else primary)
        r["genome_accession"] = chosen
        info = gtdb.reindex([acc_key(a) for a in r["genome_accession"]])
        for c in GTDB_COLUMNS:
            r[c] = info[c].to_numpy()
        r["gtdb_representative"] = r["gtdb_representative"].fillna(False).astype(bool)
        log.info("GTDB r226 match: %d of %d strains; %d use a GTDB species representative",
                 int(info["gtdb_accession"].notna().sum()), len(r), int(r["gtdb_representative"].sum()))
        if gtdb_reps_only:
            r = r[r["gtdb_representative"]].copy()
    elif gtdb_reps_only:
        raise ValueError("gtdb_reps_only needs GTDB metadata")
    r, dup = _merge_duplicate_genomes(r)
    if len(dup):
        log.warning("%d BacDive strains share a genome with another strain; labels merged "
                    "(disagreements -> conflict)", len(dup))
    m = pd.DataFrame({
        "genome_id": r["genome_accession"].to_numpy(),
        "accession": r["genome_accession"].to_numpy(),
        "genome_fasta": [str(Path(genome_dir) / f"{a}.fna.gz") for a in r["genome_accession"]],
    })
    cols = MANIFEST_COLUMNS + ([c for c in GTDB_COLUMNS] if gtdb is not None else [])
    for c in cols:
        if c not in m.columns:
            m[c] = r[c].to_numpy() if c in r.columns else ""
    return m[cols].reset_index(drop=True), dup


def validate_fasta(path) -> tuple[bool, str, int, int]:
    """Readable FASTA with >=1 record and >=1 sequence base; returns (ok, msg, n_contigs, n_bases)."""
    n_rec = n_bases = 0
    try:
        with open_text(path) as fh:
            for line in fh:
                if line.startswith(">"):
                    n_rec += 1
                elif line.strip():
                    if n_rec == 0:
                        return False, "sequence before first header", 0, 0
                    n_bases += len(line.strip())
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", n_rec, n_bases
    if n_rec == 0 or n_bases == 0:
        return False, "empty FASTA", n_rec, n_bases
    return True, "ok", n_rec, n_bases


def sha256(path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _datasets_batch(accessions: list[str], outdir: Path, datasets_bin: str) -> dict[str, str]:
    """Download one batch; returns {accession: error or ''}."""
    res = {a: "not returned by NCBI Datasets" for a in accessions}
    with tempfile.TemporaryDirectory(dir=outdir) as td:
        td = Path(td)
        acc_file = td / "acc.txt"
        acc_file.write_text("\n".join(accessions) + "\n")
        zpath = td / "batch.zip"
        cmd = [datasets_bin, "download", "genome", "accession", "--inputfile", str(acc_file),
               "--include", "genome", "--filename", str(zpath), "--no-progressbar"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not zpath.exists():
            msg = (p.stderr or p.stdout).strip().splitlines()
            return {a: f"datasets exit {p.returncode}: {msg[-1] if msg else ''}" for a in accessions}
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.endswith((".fna", ".fasta", ".fa")):
                    continue
                folder = Path(name).parent.name  # GCA_000001405.15
                base = folder.split(".")[0]
                match = next((a for a in accessions if a == folder or a.split(".")[0] == base), None)
                if match is None:
                    continue
                dest = outdir / f"{match}.fna.gz"
                part = dest.with_name(dest.name + ".part")
                import gzip
                with z.open(name) as src, gzip.open(part, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst)
                part.replace(dest)  # atomic: a killed job never leaves a truncated .fna.gz
                res[match] = ""
    return res


def _fetch_batch(batch: list[str], outdir: Path, datasets_bin: str) -> dict[str, str]:
    res = _datasets_batch(batch, outdir, datasets_bin)
    failed = [acc for acc, e in res.items() if e]
    if len(batch) > 1 and len(failed) == len(batch):
        # whole batch failed (e.g. one invalid accession or a transient error): retry one by one
        log.warning("batch failed (%s); retrying %d accessions individually", res[batch[0]], len(batch))
        for acc in batch:
            res.update(_datasets_batch([acc], outdir, datasets_bin))
    return res


def download_genomes(manifest: pd.DataFrame, outdir: str | Path, batch_size: int = 50,
                     datasets_bin: str = "datasets", skip_existing: bool = True,
                     workers: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch genomes for manifest rows; returns (clean labeled manifest, failure table).

    ``workers`` > 1 runs that many NCBI Datasets batch downloads concurrently.
    Setting NCBI_API_KEY in the environment raises NCBI's request-rate limit.
    """
    if shutil.which(datasets_bin) is None:
        raise FileNotFoundError(f"NCBI Datasets CLI {datasets_bin!r} not on PATH "
                                "(conda install -c conda-forge ncbi-datasets-cli)")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    m = manifest.copy()
    m["genome_fasta"] = [str(outdir / f"{a}.fna.gz") for a in m["accession"]]
    todo = [a for a, f in zip(m["accession"], m["genome_fasta"])
            if not (skip_existing and Path(f).exists() and validate_fasta(f)[0])]
    for stale in outdir.glob("*.fna.gz.part"):
        stale.unlink()
    errors: dict[str, str] = {}
    batches = [todo[s:s + batch_size] for s in range(0, len(todo), batch_size)]
    workers = max(1, workers)
    log.info("%d genomes to download in %d batches of <=%d with %d parallel workers (%d already present)",
             len(todo), len(batches), batch_size, workers, len(m) - len(todo))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_batch, b, outdir, datasets_bin) for b in batches]
        for f in as_completed(futs):
            res = f.result()
            errors.update(res)
            done += len(res)
            log.info("downloaded %d/%d (%d failed so far)", done, len(todo),
                     sum(1 for e in errors.values() if e))
    fails, ok_rows, seen_hash = [], [], {}
    for r in m.to_dict("records"):
        err = errors.get(r["accession"], "")
        if not err:
            ok, msg, n_rec, n_bases = validate_fasta(r["genome_fasta"])
            if not ok:
                err = f"invalid FASTA: {msg}"
        if err:
            fails.append({"accession": r["accession"], "species": r["species"], "error": err})
            continue
        h = sha256(r["genome_fasta"])
        if h in seen_hash:
            fails.append({"accession": r["accession"], "species": r["species"],
                          "error": f"identical file to {seen_hash[h]} (dropped as duplicate)"})
            continue
        seen_hash[h] = r["accession"]
        r.update({"n_contigs": n_rec, "genome_length": n_bases, "sha256": h})
        ok_rows.append(r)
    return pd.DataFrame(ok_rows), pd.DataFrame(fails, columns=["accession", "species", "error"])
