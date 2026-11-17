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


def make_manifest(reference: pd.DataFrame, genome_dir: str = "genomes") -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per unique primary genome accession; returns (manifest, duplicate log)."""
    r = reference.copy()
    r["genome_accession"] = r["genome_accession"].fillna("").astype(str)
    r = r[r["genome_accession"] != ""]
    dup = r[r.duplicated("genome_accession", keep="first")]
    r = r.drop_duplicates("genome_accession", keep="first")
    m = pd.DataFrame({
        "genome_id": r["genome_accession"],
        "accession": r["genome_accession"],
        "genome_fasta": [str(Path(genome_dir) / f"{a}.fna.gz") for a in r["genome_accession"]],
    })
    for c in MANIFEST_COLUMNS:
        if c not in m.columns:
            m[c] = r[c].to_numpy() if c in r.columns else ""
    if len(dup):
        log.warning("%d BacDive strains share an accession with an earlier strain; kept first", len(dup))
    return m[MANIFEST_COLUMNS].reset_index(drop=True), dup


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
                import gzip
                with z.open(name) as src, gzip.open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                res[match] = ""
    return res


def download_genomes(manifest: pd.DataFrame, outdir: str | Path, batch_size: int = 50,
                     datasets_bin: str = "datasets", skip_existing: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch genomes for manifest rows; returns (clean labeled manifest, failure table)."""
    if shutil.which(datasets_bin) is None:
        raise FileNotFoundError(f"NCBI Datasets CLI {datasets_bin!r} not on PATH "
                                "(conda install -c conda-forge ncbi-datasets-cli)")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    m = manifest.copy()
    m["genome_fasta"] = [str(outdir / f"{a}.fna.gz") for a in m["accession"]]
    todo = [a for a, f in zip(m["accession"], m["genome_fasta"])
            if not (skip_existing and Path(f).exists() and validate_fasta(f)[0])]
    errors: dict[str, str] = {}
    for s in range(0, len(todo), batch_size):
        batch = todo[s:s + batch_size]
        log.info("downloading %d-%d of %d", s + 1, s + len(batch), len(todo))
        errors.update(_datasets_batch(batch, outdir, datasets_bin))
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
