"""Sequence feature extraction: streaming canonical k-mer counts and external embeddings.

Canonical k-mer counting is a *baseline* genome representation, not a foundation model.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_LUT = np.full(256, 4, dtype=np.uint8)
for _c, _v in zip(b"ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
    _LUT[_c] = _v

EMBEDDING_PREFIXES = ("dna_", "protein_", "f_", "kmer_")  # kmer_ = a table written by `mss features`
CHUNK_BASES = 4_000_000


def open_text(path: str | os.PathLike):
    """Open plain or gzip-compressed text, detected from magic bytes."""
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="ascii", errors="replace")
    return open(path, "r", encoding="ascii", errors="replace")


class KmerSpace:
    """Canonical k-mer index: forward code -> canonical column."""

    def __init__(self, k: int = 5):
        if not 1 <= k <= 12:
            raise ValueError("k must be in 1..12")
        self.k = k
        n = 4 ** k
        codes = np.arange(n, dtype=np.int64)
        rc = np.zeros(n, dtype=np.int64)
        tmp = codes.copy()
        for _ in range(k):
            rc = (rc << 2) | (3 - (tmp & 3))
            tmp >>= 2
        canon = np.minimum(codes, rc)
        uniq, self.index = np.unique(canon, return_inverse=True)
        self.n_features = len(uniq)
        self.names = [f"kmer_{self.decode(int(c))}" for c in uniq]

    def decode(self, code: int) -> str:
        s = []
        for _ in range(self.k):
            s.append("ACGT"[code & 3])
            code >>= 2
        return "".join(reversed(s))

    def canonicalize(self, forward_counts: np.ndarray) -> np.ndarray:
        return np.bincount(self.index, weights=forward_counts, minlength=self.n_features)


def count_forward(seq: bytes, k: int, counts: np.ndarray) -> int:
    """Add forward-strand k-mer codes of ``seq`` to ``counts``; windows with non-ACGT skipped."""
    a = _LUT[np.frombuffer(seq, dtype=np.uint8)]
    n = a.size
    if n < k:
        return 0
    bad = (a == 4)
    cs = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    valid = (cs[k:] - cs[:-k]) == 0
    a64 = a.astype(np.int64)
    a64[bad] = 0
    m = n - k + 1
    code = np.zeros(m, dtype=np.int64)
    for j in range(k):
        code <<= 2
        code |= a64[j:j + m]
    c = code[valid]
    counts += np.bincount(c, minlength=counts.size)
    return int(c.size)


@dataclass
class KmerProfile:
    counts: np.ndarray        # canonical counts
    n_records: int            # contigs or reads
    n_bases: int
    n_kmers: int

    def frequencies(self) -> np.ndarray:
        tot = self.counts.sum()
        if tot <= 0:
            raise ValueError("no valid k-mers (sequence empty or all ambiguous)")
        return self.counts / tot


def fasta_kmers(path, k: int = 5, space: KmerSpace | None = None) -> KmerProfile:
    """Stream a (gzipped) FASTA; contigs are processed in chunks, never held whole."""
    space = space or KmerSpace(k)
    fwd = np.zeros(4 ** k, dtype=np.int64)
    n_rec = n_bases = n_kmers = 0
    buf: list[bytes] = []
    buf_len = 0
    seen_header = False

    def flush(final: bool):
        nonlocal buf, buf_len, n_kmers
        if not buf:
            return
        s = b"".join(buf)
        n_kmers += count_forward(s, k, fwd)
        if final:
            buf, buf_len = [], 0
        else:
            tail = s[-(k - 1):] if k > 1 else b""
            buf, buf_len = [tail], len(tail)

    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                flush(True)
                n_rec += 1
                seen_header = True
                continue
            if not seen_header:
                raise ValueError(f"{path}: not FASTA (first record lacks '>')")
            b = line.encode("ascii", "replace")
            buf.append(b)
            buf_len += len(b)
            n_bases += len(b)
            if buf_len >= CHUNK_BASES:
                flush(False)
        flush(True)
    if n_rec == 0:
        raise ValueError(f"{path}: no FASTA records")
    return KmerProfile(space.canonicalize(fwd), n_rec, n_bases, n_kmers)


def iter_fastq(path):
    """Yield sequences from a FASTQ, validating 4-line structure; raises on truncation."""
    with open_text(path) as fh:
        lineno = 0
        while True:
            h = fh.readline()
            if not h:
                return
            lineno += 1
            if not h.strip():
                continue
            s, p, q = fh.readline(), fh.readline(), fh.readline()
            lineno += 3
            if not h.startswith("@"):
                raise ValueError(f"{path}: line {lineno - 3}: FASTQ header must start with '@'")
            if not q:
                raise ValueError(f"{path}: truncated FASTQ record at line {lineno - 3}")
            if not p.startswith("+"):
                raise ValueError(f"{path}: line {lineno - 1}: expected '+' separator")
            s, q = s.rstrip("\r\n"), q.rstrip("\r\n")
            if len(s) != len(q):
                raise ValueError(f"{path}: line {lineno - 2}: sequence/quality length mismatch")
            yield s


def fastq_kmers(paths, k: int = 5, max_reads: int | None = None, batch: int = 20000,
                space: KmerSpace | None = None) -> KmerProfile:
    """Stream one or more FASTQ files (e.g. R1 and R2) into one canonical k-mer profile."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    space = space or KmerSpace(k)
    fwd = np.zeros(4 ** k, dtype=np.int64)
    n_reads = n_bases = n_kmers = 0
    chunk: list[str] = []
    done = False
    for path in paths:
        for seq in iter_fastq(path):
            chunk.append(seq)
            n_reads += 1
            n_bases += len(seq)
            if len(chunk) >= batch:
                n_kmers += count_forward("N".join(chunk).encode("ascii", "replace"), k, fwd)
                chunk = []
            if max_reads and n_reads >= max_reads:
                done = True
                break
        if done:
            break
    if chunk:
        n_kmers += count_forward("N".join(chunk).encode("ascii", "replace"), k, fwd)
    if n_reads == 0:
        raise ValueError(f"no reads in {list(map(str, paths))}")
    return KmerProfile(space.canonicalize(fwd), n_reads, n_bases, n_kmers)


# --------------------------------------------------------------------------------------
# Genome feature table with on-disk cache
# --------------------------------------------------------------------------------------
def _cache_key(path: Path, k: int) -> str:
    st = path.stat()
    h = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|k={k}".encode()).hexdigest()
    return h[:20]


def _genome_job(args):
    gid, path, k, cache_dir = args
    path = Path(path)
    cache = Path(cache_dir) / f"{_cache_key(path, k)}.npz" if cache_dir else None
    if cache and cache.exists():
        z = np.load(cache)
        return gid, z["counts"], int(z["n_bases"]), int(z["n_records"]), None
    try:
        prof = fasta_kmers(path, k)
    except Exception as exc:  # recorded, not fatal
        return gid, None, 0, 0, f"{type(exc).__name__}: {exc}"
    if cache:
        tmp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, counts=prof.counts, n_bases=prof.n_bases, n_records=prof.n_records)
        os.replace(tmp, cache)
    return gid, prof.counts, prof.n_bases, prof.n_records, None


def genome_kmer_table(manifest: pd.DataFrame, k: int = 5, cache_dir: str | None = None,
                      threads: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (frequency table indexed by genome_id, per-genome stats incl. errors)."""
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(r.genome_id, r.genome_fasta, k, cache_dir) for r in manifest.itertuples()]
    space = KmerSpace(k)
    if threads > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=threads) as ex:
            results = list(ex.map(_genome_job, jobs, chunksize=max(1, len(jobs) // (threads * 4))))
    else:
        results = [_genome_job(j) for j in jobs]
    rows, stats = {}, []
    for gid, counts, n_bases, n_rec, err in results:
        stats.append({"genome_id": gid, "n_bases": n_bases, "n_contigs": n_rec, "error": err or ""})
        if err is None:
            tot = counts.sum()
            if tot > 0:
                rows[gid] = counts / tot
            else:
                stats[-1]["error"] = "no valid k-mers"
    X = pd.DataFrame.from_dict(rows, orient="index", columns=space.names)
    X.index.name = "genome_id"
    return X, pd.DataFrame(stats)


# --------------------------------------------------------------------------------------
# External embeddings
# --------------------------------------------------------------------------------------
def load_embeddings(path, key: str = "genome_id") -> pd.DataFrame:
    """Load a precomputed embedding TSV: a key column + numeric dna_*/protein_*/f_* columns."""
    df = pd.read_csv(path, sep="\t")
    if key not in df.columns:
        raise ValueError(f"{path}: key column {key!r} not found (columns: {list(df.columns)[:6]}...)")
    feat = [c for c in df.columns if c.startswith(EMBEDDING_PREFIXES)]
    if not feat:
        raise ValueError(f"{path}: no feature columns with prefixes {EMBEDDING_PREFIXES}")
    if df[key].duplicated().any():
        raise ValueError(f"{path}: duplicated {key} values")
    X = df.set_index(key)[feat]
    bad = [c for c in feat if not pd.api.types.is_numeric_dtype(X[c])]
    if bad:
        raise ValueError(f"{path}: non-numeric feature columns: {bad[:5]}")
    if X.isna().any().any():
        raise ValueError(f"{path}: missing values in embedding columns")
    X.index = X.index.astype(str)
    return X.astype(float)
