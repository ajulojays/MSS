"""Pfam features: gene calling, GA-thresholded search, caching, table, and use as training features."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pyhmmer = pytest.importorskip("pyhmmer")
pytest.importorskip("pyrodigal")

from mss.cli import main  # noqa: E402
from mss.pfam import call_proteins, pfam_table, read_cache  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _toy_pfam(prot: str, path: Path, acc: str = "PF99999", ga: float = 20.0):
    """One HMM built from a single protein, with GA cutoffs, written like a Pfam-A.hmm file."""
    ab = pyhmmer.easel.Alphabet.amino()
    seq = pyhmmer.easel.TextSequence(name=b"seed", sequence=prot)
    msa = pyhmmer.easel.TextMSA(name=acc.encode(), sequences=[seq]).digitize(ab)
    hmm, _, _ = pyhmmer.plan7.Builder(ab).build_msa(msa, pyhmmer.plan7.Background(ab))
    hmm.accession = f"{acc}.1".encode() if isinstance(hmm.accession, (bytes, type(None))) else f"{acc}.1"
    hmm.cutoffs.gathering = (ga, ga)
    with open(path, "wb") as fh:
        hmm.write(fh)


def test_pfam_end_to_end(tmp_path, monkeypatch):
    d = tmp_path / "demo"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo.py"), "--out", str(d),
                    "--genera", "4", "--per-genus", "4", "--length", "30000", "--reads", "100"], check=True)
    monkeypatch.chdir(tmp_path)
    g = pd.read_csv(d / "genomes.tsv", sep="\t")
    first = g.iloc[0]
    prots = call_proteins(d / first.genome_fasta)  # manifest paths are relative to the manifest
    assert prots and all("*" not in p for p in prots)
    seed = max(prots, key=len)
    _toy_pfam(seed, tmp_path / "toy.hmm")

    # split across two array tasks; together they annotate every genome once
    for t in ("0", "1"):
        main(["pfam-annotate", "--manifest", str(d / "genomes.tsv"), "--hmm", "toy.hmm",
              "--cache-dir", "pf", "--task-id", t, "--n-tasks", "2", "--batch", "3"])
    caches = sorted(Path("pf").glob("*.tsv"))
    assert len(caches) == g.genome_id.nunique()
    n_prot, counts, err = read_cache(Path("pf") / f"{first.genome_id}.tsv")
    assert err == "" and n_prot == len(prots) and counts.get("PF99999", 0) >= 1

    X, stats = pfam_table(g, "pf", value="count")
    assert "f_PF99999" in X.columns and (stats.error == "").all()
    main(["pfam-table", "--manifest", str(d / "genomes.tsv"), "--cache-dir", "pf", "--out", "pfam.tsv"])
    t = pd.read_csv("pfam.tsv", sep="\t")
    assert set(t["f_PF99999"]) <= {0, 1} and t.loc[t.genome_id == first.genome_id, "f_PF99999"].item() == 1

    # hybrid: k-mer table + Pfam table joined as training features
    import json
    main(["features", "--manifest", str(d / "genomes.tsv"), "--out", "kmer.tsv.gz"])
    main(["train", "--manifest", str(d / "genomes.tsv"), "--out", "m_hybrid", "--tasks", "oxygen",
          "--embeddings", "kmer.tsv.gz", "pfam.tsv"])
    rep = json.load(open("m_hybrid/training_report.json"))
    n_kmer = pd.read_csv("kmer.tsv.gz", sep="\t", nrows=1).shape[1] - 1
    assert rep["feature_type"] == "embedding:kmer.tsv.gz+pfam.tsv"
    assert rep["n_features"] == n_kmer + t.shape[1] - 1
    with pytest.raises(SystemExit):  # the same table twice = clashing columns
        main(["train", "--manifest", str(d / "genomes.tsv"), "--out", "m_bad", "--tasks", "oxygen",
              "--embeddings", "pfam.tsv", "pfam.tsv"])

    # missing caches are refused unless --allow-missing
    (Path("pf") / f"{first.genome_id}.tsv").unlink()
    with pytest.raises(SystemExit):
        main(["pfam-table", "--manifest", str(d / "genomes.tsv"), "--cache-dir", "pf", "--out", "p2.tsv"])
    main(["pfam-table", "--manifest", str(d / "genomes.tsv"), "--cache-dir", "pf", "--out", "p2.tsv",
          "--allow-missing"])
    assert len(pd.read_csv("p2.tsv", sep="\t")) == g.genome_id.nunique() - 1
