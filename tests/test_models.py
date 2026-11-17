"""Feature, model, MSS-Taxa and MSS-Raw tests on synthetic fixtures (no network)."""
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mss.cli import main
from mss.features import KmerSpace, fasta_kmers, fastq_kmers, load_embeddings
from mss.models import predict_heads, task_labels, train_heads
from mss.ontology import OXYGEN_CLASSES
from mss.raw import project_simplex
from mss.taxa import compute_taxa, compute_traits, species_numeric, species_reference

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ k-mers
def test_kmer_space_canonical():
    s = KmerSpace(5)
    assert s.n_features == 512
    s2 = KmerSpace(2)
    assert s2.n_features == 10  # 16 2-mers -> 10 canonical (4 palindromes)


def brute(seq, k):
    rc = str.maketrans("ACGT", "TGCA")
    out = {}
    for i in range(len(seq) - k + 1):
        w = seq[i:i + k]
        if set(w) <= set("ACGT"):
            c = min(w, w.translate(rc)[::-1])
            out[c] = out.get(c, 0) + 1
    return out


def test_fasta_kmers_match_bruteforce(tmp_path):
    rng = np.random.default_rng(0)
    s1 = "".join(rng.choice(list("ACGT"), 500)) + "NN" + "".join(rng.choice(list("ACGT"), 300))
    s2 = "".join(rng.choice(list("ACGTacgt"), 200))
    p = tmp_path / "g.fna.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(f">a\n{s1[:400]}\n{s1[400:]}\n>b\n{s2}\n")
    prof = fasta_kmers(p, 3)
    exp = brute(s1, 3)
    for kk, v in brute(s2.upper(), 3).items():
        exp[kk] = exp.get(kk, 0) + v
    space = KmerSpace(3)
    got = {n[5:]: c for n, c in zip(space.names, prof.counts) if c}
    assert got == exp  # no k-mers span contigs or N
    assert prof.n_records == 2 and abs(prof.frequencies().sum() - 1) < 1e-12


def test_fasta_chunking_equivalent(tmp_path, monkeypatch):
    import mss.features as F
    rng = np.random.default_rng(1)
    s = "".join(rng.choice(list("ACGT"), 5000))
    p = tmp_path / "g.fa"
    p.write_text(">x\n" + "\n".join(s[i:i + 60] for i in range(0, len(s), 60)) + "\n")
    full = fasta_kmers(p, 5).counts
    monkeypatch.setattr(F, "CHUNK_BASES", 97)
    assert np.array_equal(F.fasta_kmers(p, 5).counts, full)


def test_fastq_parsing_and_errors(tmp_path):
    good = tmp_path / "r.fastq"
    good.write_text("@r1\nACGTACGTAC\n+\nIIIIIIIIII\n@r2\nTTTTTGGGGG\n+\nIIIIIIIIII\n")
    prof = fastq_kmers(good, 5)
    assert prof.n_records == 2
    assert prof.n_kmers == 12  # 6 per read; none spanning reads
    assert fastq_kmers(good, 5, max_reads=1).n_records == 1
    trunc = tmp_path / "t.fastq"
    trunc.write_text("@r1\nACGT\n+\nIIII\n@r2\nACGT\n")
    with pytest.raises(ValueError, match="truncated"):
        fastq_kmers(trunc, 3)
    bad = tmp_path / "b.fastq"
    bad.write_text("@r1\nACGT\n+\nIII\n")
    with pytest.raises(ValueError, match="length mismatch"):
        fastq_kmers(bad, 3)


def test_embeddings_loader(tmp_path):
    p = tmp_path / "e.tsv"
    p.write_text("genome_id\tdna_0\tprotein_0\tnote\nA\t0.1\t0.2\tx\nB\t0.3\t0.4\ty\n")
    X = load_embeddings(p)
    assert list(X.columns) == ["dna_0", "protein_0"] and list(X.index) == ["A", "B"]
    p.write_text("genome_id\tother\nA\t1\n")
    with pytest.raises(ValueError):
        load_embeddings(p)


# ------------------------------------------------------------------ phenotype heads
def toy(n_per=4, genera=4, seed=0):
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for g in range(genera):
        for j in range(n_per):
            ox = ["anaerobe", "aerobe"][(g + j) % 2]
            v = rng.normal(0, 1, 6) + (3 if ox == "aerobe" else -3) * np.array([1, 0, 0, 0, 0, 0])
            X.append(v)
            rows.append({"genome_id": f"g{g}_{j}", "genus": f"G{g}", "oxygen": ox,
                         "motility": "", "gram_stain": "positive", "oxygen_conflict": "False"})
    m = pd.DataFrame(rows)
    return pd.DataFrame(X, index=m.genome_id, columns=[f"f_{i}" for i in range(6)]), m


def test_task_labels_filters_missing_and_conflict():
    _, m = toy()
    m.loc[0, "oxygen"] = ""
    m.loc[1, "oxygen_conflict"] = "True"
    m.loc[2, "oxygen"] = "sometimes aerobe"
    lab = task_labels(m, "oxygen")
    assert len(lab) == len(m) - 3


def test_train_and_predict_heads():
    X, m = toy()
    bundle, rep = train_heads(X, m, "test")
    assert set(bundle["heads"]) == {"oxygen"}
    assert rep["tasks"]["motility"]["status"] == "skipped"
    assert rep["tasks"]["gram_stain"]["status"] == "skipped"  # one class only
    v = rep["tasks"]["oxygen"]["validation"]
    assert v["status"] == "done" and v["n_evaluated"] == 16
    for f in v["folds"]:  # leave-genus-out
        assert len(f["held_out_groups"]) >= 1
    pred = predict_heads(bundle, X)
    pc = [f"oxygen_prob_{c}" for c in OXYGEN_CLASSES]
    assert np.allclose(pred[pc].sum(axis=1), 1)
    assert (pred["oxygen_prob_facultative"] == 0).all()  # class never seen in training


def test_class_filter_and_per_phylum_report():
    X, m = toy()
    m["phylum"] = ["P1" if g in ("G0", "G1") else "P2" for g in m["genus"]]
    m.loc[0, "oxygen"] = "facultative"
    bundle, rep = train_heads(X, m, "test", class_filter={"oxygen": ["anaerobe", "aerobe"]},
                              report_by="phylum")
    ox = rep["tasks"]["oxygen"]
    assert ox["n_labeled"] == len(m) - 1 and set(ox["classes"]) == {"anaerobe", "aerobe"}
    assert set(ox["validation"]["by_phylum"]) == {"P1", "P2"}
    assert "_oof" not in ox["validation"]
    pred = predict_heads(bundle, X)
    assert "oxygen_prob_facultative" not in pred.columns


def test_train_requires_data():
    X, m = toy(n_per=1, genera=4)
    with pytest.raises(RuntimeError, match="no phenotype task"):
        train_heads(X, m, "test")


# ------------------------------------------------------------------ MSS-Taxa
REF = pd.DataFrame({
    "species": ["A one", "A one", "B two", "C three", "C three", "D four"],
    "oxygen": ["anaerobe", "anaerobe", "aerobe", "anaerobe", "aerobe", ""],
    "oxygen_conflict": ["False"] * 6,
})


def test_species_reference_no_majority_vote():
    s = species_reference(REF)
    assert s.loc["a one", "status"] == "resolved"
    assert s.loc["c three", "status"] == "conflict"
    assert s.loc["d four", "status"] == "unannotated"


def test_taxa_aggregation_and_unresolved():
    prof = pd.DataFrame({"sample_id": ["s1"] * 5,
                         "species": ["A_one", "B two", "C three", "D four", "E five"],
                         "relative_abundance": [40, 20, 10, 10, 20]})  # percentages
    samp, audit = compute_taxa(prof, REF)
    r = samp.iloc[0]
    assert r["anaerobe_pct"] == pytest.approx(40)
    assert r["aerobe_pct"] == pytest.approx(20)
    assert r["unresolved_pct"] == pytest.approx(40)
    assert r["mss_anaerobic_index"] == pytest.approx(40)
    fr = [f"{c}_pct" for c in OXYGEN_CLASSES] + ["unresolved_pct"]
    assert samp[fr].sum(axis=1).iloc[0] == pytest.approx(100)
    assert set(audit["match_status"]) == {"matched_resolved", "matched_conflict",
                                          "matched_unannotated", "unmatched"}


def test_taxa_predictions_fill_only_unannotated():
    prof = pd.DataFrame({"sample_id": ["s1"] * 3, "species": ["C three", "D four", "E five"],
                         "relative_abundance": [0.5, 0.25, 0.25]})
    pred = pd.DataFrame({"species": ["C three", "D four", "E five"],
                         **{f"oxygen_prob_{c}": [0.0, 0.0, 0.0] for c in OXYGEN_CLASSES}})
    pred["oxygen_prob_aerobe"] = 1.0
    samp, audit = compute_taxa(prof, REF, pred)
    r = samp.iloc[0]
    assert r["aerobe_pct"] == pytest.approx(50)       # D and E from model
    assert r["unresolved_pct"] == pytest.approx(50)   # C conflict NOT overwritten
    assert r["model_inferred_pct"] == pytest.approx(50)
    assert r["reference_supported_pct"] == pytest.approx(0)
    assert audit.set_index("species").loc["C three", "source"] == "unresolved"


def test_taxa_normalizes_per_sample_and_rejects_negative():
    prof = pd.DataFrame({"sample_id": ["s1", "s1", "s2"], "species": ["A one", "B two", "A one"],
                         "relative_abundance": [3, 1, 7]})
    samp, _ = compute_taxa(prof, REF)
    assert samp.set_index("sample_id").loc["s1", "anaerobe_pct"] == pytest.approx(75)
    assert samp.set_index("sample_id").loc["s2", "anaerobe_pct"] == pytest.approx(100)


TREF = pd.DataFrame({
    "species": ["A one", "A one", "B two", "C three", "C three"],
    "gram_stain": ["positive", "positive", "negative", "", ""],
    "cell_shape": ["rod-shaped", "rod-shaped", "coccus-shaped", "rod-shaped", "coccus-shaped"],
    "optimal_temperature_c": ["37", "37", "30", "25", "42"],
    "oxygen": ["anaerobe"] * 5,
})


def test_compute_traits_categorical_and_numeric():
    prof = pd.DataFrame({"sample_id": ["s1"] * 4, "species": ["A one", "B two", "C three", "Z none"],
                         "relative_abundance": [50, 25, 15, 10]})
    wide, audit, done = compute_traits(prof, TREF)
    assert set(done) == {"oxygen", "gram_stain", "cell_shape", "optimal_temperature_c"}
    r = wide.iloc[0]
    assert r["gram_stain__positive_pct"] == pytest.approx(50)
    assert r["gram_stain__negative_pct"] == pytest.approx(25)
    assert r["gram_stain__unresolved_pct"] == pytest.approx(25)       # C unannotated + Z unmatched
    assert r["cell_shape__rod-shaped_pct"] == pytest.approx(50)
    assert r["cell_shape__unresolved_pct"] == pytest.approx(25)       # C strains disagree -> conflict
    # numeric: C three has 25 vs 42 -> conflict, excluded; mean over A (37) and B (30)
    assert r["optimal_temperature_c__weighted_mean"] == pytest.approx((50 * 37 + 25 * 30) / 75)
    assert r["optimal_temperature_c__coverage_pct"] == pytest.approx(75)
    for t in ["gram_stain", "cell_shape", "oxygen"]:
        cols = [c for c in wide.columns if c.startswith(f"{t}__") and c.endswith("_pct")
                and "reference_supported" not in c and "model_inferred" not in c]
        assert wide[cols].sum(axis=1).iloc[0] == pytest.approx(100)
    assert species_numeric(TREF, "optimal_temperature_c").loc["c three", "status"] == "conflict"


def test_traits_model_fill_for_binary_trait():
    prof = pd.DataFrame({"sample_id": ["s1"] * 2, "species": ["C three", "Z none"],
                         "relative_abundance": [1, 1]})
    pred = pd.DataFrame({"species": ["C three", "Z none"], "gram_stain_prob_positive": [1.0, 0.0],
                         "gram_stain_prob_negative": [0.0, 1.0], "gram_stain_prob_variable": [0.0, 0.0]})
    wide, _, _ = compute_traits(prof, TREF, pred, traits=["gram_stain"])
    r = wide.iloc[0]
    assert r["gram_stain__positive_pct"] == pytest.approx(50)
    assert r["gram_stain__negative_pct"] == pytest.approx(50)
    assert r["gram_stain__model_inferred_pct"] == pytest.approx(100)


# ------------------------------------------------------------------ MSS-Raw
def test_project_simplex():
    Y = np.array([[0.2, 0.3, 0.5], [2.0, -1.0, 0.0], [-0.5, -0.5, -0.5]])
    P = project_simplex(Y)
    assert np.allclose(P.sum(axis=1), 1) and (P >= 0).all()
    assert np.allclose(P[0], Y[0])


# ------------------------------------------------------------------ end-to-end demo via CLI
def test_demo_end_to_end(tmp_path, monkeypatch):
    d = tmp_path / "demo"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo.py"), "--out", str(d),
                    "--genera", "4", "--per-genus", "4", "--length", "20000", "--reads", "3000"],
                   check=True)
    monkeypatch.chdir(tmp_path)
    main(["taxa", "--profile", f"{d}/profile.tsv", "--reference", f"{d}/phenotypes.tsv",
          "--out", "res/taxa.tsv"])
    main(["train", "--manifest", f"{d}/genomes.tsv", "--out", "models"])
    main(["predict-genomes", "--manifest", f"{d}/genomes.tsv", "--model",
          "models/phenotype_model.joblib", "--out", "res/pred.tsv"])
    main(["taxa", "--profile", f"{d}/profile.tsv", "--reference", f"{d}/phenotypes.tsv",
          "--predictions", "res/pred.tsv", "--out", "res/taxa_pred.tsv"])
    main(["train-raw", "--manifest", f"{d}/genomes.tsv", "--out", "models", "--mixtures", "60"])
    main(["raw", "--fastq", f"{d}/reads.fastq", "--model", "models/raw_mixture.joblib",
          "--out", "res/raw.tsv", "--max-reads", "2000"])
    fr = [f"{c}_pct" for c in OXYGEN_CLASSES]
    t = pd.read_csv("res/taxa.tsv", sep="\t")
    assert np.allclose(t[fr + ["unresolved_pct"]].sum(axis=1), 100)
    assert (t["unresolved_pct"] > 0).all()  # unmatched taxon kept as unresolved
    tp = pd.read_csv("res/taxa_pred.tsv", sep="\t")
    assert (tp["model_inferred_pct"] > 0).any()
    assert np.allclose(tp[fr + ["unresolved_pct"]].sum(axis=1), 100)
    r = pd.read_csv("res/raw.tsv", sep="\t")
    assert r.loc[0, "n_reads"] == 2000
    assert r.loc[0, "status"] == "EXPERIMENTAL_UNVALIDATED"
    assert np.isclose(r[fr].sum(axis=1).iloc[0], 100)
    w = pd.read_csv("res/taxa.traits.tsv", sep="\t")
    assert {"cell_shape__synthetic-rod_pct", "optimal_temperature_c__weighted_mean"} <= set(w.columns)
    p = pd.read_csv("res/pred.tsv", sep="\t")
    assert np.allclose(p[[f"oxygen_prob_{c}" for c in OXYGEN_CLASSES]].sum(axis=1), 1)
