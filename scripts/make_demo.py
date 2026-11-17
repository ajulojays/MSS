#!/usr/bin/env python3
"""Generate a FULLY SYNTHETIC offline demo dataset (test data only).

Every genome, species name, phenotype label, abundance profile and read in demo_generated/
is randomly generated. None of it describes a real organism. Names start with "Synthgenus"
and every table carries source=SYNTHETIC_TEST_DATA.

Signal design (so the pipeline has something learnable):
* oxygen class  -> class-specific short motif enrichment
* Gram stain    -> GC content shift
* motility, sporulation, cell shape -> random labels (no signal; expect chance-level CV)
* numeric traits (optimum temperature / pH / NaCl) -> random synthetic numbers
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

SRC = "SYNTHETIC_TEST_DATA"
CLASSES = ["anaerobe", "facultative", "aerobe"]
MOTIF = {"anaerobe": "AATTGCA", "facultative": "GGCTAGC", "aerobe": "CCGATCG"}


def make_genome(rng, length, gc, motif, motif_rate, n_contigs):
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    seq = rng.choice(np.array(list("ACGT")), size=length, p=p)
    n_ins = int(length * motif_rate / len(motif))
    pos = rng.choice(length - len(motif), size=n_ins, replace=False)
    for q in pos:
        seq[q:q + len(motif)] = list(motif)
    for q in rng.choice(length, size=max(1, length // 5000), replace=False):
        seq[q] = "N"  # a few ambiguous bases
    s = "".join(seq)
    cuts = np.sort(rng.choice(np.arange(1000, length - 1000), size=n_contigs - 1, replace=False))
    return [s[a:b] for a, b in zip(np.r_[0, cuts], np.r_[cuts, length])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="demo_generated")
    ap.add_argument("--genera", type=int, default=6)
    ap.add_argument("--per-genus", type=int, default=5)
    ap.add_argument("--length", type=int, default=60000)
    ap.add_argument("--reads", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "genomes").mkdir(parents=True, exist_ok=True)

    rows, seqs = [], {}
    for gi in range(a.genera):
        genus = f"Synthgenus{chr(65 + gi)}"
        family = f"Synthfamily{gi // 2 + 1}"
        for j in range(a.per_genus):
            ox = CLASSES[(gi + j) % 3]
            gram = "positive" if rng.random() < 0.5 else "negative"
            gc = 0.36 if gram == "positive" else 0.52
            gc += rng.normal(0, 0.015)
            gid = f"SYN{gi:02d}{j:02d}"
            species = f"{genus} synthspecies{j + 1}"
            contigs = make_genome(rng, a.length, gc, MOTIF[ox], 0.03, n_contigs=3)
            fa = out / "genomes" / f"{gid}.fna.gz"
            with gzip.open(fa, "wt") as fh:
                for c, s in enumerate(contigs):
                    fh.write(f">{gid}_contig{c + 1} synthetic test sequence\n")
                    for k in range(0, len(s), 80):
                        fh.write(s[k:k + 80] + "\n")
            seqs[gid] = "".join(contigs)
            rows.append({"genome_id": gid, "species": species, "genus": genus, "family": family,
                         "genome_fasta": f"genomes/{gid}.fna.gz", "oxygen": ox,
                         "motility": rng.choice(["yes", "no"]), "sporulation": rng.choice(["yes", "no"]),
                         "gram_stain": gram,
                         "cell_shape": rng.choice(["synthetic-rod", "synthetic-coccus"]),
                         "optimal_temperature_c": float(rng.choice([25, 30, 37, 42])),
                         "optimal_ph": float(rng.choice([6.5, 7.0, 7.5])),
                         "optimal_nacl_pct": float(rng.choice([0.5, 1.0, 3.5])),
                         "source": SRC})
    man = pd.DataFrame(rows)
    # One missing label and one conflicting label to exercise the filters.
    man.loc[1, "oxygen"] = ""
    man["oxygen_conflict"] = False
    man.loc[2, "oxygen_conflict"] = True
    man.loc[2, "oxygen"] = ""
    man.to_csv(out / "genomes.tsv", sep="\t", index=False)

    # Phenotype reference (bacdive_reference.tsv layout) for part of the species.
    ref = man[["species", "genus", "family", "oxygen", "oxygen_conflict", "motility",
               "sporulation", "gram_stain", "cell_shape", "optimal_temperature_c", "optimal_ph",
               "optimal_nacl_pct"]].copy()
    ref.insert(0, "bacdive_id", [f"SYNTH-{i}" for i in range(len(ref))])
    ref["oxygen_detail"] = ref["oxygen"].map(lambda o: f"{o}:unspecified" if o else "")
    ref["oxygen_raw"] = ref["oxygen"]
    ref["evidence_references"] = "synthetic"
    ref["genome_accession"] = man["genome_id"]
    ref["source"] = SRC
    ref = ref.iloc[::2].copy()  # every other species is "unannotated" for MSS-Taxa
    # A species with two strains that disagree -> species-level conflict (no majority vote).
    extra = ref.iloc[[0]].copy()
    extra["bacdive_id"] = "SYNTH-extra"
    extra["oxygen"] = "aerobe" if ref.iloc[0]["oxygen"] != "aerobe" else "anaerobe"
    ref = pd.concat([ref, extra], ignore_index=True)
    ref.to_csv(out / "phenotypes.tsv", sep="\t", index=False)

    # Taxonomic profile: 4 samples, percentages, plus one taxon absent from everything.
    prof = []
    for s in range(4):
        pick = rng.choice(len(man), size=8, replace=False)
        ab = rng.dirichlet(np.ones(8)) * 100
        for i, w in zip(pick, ab):
            prof.append({"sample_id": f"synthetic_sample_{s + 1}", "species": man.species[i],
                         "relative_abundance": round(float(w) * 0.95, 4)})
        prof.append({"sample_id": f"synthetic_sample_{s + 1}", "species": "Unlistedgenus synthspecies0",
                     "relative_abundance": 5.0})
    pd.DataFrame(prof).to_csv(out / "profile.tsv", sep="\t", index=False)

    # Reads: 150 bp error-free fragments from a known DNA-fraction mixture.
    labeled = man[man.oxygen != ""].reset_index(drop=True)
    pick = rng.choice(len(labeled), size=6, replace=False)
    w = rng.dirichlet(np.ones(6))
    truth = {c: 0.0 for c in CLASSES}
    for i, wi in zip(pick, w):
        truth[labeled.oxygen[i]] += wi
    src = rng.choice(pick, size=a.reads, p=w)
    with open(out / "reads.fastq", "w") as fh:
        for r, gi in enumerate(src):
            s = seqs[labeled.genome_id[gi]]
            st = rng.integers(0, len(s) - 150)
            read = s[st:st + 150]
            if rng.random() < 0.5:
                read = read[::-1].translate(str.maketrans("ACGTN", "TGCAN"))
            fh.write(f"@synthetic_read_{r}\n{read}\n+\n{'I' * 150}\n")
    pd.DataFrame([{"sample_id": "reads", **{f"{c}_pct": 100 * v for c, v in truth.items()},
                   "note": "DNA-fraction truth of the synthetic read mixture", "source": SRC}]
                 ).to_csv(out / "reads_truth.tsv", sep="\t", index=False)
    (out / "README.txt").write_text(
        "ALL FILES IN THIS DIRECTORY ARE SYNTHETIC TEST DATA generated by scripts/make_demo.py.\n"
        "They describe no real organism and must not be used as biological reference data.\n")
    print(f"synthetic demo written to {out}/ ({len(man)} genomes, {a.reads} reads)")


if __name__ == "__main__":
    main()
