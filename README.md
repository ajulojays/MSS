# MSS — Microbiome State Score

MSS is a research prototype that links experimentally characterized microbial phenotypes
(from [BacDive](https://bacdive.dsmz.de/)) to genome features, and then estimates the
**oxygen-tolerance composition of microbial communities**. It has two pathways:

| Pathway | Input | Status |
|---|---|---|
| **MSS-Taxa** | species relative-abundance profile + phenotype reference (+ optional genome-model predictions) | Implemented, tested |
| **MSS-Raw** | shotgun FASTQ (QC'd, host-removed) | **Experimental, unvalidated** |

The headline number is the **MSS anaerobic index** = 100 × the abundance-weighted fraction of
the `anaerobe` category. In V1, `anaerobe` includes generic "anaerobe" labels whose strict
oxygen requirement is not stated, so the index is **not** an established obligate-anaerobe
fraction. It is not a measurement of intestinal oxygen, and it is not clinically validated.

MSS was first developed under the working name MAIC (Microbiome Anaerobic Index Calculator).

### Phenotypes in V1

| Phenotype | BacDive source | MSS-Taxa output (`<out>.traits.tsv`) | Genome head |
|---|---|---|---|
| Oxygen tolerance | `oxygen tolerance` | anaerobe / facultative / aerobe / microaerophile / aerotolerant %, anaerobic index | yes |
| Gram stain | `cell morphology › gram stain` | positive / negative / variable % | yes |
| Motility | `cell morphology › motility` | motile (`yes`) vs non-motile % | yes |
| Spore formation | `spore formation` | spore-forming (`yes`) vs not % | yes |
| Cell morphology | `cell morphology › cell shape` | % per BacDive shape string (`rod-shaped`, `coccus-shaped`, `oval-shaped`, `ovoid-shaped`, …; not merged) | yes (classes from data) |
| Growth temperature | `culture temp` (optimum; positive-growth tests) | weighted mean of optimum, min and max observed growth temperature, + coverage % | no |
| Growth pH | `culture pH` (optimum; positive growth; BacDive `PH range` label) | weighted means + coverage; % alkaliphile / acidophile label | no |
| Salt tolerance | `halophily` (NaCl) | weighted mean NaCl optimum and max NaCl with observed growth (%), + coverage | no |

Numeric traits are reported as abundance-weighted means over taxa that have a resolved value,
together with the abundance those taxa cover. They are **not binned** (no mesophile/thermophile
or halophile cut-offs are applied). BacDive v2 records give no categorical temperature class,
and any threshold would be an analysis choice to make explicitly. Observed growth ranges are
the span of positive growth tests. A range contradicted by a negative test inside it is left
empty and flagged (`growth_*_conflict`). NaCl values are parsed only when given as plain `%`.
Other units, such as `g/L` or `%(w/v)`, are kept raw and not converted.

---

## Evidence levels

| Level | What |
|---|---|
| **Implemented** (runs, unit-tested) | BacDive v2 client (retries, 429/Retry-After, pagination, JSON cache); phenotype extraction with conflict and evidence preservation; genome manifest; NCBI Datasets download with per-accession failure log; streaming canonical 5-mer features (FASTA/FASTQ, gzip); cached genome features; logistic-regression phenotype heads with leave-genus-out CV; MSS-Taxa; external embedding loader; SLURM wrappers |
| **Experimental** (runs, no independent scientific validation) | MSS-Raw synthetic-mixture Ridge model; all phenotype-head accuracy on real genomes (not benchmarked yet) |
| **Planned** | Genome/protein foundation-model *inference* (Evo2, ESM2, BacPT, FastNOG features), multimodal heads, metagenome teacher–student distillation, benchmarks against OxyMetaG and sumTraits/metaTraits. See [docs/ROADMAP.md](docs/ROADMAP.md) |

MSS does **not** run any foundation model. It *accepts* precomputed embeddings
(`dna_*`, `protein_*`, `f_*` columns) and trains its heads on them.

---

## Install

```bash
python3 -m venv .venv            # Python >= 3.10
source .venv/bin/activate
pip install -e ".[dev]"
```

Or with conda (as on UNC Longleaf):

```bash
conda create -p $HOME/envs/mss -c conda-forge -c bioconda python=3.11 numpy pandas scikit-learn joblib requests pytest pip ncbi-datasets-cli
$HOME/envs/mss/bin/pip install -e ".[dev]"
```

`ncbi-datasets-cli` is only needed for `mss download-genomes`.

---

## Offline synthetic demo (no downloads)

Everything in `demo_generated/` is **synthetic test data**. Nothing in it describes a real organism.

```bash
python scripts/make_demo.py
mss taxa --profile demo_generated/profile.tsv --reference demo_generated/phenotypes.tsv --out results/demo_taxa.tsv
mss train --manifest demo_generated/genomes.tsv --out models/demo
mss predict-genomes --manifest demo_generated/genomes.tsv --model models/demo/phenotype_model.joblib --out results/demo_genome_predictions.tsv
mss train-raw --manifest demo_generated/genomes.tsv --out models/demo --mixtures 250
mss raw --fastq demo_generated/reads.fastq --model models/demo/raw_mixture.joblib --out results/demo_raw.tsv
python -m pytest -q
```

The demo plants a learnable signal for oxygen class and Gram stain. Motility and sporulation
are random labels, so their CV scores should sit near chance. That is a check that the
validation does not leak.

---

## Commands

`mss <command> --help` documents every option. **[internet]** commands need network access;
all others run offline.

| Command | Purpose |
|---|---|
| `mss bacdive` **[internet]** | `--taxon "Bacteroides fragilis" Clostridium`, `--taxa-file`, `--ids`, `--ids-file` → `data/bacdive_reference.tsv` + `.summary.json`; raw JSON in `data/bacdive_cache/` |
| `mss genome-manifest` | reference → one row per unique genome accession |
| `mss download-genomes` **[internet]** | NCBI Datasets → `genomes/<acc>.fna.gz`, `data/labeled_genomes.tsv`, `*.failures.tsv` |
| `mss features` | precompute / cache canonical k-mer features (`--threads`) |
| `mss train` | phenotype heads → `models/phenotype_model.joblib`, `training_report.json` |
| `mss predict-genomes` | per-genome class probabilities (`--manifest` or `--fasta`) |
| `mss taxa` | MSS-Taxa composition + per-taxon audit table |
| `mss train-raw` | EXPERIMENTAL synthetic-mixture model → `raw_mixture.joblib` |
| `mss raw` | EXPERIMENTAL FASTQ → composition (`--fastq R1 R2`, or `--sample-sheet` + `--array-index env`) |

### Real-data example (internet)

```bash
mss bacdive --taxa-file taxa.txt --out data/bacdive_reference.tsv
mss genome-manifest --reference data/bacdive_reference.tsv --out data/genome_manifest.tsv
mss download-genomes --manifest data/genome_manifest.tsv --outdir genomes --out data/labeled_genomes.tsv
mss train --manifest data/labeled_genomes.tsv --out models --threads 16
```

---

## Running on SLURM (UNC Longleaf and other clusters)

```bash
cp slurm/config.example.sh slurm/config.sh      # set MSS_ENV to your env prefix
```

* `slurm/submit.sh [--gpu] [--cpus N] [--mem 32G] [--time 12:00:00] [--after JOBID] -- <mss args>`
  submits any single step. `--cmd "<shell>"` runs any other command, such as an external
  embedding script. `--gpu` adds `--partition=$MSS_GPU_PARTITION --qos=gpu_access --gres=gpu:1`
  (Longleaf defaults: `l40-gpu`, `gpu_access`).
* `slurm/run_pipeline.sh --taxa-file taxa.txt --workdir run1 --threads 16` chains
  bacdive → manifest → download → features → train → predict → train-raw with `afterok`
  dependencies.
* `slurm/raw_array.sh --sample-sheet samples.tsv --model models/raw_mixture.joblib --outdir results/raw`
  runs MSS-Raw as an array (one sample per task) and merges the results into one table.
* `sbatch slurm/demo_and_tests.sbatch` runs the synthetic demo and the tests on a compute node.

**Genome selection (default in `run_pipeline.sh`), following OxyMetaG.** The OxyMetaG authors
(Bueno de Mesquita, Stallard-Olivera & Fierer 2026; code in `cliffbueno/Oxygen`,
`code/BacDive.R`) matched BacDive genome accessions to GTDB r226 and kept only **GTDB species
representatives**, one genome per GTDB species cluster. MSS does the same:
- `mss genome-manifest --gtdb-metadata bac120_metadata_r226.tsv.gz ar53_metadata_r226.tsv.gz --gtdb-reps-only`
- only strains with at least one measured label are kept (`--require-label`);
- if any of a strain's BacDive accessions is the GTDB representative, that genome is used;
- several BacDive strains on one genome have their labels merged without voting
  (disagreement → conflict);
- GTDB taxonomy and CheckM2 columns are added;
- validation holds out GTDB genera (`--cv-group gtdb_genus`), and the report gives
  out-of-fold accuracy per GTDB phylum (`--report-by gtdb_phylum`).

OxyMetaG trained only on clear aerobes and anaerobes. MSS keeps all five classes by default and
also trains that clear-label head (`--oxygen-classes anaerobe aerobe`, written to `models_clear/`).
MSS does not use OxyMetaG's random 80:20 split: holding out whole genera is the stricter test.

**When is a GPU used?** No MSS V1 step needs one: k-mer counting, logistic regression and
Ridge all run on CPU. Request `--gpu` only for an external foundation-model embedding job.
`run_pipeline.sh --embed-cmd "<cmd>" --embed-out emb.tsv` submits that job to a GPU node and
then trains on its output. MSS does not provide that embedding command.

Each job log records the host, CPUs, GPUs, and `/usr/bin/time -v` peak memory and elapsed time.

---

## Evidence rules (scientific conservatism)

* **Missing ≠ negative.** An absent BacDive field stays empty.
* **No majority vote.** If one strain's measurements disagree, the label is empty and
  `<trait>_conflict=True`, with the raw strings and `@ref` IDs kept in `oxygen_raw` and
  `evidence_references`. MSS-Taxa marks a species as `conflict` (unresolved) when its strains
  disagree, and a model prediction never overwrites it.
* **Obligate vs generic.** `oxygen_detail` records `anaerobe:obligate` only when BacDive says
  "obligate"/"strict". A generic "anaerobe" is never upgraded.
* **Predictions are not ground truth.** BacDive's `Genome-based predictions` section is ignored.
  Model probabilities enter MSS-Taxa only for species with no measured oxygen label, and the
  output reports how much of each sample they cover (`model_inferred_pct`).
* **Unmapped strings are reported, not guessed.** BacDive strings that literally name an MSS
  category (optionally with obligate/strict) are mapped. Two more are mapped by project decision:
  `facultative aerobe` → `facultative` and `microaerotolerant` → `microaerophile`. The raw
  string is always kept in `oxygen_raw`. See `mss/ontology.py`. Any other string is left empty
  and listed in `bacdive_reference.summary.json`. Override or extend the mapping with
  `--oxygen-map` (template: `examples/oxygen_map_template.tsv`; map a term to `unmapped` to undo a default).
* **Unmatched taxa are kept** as `unresolved_pct`. Each sample's categories plus unresolved sum to 100%.
* **Species matching** is normalized exact-name matching (case, `s__` prefix, `_`/space).
  GTDB suffixes are kept (`Blautia_A` ≠ `Blautia`). The per-taxon audit table shows every
  match. Mapping through NCBI/GTDB identifiers is planned.
* **Validation**: phenotype heads report only leave-genus-out (`--cv-group family` for
  leave-family-out) balanced accuracy and macro-F1, never the training-set fit. Identical
  genome files are dropped at download. Grouping by genus keeps near-identical strains of a
  genus on the same side of every split.
* **Relative-abundance weighting is not a cell count.** MSS-Raw targets are DNA-fraction-weighted.

## MSS-Raw limitations (read before using)

Training mixtures are built by multinomial sampling of genome k-mer profiles under Dirichlet
DNA-fraction weights. This is **not** a read simulator: it models no sequencing error, GC
bias, coverage bias, read length, unlabeled organisms or host DNA. The held-out-genus MAE in
`raw_training_report.json` measures fit on synthetic mixtures only. Every MSS-Raw output row
carries `status=EXPERIMENTAL_UNVALIDATED`.

## Outputs

**MSS-Taxa** (`--out`): `sample_id, mss_anaerobic_index, anaerobe_pct, facultative_pct,
aerobe_pct, microaerophile_pct, aerotolerant_pct, unresolved_pct, reference_supported_pct,
model_inferred_pct, n_taxa, n_taxa_unresolved, method`. Also `<out>.per_taxon.tsv` (audit).
`<out>.traits.tsv` has one row per sample:
- `<trait>__<category>_pct`, `<trait>__unresolved_pct`, `<trait>__reference_supported_pct` and
  `<trait>__model_inferred_pct` for categorical traits;
- `<trait>__weighted_mean`, `<trait>__coverage_pct` and `<trait>__n_taxa_with_value` for numeric traits.

`<out>.traits_per_taxon.tsv` is its audit table. Genome-model predictions (`--predictions`) fill
in any categorical trait that has a trained head, but only for species with no measured value.

**MSS-Raw**: `sample_id, n_reads, n_kmers, <class>_pct …, mss_anaerobic_index, model_id, status, seconds`.

## Benchmarking (planned, not done)

MSS has **not** been compared with OxyMetaG, sumTraits/metaTraits or other genome phenotype
predictors, and it makes no claim to beat them. An oxygen-marker score (OxyMetaG) and a
measured-phenotype composition answer different questions, so they will not be scored against
a single ground truth. See [docs/ROADMAP.md](docs/ROADMAP.md).

## License / citation

MIT (see `LICENSE`). See `CITATION.cff`. BacDive data are © DSMZ and licensed separately.
Cite BacDive when you use the reference tables.
