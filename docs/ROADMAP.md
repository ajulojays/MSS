# MSS roadmap

Status labels: **Implemented** = exists and is unit-tested · **Experimental** = runs, no
independent validation · **Planned** = not implemented.

## V1 (this release)

| Component | Status |
|---|---|
| BacDive v2 client, phenotype extraction, conflict/evidence preservation | Implemented |
| Genome manifest + NCBI Datasets download + FASTA validation | Implemented |
| Canonical 5-mer genome features (streaming, gzip, cached, multi-process) | Implemented |
| External embedding table loader (`dna_*`, `protein_*`, `f_*`) | Implemented |
| Logistic-regression phenotype heads, leave-genus/family-out CV | Implemented (real-genome accuracy: not yet benchmarked) |
| MSS-Taxa composition + audit | Implemented |
| MSS-Raw k-mer Ridge on synthetic mixtures | Experimental |
| SLURM submit / pipeline / array wrappers, optional GPU queue | Implemented |

## Next: data and evaluation

1. Build a large BacDive reference. Review the unmapped oxygen strings and decide on them
   explicitly (`--oxygen-map`).
2. Species matching through NCBI taxids / GTDB identifiers instead of exact names.
3. Near-duplicate control by ANI/Mash clustering in addition to genus grouping.
4. Benchmarks:
   1. genome phenotype accuracy (macro-F1, balanced accuracy)
   2. held-out genus *and* family
   3. multiclass composition reconstruction (MAE)
   4. facultative abundance
   5. unknown/poorly characterized taxa
   6. calibration (Brier score, reliability curves)
   7. low-depth robustness
   8. runtime and memory, from the `/usr/bin/time -v` logs in the SLURM wrappers
5. Comparators: OxyMetaG (marker-based % aerobe), sumTraits/metaTraits (taxonomy-based traits),
   and existing genome phenotype predictors. Compare only like with like: a marker score is not
   a measured obligate-anaerobe abundance.

## Planned: foundation-model representations

* **Genome models**: DNA foundation-model embeddings of genomes or regions; test whether they
  beat k-mers under held-out-genus CV.
* **Protein models**: ESM2 (or similar) protein embeddings, with whole-proteome aggregation;
  orthologous-group / FastNOG functional features.
* **Multimodal**: genome-only vs protein-only vs function-only vs combined → a shared phenotype
  embedding → compact per-trait heads. The heads take a feature table, so MSS-Taxa and MSS-Raw
  outputs do not change.
* **GPU execution**: `slurm/run_pipeline.sh --embed-cmd ... --embed-out ...` already schedules
  an external GPU embedding job and trains on its output. The embedding code itself is planned.

## Planned: metagenome student (MSS-Raw v2)

* Teacher = the genome/protein phenotype predictor. Student = a lightweight read-level encoder.
* Train on synthetic communities built from realistic read simulation (errors, GC/coverage bias,
  unlabeled taxa, host DNA). Loss = composition reconstruction + teacher–student alignment +
  calibration.
* Claim a benefit only if alignment improves accuracy on independent test communities.
* Speed: distillation, feature compression, quantization, batched CPU inference. Never run large
  foundation models on each new sample.
