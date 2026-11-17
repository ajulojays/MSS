#!/bin/bash
# Real-data MSS V1 pipeline as chained SLURM jobs (afterok dependencies).
#
#   slurm/run_pipeline.sh --taxa-file taxa.txt | --ids-file ids.txt | --id-range 1 180000
#                         [--workdir run1] [--threads 16] [--all-genomes]
#   By default only genomes of strains with >=1 measured oxygen/Gram/motility/spore/shape
#   label are downloaded; --all-genomes downloads every BacDive-linked genome.
#   Genome selection follows OxyMetaG (Bueno de Mesquita et al. 2026, cliffbueno/Oxygen
#   code/BacDive.R): BacDive accessions are matched to GTDB r226 and only GTDB species
#   representatives are kept (one genome per GTDB species). --no-gtdb-reps keeps all matches.
#   Validation holds out GTDB genera and reports accuracy per GTDB phylum. A second,
#   OxyMetaG-style clear-label oxygen head (anaerobe vs aerobe) is trained into models_clear/.
#                         [--mixtures 5000] [--embed-cmd "<cmd>" --embed-out emb.tsv]
#
# Steps:   1 bacdive [internet]  -> data/bacdive_reference.tsv
#          2 genome-manifest     -> data/genome_manifest.tsv
#          3 download-genomes [internet] -> data/labeled_genomes.tsv
#          4 features (CPU, cached k-mers)
#          (4b optional GPU embedding: --embed-cmd must write --embed-out with a genome_id
#              column and dna_*/protein_*/f_* columns; MSS does not ship a foundation model)
#          5 train  6 predict-genomes  7 train-raw (EXPERIMENTAL)
# Only the optional embedding step requests a GPU; every MSS step itself is CPU-only.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAXA=""; IDS=""; RANGE=(); GTDB_DIR=""; REPS=(--gtdb-reps-only); LABELED=(--require-label oxygen gram_stain motility sporulation cell_shape); WORK=mss_run; THREADS=16; MIX=5000; EMBED_CMD=""; EMBED_OUT=""; EXTRA_BD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --taxa-file) TAXA="$(realpath "$2")"; shift 2;;
    --ids-file) IDS="$(realpath "$2")"; shift 2;;
    --id-range) RANGE=(--id-range "$2" "$3"); shift 3;;
    --oxygen-map) EXTRA_BD+=(--oxygen-map "$(realpath "$2")"); shift 2;;
    --workdir) WORK="$2"; shift 2;;
    --all-genomes) LABELED=(); shift;;
    --gtdb-dir) GTDB_DIR="$(realpath -m "$2")"; shift 2;;
    --no-gtdb-reps) REPS=(); shift;;
    --threads) THREADS="$2"; shift 2;;
    --mixtures) MIX="$2"; shift 2;;
    --embed-cmd) EMBED_CMD="$2"; shift 2;;
    --embed-out) EMBED_OUT="$2"; shift 2;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -n "$TAXA$IDS" || ${#RANGE[@]} -gt 0 ]] || { echo "need --taxa-file, --ids-file or --id-range" >&2; exit 2; }
[[ -z "$EMBED_CMD" || -n "$EMBED_OUT" ]] || { echo "--embed-cmd needs --embed-out" >&2; exit 2; }
mkdir -p "$WORK"; cd "$WORK"
GTDB_DIR="${GTDB_DIR:-$(pwd)/db/gtdb_r226}"
GTDB_URL=https://data.gtdb.ecogenomic.org/releases/release226/226.0
GTDB_FILES=("$GTDB_DIR/bac120_metadata_r226.tsv.gz" "$GTDB_DIR/ar53_metadata_r226.tsv.gz")
S="$DIR/submit.sh"
SRC=("${RANGE[@]}"); [[ -n "$TAXA" ]] && SRC+=(--taxa-file "$TAXA"); [[ -n "$IDS" ]] && SRC+=(--ids-file "$IDS")
j1=$($S --name mss_bacdive --cpus 1 --mem 16G --time 2-00:00:00 -- bacdive "${SRC[@]}" "${EXTRA_BD[@]}" --out data/bacdive_reference.tsv)
# GTDB metadata download (internet) runs in parallel with the BacDive scan.
jg=$($S --name mss_gtdb --cpus 1 --mem 2G --time 02:00:00 --cmd "mkdir -p $GTDB_DIR && cd $GTDB_DIR && for f in bac120_metadata_r226.tsv.gz ar53_metadata_r226.tsv.gz; do [ -s \$f ] || { curl -fsSL -o \$f.part $GTDB_URL/\$f && mv \$f.part \$f; }; done && ls -la")
j2=$($S --name mss_manifest --cpus 1 --mem 16G --time 01:00:00 --after "$j1:$jg" -- genome-manifest --reference data/bacdive_reference.tsv --out data/genome_manifest.tsv "${LABELED[@]}" --gtdb-metadata "${GTDB_FILES[@]}" "${REPS[@]}")
j3=$($S --name mss_download --cpus 8 --mem 16G --time 2-00:00:00 --after "$j2" -- download-genomes --manifest data/genome_manifest.tsv --outdir genomes --out data/labeled_genomes.tsv --batch-size 25 --workers 16)
j4=$($S --name mss_features --cpus "$THREADS" --mem 32G --time 2-00:00:00 --after "$j3" -- features --manifest data/labeled_genomes.tsv --out features/kmer5.tsv.gz --threads "$THREADS")
FEAT=(); TRAIN_AFTER="$j4"
if [[ -n "$EMBED_CMD" ]]; then
  j4b=$($S --gpu --name mss_embed --cpus 8 --mem 64G --time 24:00:00 --after "$j3" --cmd "$EMBED_CMD")
  FEAT=(--embeddings "$EMBED_OUT"); TRAIN_AFTER="$j4b"
  echo "embedding (GPU) job $j4b"
fi
j5=$($S --name mss_train --cpus "$THREADS" --mem 32G --time 12:00:00 --after "$TRAIN_AFTER" -- train --manifest data/labeled_genomes.tsv --out models "${FEAT[@]}" --threads "$THREADS" --cv-group gtdb_genus --report-by gtdb_phylum)
j5b=$($S --name mss_train_clear --cpus "$THREADS" --mem 32G --time 12:00:00 --after "$TRAIN_AFTER" -- train --manifest data/labeled_genomes.tsv --out models_clear "${FEAT[@]}" --threads "$THREADS" --cv-group gtdb_genus --report-by gtdb_phylum --tasks oxygen --oxygen-classes anaerobe aerobe)
j6=$($S --name mss_predict --cpus "$THREADS" --mem 16G --time 04:00:00 --after "$j5" -- predict-genomes --manifest data/labeled_genomes.tsv --model models/phenotype_model.joblib --out results/genome_phenotype_predictions.tsv "${FEAT[@]}" --threads "$THREADS")
j7=$($S --name mss_train_raw --cpus "$THREADS" --mem 32G --time 12:00:00 --after "$j4" -- train-raw --manifest data/labeled_genomes.tsv --out models --mixtures "$MIX" --threads "$THREADS")
echo "workdir $(pwd)"; echo "jobs: bacdive=$j1 gtdb=$jg manifest=$j2 train_clear=$j5b download=$j3 features=$j4 train=$j5 predict=$j6 train_raw=$j7"
