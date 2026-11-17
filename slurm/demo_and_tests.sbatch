#!/bin/bash
# Run the synthetic offline demo and the pytest suite on a compute node.
#   MSS_SLURM_DIR=$PWD/slurm sbatch slurm/demo_and_tests.sbatch   (from the repo root)
#SBATCH --job-name=mss_demo
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source "${MSS_SLURM_DIR:-$PWD/slurm}/_common.sh"; mss_activate
python scripts/make_demo.py
mss taxa --profile demo_generated/profile.tsv --reference demo_generated/phenotypes.tsv --out results/demo_taxa.tsv
mss train --manifest demo_generated/genomes.tsv --out models/demo
mss predict-genomes --manifest demo_generated/genomes.tsv --model models/demo/phenotype_model.joblib --out results/demo_genome_predictions.tsv
mss taxa --profile demo_generated/profile.tsv --reference demo_generated/phenotypes.tsv --predictions results/demo_genome_predictions.tsv --out results/demo_taxa_with_predictions.tsv
mss train-raw --manifest demo_generated/genomes.tsv --out models/demo --mixtures 250
mss raw --fastq demo_generated/reads.fastq --model models/demo/raw_mixture.joblib --out results/demo_raw.tsv
python -m pytest -q
echo "DEMO AND TESTS OK"
