#!/bin/bash
# Pfam features for an existing MSS run (after download-genomes), then the same training as the k-mer run,
# so the two feature sets can be compared on identical genomes and held-out genera.
#
#   slurm/run_pfam.sh --workdir runs/bacdive_all_v1 --hmm /path/Pfam-A.hmm.gz
#                     [--tasks 100] [--concurrent 60] [--cpus 8] [--mem 16G] [--threads 16]
#
# Steps:  pfam-annotate (array; pyrodigal genes + pyhmmer hmmsearch, Pfam GA cutoffs, cached per genome)
#      -> pfam-table   (genome x Pfam presence/absence: features/pfam_pa.tsv.gz)
#      -> train        (models_pfam/, same heads and class filters as models_v2/)
#      -> train clear  (models_pfam_clear/, oxygen aerobe vs anaerobe)
#      -> hybrid       (models_hybrid/, models_hybrid_clear/: k-mer table + Pfam table joined; needs features/kmer5.tsv.gz)
# Resubmitting is safe: genomes with a cache file are skipped.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=""; HMM=""; NT=100; CONC=60; CPUS=8; MEM=16G; THREADS=16
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) WORK="$2"; shift 2;;
    --hmm) HMM="$(realpath "$2")"; shift 2;;
    --tasks) NT="$2"; shift 2;;
    --concurrent) CONC="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -n "$WORK" && -n "$HMM" ]] || { echo "need --workdir and --hmm" >&2; exit 2; }
[[ -s "$HMM" ]] || { echo "HMM file not found: $HMM" >&2; exit 2; }
cd "$WORK"
[[ -s data/labeled_genomes.tsv ]] || { echo "$WORK/data/labeled_genomes.tsv missing (run download-genomes first)" >&2; exit 2; }
S="$DIR/submit.sh"
CV=(--cv-group gtdb_genus --report-by gtdb_phylum)
FILTERS=(--class-filter gram_stain=negative,positive --class-filter cell_shape=rod-shaped,coccus-shaped)

ja=$($S --name mss_pfam --array "0-$((NT - 1))%$CONC" --cpus "$CPUS" --mem "$MEM" --time 1-00:00:00 -- \
     pfam-annotate --manifest data/labeled_genomes.tsv --hmm "$HMM" --cache-dir .mss_cache/pfam --n-tasks "$NT")
jt=$($S --name mss_pfam_table --cpus 2 --mem 32G --time 04:00:00 --after "$ja" -- \
     pfam-table --manifest data/labeled_genomes.tsv --cache-dir .mss_cache/pfam --out features/pfam_pa.tsv.gz)
j1=$($S --name mss_train_pfam --cpus "$THREADS" --mem 64G --time 12:00:00 --after "$jt" -- \
     train --manifest data/labeled_genomes.tsv --out models_pfam --embeddings features/pfam_pa.tsv.gz \
     --threads "$THREADS" "${CV[@]}" "${FILTERS[@]}")
j2=$($S --name mss_train_pfam_clear --cpus "$THREADS" --mem 64G --time 12:00:00 --after "$jt" -- \
     train --manifest data/labeled_genomes.tsv --out models_pfam_clear --embeddings features/pfam_pa.tsv.gz \
     --threads "$THREADS" "${CV[@]}" --tasks oxygen --oxygen-classes anaerobe aerobe)
HYB=(--embeddings features/kmer5.tsv.gz features/pfam_pa.tsv.gz)
j3=$($S --name mss_train_hybrid --cpus "$THREADS" --mem 64G --time 12:00:00 --after "$jt" -- \
     train --manifest data/labeled_genomes.tsv --out models_hybrid "${HYB[@]}" \
     --threads "$THREADS" "${CV[@]}" "${FILTERS[@]}")
j4=$($S --name mss_train_hybrid_clear --cpus "$THREADS" --mem 64G --time 12:00:00 --after "$jt" -- \
     train --manifest data/labeled_genomes.tsv --out models_hybrid_clear "${HYB[@]}" \
     --threads "$THREADS" "${CV[@]}" --tasks oxygen --oxygen-classes anaerobe aerobe)
echo "workdir $(pwd)"; echo "jobs: pfam_annotate=$ja (array 0-$((NT - 1))%$CONC) pfam_table=$jt train_pfam=$j1 train_pfam_clear=$j2 train_hybrid=$j3 train_hybrid_clear=$j4"
