#!/bin/bash
# EXPERIMENTAL MSS-Raw over many samples as a SLURM array, then merge.
#   slurm/raw_array.sh --sample-sheet samples.tsv --model models/raw_mixture.joblib \
#                      --outdir results/raw [--max-reads 5000000] [--concurrent 50]
# samples.tsv: columns sample_id, fastq  (fastq may list R1,R2 comma-separated)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHEET=""; MODEL=""; OUT=results/raw; MAXR=""; CONC=50
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample-sheet) SHEET="$(realpath "$2")"; shift 2;;
    --model) MODEL="$(realpath "$2")"; shift 2;;
    --outdir) OUT="$2"; shift 2;;
    --max-reads) MAXR="$2"; shift 2;;
    --concurrent) CONC="$2"; shift 2;;
    -h|--help) sed -n '2,6p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -f "$SHEET" && -f "$MODEL" ]] || { echo "need existing --sample-sheet and --model" >&2; exit 2; }
N=$(( $(grep -cv '^\s*$' "$SHEET") - 1 ))
(( N > 0 )) || { echo "sample sheet has no rows" >&2; exit 2; }
mkdir -p "$OUT"
MR=(); [[ -n "$MAXR" ]] && MR=(--max-reads "$MAXR")
ja=$("$DIR/submit.sh" --name mss_raw --cpus 1 --mem 4G --time 04:00:00 --array "0-$((N-1))%$CONC" \
     --cmd "mss raw --sample-sheet $SHEET --array-index env --model $MODEL ${MR[*]:-} --out $OUT/part_\${SLURM_ARRAY_TASK_ID}.tsv")
jm=$("$DIR/submit.sh" --name mss_raw_merge --cpus 1 --mem 2G --time 00:20:00 --after "$ja" \
     --cmd "awk 'FNR==1 && NR!=1 {next} {print}' \$(ls $OUT/part_*.tsv | sort -V) > $OUT/mss_raw.tsv")
echo "array job $ja ($N samples), merge job $jm -> $OUT/mss_raw.tsv"
