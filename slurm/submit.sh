#!/bin/bash
# Submit one MSS step to SLURM. Prints the job ID.
#
#   slurm/submit.sh [options] -- <mss subcommand and args>
#   slurm/submit.sh [options] --cmd "<any shell command>"      (e.g. external embedding script)
#
# Options:
#   --gpu              request a GPU (partition $MSS_GPU_PARTITION, qos $MSS_GPU_QOS, gres $MSS_GPU_GRES)
#   --cpus N           CPUs per task (default 4)       --mem SIZE   (default 16G)
#   --time T           walltime (default 04:00:00)     --partition P
#   --after JOBID[:JOBID]   start after these jobs succeed (afterok)
#   --array SPEC       SLURM array spec, e.g. 0-49%20
#   --name NAME        job name
#   --dry-run          print the sbatch command only
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/_common.sh"
GPU=0; CPUS=4; MEM=16G; TIME=04:00:00; PART=""; AFTER=""; ARRAY=""; NAME=mss; DRY=0; CMD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU=1; shift;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --partition) PART="$2"; shift 2;;
    --after) AFTER="$2"; shift 2;;
    --array) ARRAY="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --cmd) CMD=(--cmd "$2"); shift 2;;
    --) shift; CMD=("$@"); break;;
    -h|--help) sed -n '2,17p' "$0"; exit 0;;
    *) echo "unknown option $1 (put mss arguments after --)" >&2; exit 2;;
  esac
done
[[ ${#CMD[@]} -gt 0 ]] || { echo "nothing to run; see --help" >&2; exit 2; }
mkdir -p "$MSS_LOGDIR"
ARGS=(--parsable --job-name="$NAME" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
      --export=ALL,MSS_SLURM_DIR="$DIR")
if [[ -n "$ARRAY" ]]; then ARGS+=(--array="$ARRAY" --output="$MSS_LOGDIR/%x_%A_%a.out")
else ARGS+=(--output="$MSS_LOGDIR/%x_%j.out"); fi
if [[ $GPU == 1 ]]; then
  ARGS+=(--partition="${PART:-$MSS_GPU_PARTITION}" --qos="$MSS_GPU_QOS" --gres="$MSS_GPU_GRES")
else
  ARGS+=(--partition="${PART:-$MSS_CPU_PARTITION}")
fi
[[ -n "$AFTER" ]] && ARGS+=(--dependency="afterok:$AFTER" --kill-on-invalid-dep=yes)
if [[ $DRY == 1 ]]; then echo sbatch "${ARGS[@]}" "$DIR/mss_job.sbatch" "${CMD[@]}"; exit 0; fi
sbatch "${ARGS[@]}" "$DIR/mss_job.sbatch" "${CMD[@]}"
