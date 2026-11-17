#!/bin/bash
# Generic MSS job: runs `mss <args>` (or any command after `--cmd`).
# Normally submitted through slurm/submit.sh, which sets resources.
#SBATCH --job-name=mss
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
source "${MSS_SLURM_DIR:?MSS_SLURM_DIR unset; submit via slurm/submit.sh}/_common.sh"
mss_activate
echo "[mss] host=$(hostname) job=${SLURM_JOB_ID:-} cpus=${SLURM_CPUS_PER_TASK:-1} gpus=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
if [[ "${1:-}" == "--cmd" ]]; then shift; echo "[mss] cmd: $*"; bash -c "$*"
else
  echo "[mss] mss $*"
  # /usr/bin/time -v records elapsed time and peak memory (RSS) in the log
  if [[ -x /usr/bin/time ]]; then /usr/bin/time -v mss "$@"; else mss "$@"; fi
fi
echo "[mss] done $(date -Is)"
