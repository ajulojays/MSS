# MSS SLURM configuration. Copy to slurm/config.sh (git-ignored) and edit.
# Conda env (or venv) prefix that contains the `mss` command:
export MSS_ENV="${MSS_ENV:-$HOME/envs/mss}"
# Where logs go (relative to the directory you submit from):
export MSS_LOGDIR="${MSS_LOGDIR:-logs/slurm}"
# CPU defaults
export MSS_CPU_PARTITION="${MSS_CPU_PARTITION:-general}"
# GPU defaults (UNC Longleaf: GPU partitions need --qos=gpu_access)
export MSS_GPU_PARTITION="${MSS_GPU_PARTITION:-l40-gpu}"
export MSS_GPU_QOS="${MSS_GPU_QOS:-gpu_access}"
export MSS_GPU_GRES="${MSS_GPU_GRES:-gpu:1}"
