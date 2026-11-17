# Sourced by MSS SLURM scripts.
_MSS_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_MSS_SLURM_DIR/config.sh" ]]; then source "$_MSS_SLURM_DIR/config.sh"
else source "$_MSS_SLURM_DIR/config.example.sh"; fi
mss_activate() {
  if [[ -n "${MSS_ENV:-}" && -x "$MSS_ENV/bin/mss" ]]; then
    export PATH="$MSS_ENV/bin:$PATH"
  fi
  command -v mss >/dev/null || { echo "mss not found: set MSS_ENV in slurm/config.sh" >&2; exit 2; }
}
