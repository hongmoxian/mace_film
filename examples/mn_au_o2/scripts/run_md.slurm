#!/bin/bash
#SBATCH --job-name=mn_au_film_md
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=mn_au_film_md-%j.out

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MACE_SOURCE="${MACE_SOURCE:-}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${MACE_SOURCE}:${ROOT}/mace_film/systems/mn_au_o2:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"

"${PYTHON}" "${ROOT}/mace_film/systems/mn_au_o2/run_md_mace_film.py" \
  --structure "${ROOT}/examples/mn_au_o2/inputs/afm_template.extxyz" \
  --structure-index 0 \
  --checkpoint "${ROOT}/examples/mn_au_o2/models/model.pt" \
  --atomic-energy-linref "${ROOT}/examples/mn_au_o2/inputs/reference_energies.json" \
  --device cuda \
  --temperature-k 50 \
  --timestep-fs 1.0 \
  --thermostat nose-hoover \
  --steps 5000 \
  --velocity-seed 123 \
  --rollout-magmom \
  --output-dir "${ROOT}/outputs/mn_au_o2/md"
