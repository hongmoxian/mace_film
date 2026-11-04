#!/bin/bash
#SBATCH --job-name=ceo2_film_md
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=ceo2_film_md-%j.out

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MACE_SOURCE="${MACE_SOURCE:-}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${MACE_SOURCE}:${ROOT}/mace_film/systems/ceo2:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"

"${PYTHON}" "${ROOT}/mace_film/systems/ceo2/run_md_mace_film.py" \
  --structure "${ROOT}/examples/ceo2/inputs/structure.extxyz" \
  --structure-index 0 \
  --checkpoint "${ROOT}/examples/ceo2/models/model.pt" \
  --atomic-energy-linref "${ROOT}/examples/ceo2/inputs/reference_energies.json" \
  --device cuda \
  --temperature-k 800 \
  --timestep-fs 1.0 \
  --thermostat nose-hoover \
  --steps 5000 \
  --velocity-seed 123 \
  --rollout-magmom \
  --constrain-active-ce \
  --fixed-active-count 2 \
  --output-dir "${ROOT}/outputs/ceo2/md"
