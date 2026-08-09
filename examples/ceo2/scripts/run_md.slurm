#!/bin/bash
#SBATCH --job-name=ceo2_film_md
#SBATCH --partition=8v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=ceo2_film_md-%j.out

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MACE_SOURCE="${MACE_SOURCE:-/home/wuzhihong/software/mace}"
PYTHON="${PYTHON:-/home/wuzhihong/app/miniconda/envs/mace/bin/python}"
export PYTHONPATH="${MACE_SOURCE}:${ROOT}/mace_film/systems/ceo2:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"

"${PYTHON}" "${ROOT}/mace_film/systems/ceo2/run_md_mace_film.py" \
  --structure "${ROOT}/examples/ceo2/inputs/2O_start.extxyz" \
  --structure-index 0 \
  --checkpoint "${ROOT}/examples/ceo2/models/mixed_2O3O4O_softmax_best.pt" \
  --atomic-energy-linref "${ROOT}/examples/ceo2/inputs/atomic_energy_linref.json" \
  --device cuda \
  --temperature-k 800 \
  --timestep-fs 1.0 \
  --thermostat nose-hoover \
  --steps 5000 \
  --velocity-seed 123 \
  --rollout-magmom \
  --constrain-active-ce \
  --fixed-active-count 2 \
  --output-dir "${ROOT}/outputs/ceo2/md_2O"
