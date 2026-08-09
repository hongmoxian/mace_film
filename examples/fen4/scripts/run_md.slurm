#!/bin/bash
#SBATCH --job-name=fen4_film_md
#SBATCH --partition=8v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=fen4_film_md-%j.out

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MACE_SOURCE="${MACE_SOURCE:-/home/wuzhihong/software/mace}"
PYTHON="${PYTHON:-/home/wuzhihong/app/miniconda/envs/mace/bin/python}"
export PYTHONPATH="${MACE_SOURCE}:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"

"${PYTHON}" "${ROOT}/mace_film/run_md_mace_film.py" \
  --structure "${ROOT}/examples/fen4/inputs/afm_template.extxyz" \
  --structure-index 0 \
  --checkpoint "${ROOT}/examples/fen4/models/md_spp_previous_best.pt" \
  --atomic-energy-linref "${ROOT}/examples/fen4/inputs/atomic_energy_linref_md.json" \
  --device cuda \
  --temperature-k 50 \
  --timestep-fs 1.0 \
  --thermostat nose-hoover \
  --steps 5000 \
  --velocity-seed 123 \
  --rollout-magmom \
  --output-dir "${ROOT}/outputs/fen4/md"
