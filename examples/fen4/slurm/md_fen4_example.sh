#!/bin/bash
# Example FeN4-O2 ML-MD rollout with magnetic-moment propagation.

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J fen4_md_example
#SBATCH -o slurm-md-fen4-example-%j.out

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PYTHON=${PYTHON:-python}
CKPT=${CKPT:-${ROOT}/examples/fen4/checkpoints/best.pt}
STRUCT=${STRUCT:-${ROOT}/examples/fen4/data/fen4_afm_aimd_secondframe.extxyz}
OUTDIR=${OUTDIR:-${ROOT}/outputs/fen4_md_example}

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTDIR}"

"${PYTHON}" "${ROOT}/mace_film/run_md_mace_film.py" \
  --structure "${STRUCT}" \
  --structure-index 0 \
  --checkpoint "${CKPT}" \
  --device cuda \
  --atomic-numbers 6 7 8 26 \
  --temperature-k 50 \
  --timestep-fs 1.0 \
  --thermostat langevin \
  --friction 0.02 \
  --steps 10000 \
  --loginterval 20 \
  --velocity-seed 101 \
  --rollout-magmom \
  --init-magmom-mode trajectory \
  --output-dir "${OUTDIR}"

