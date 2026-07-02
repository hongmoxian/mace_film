#!/bin/bash
# Example FeN4-O2 MACE-FiLM training job.
# Edit ROOT, PYTHON, and LINREF for your cluster before submission.

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J fen4_mace_film_example
#SBATCH -o slurm-train-fen4-example-%j.out

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PYTHON=${PYTHON:-python}
LINREF=${LINREF:-${ROOT}/examples/fen4/data/atomic_energy_linref.json}
DATA=${DATA:-${ROOT}/examples/fen4/data/fen4_aimd10_opt_pes_previous_sample20.extxyz}
SAVE=${SAVE:-${ROOT}/outputs/fen4_example_checkpoint}

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p "${SAVE}"

"${PYTHON}" "${ROOT}/mace_film/train_fen4_mace_magmom_film.py" \
  --extxyz "${DATA}" \
  --save-dir "${SAVE}" \
  --batch-size 2 \
  --num-workers 0 \
  --max-epochs 2 \
  --lr 1e-3 \
  --cutoff 5.0 \
  --train-fraction 0.8 \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --energy-w 1 \
  --force-w 100 \
  --magmom-w 10 \
  --total-magmom-w 1 \
  --loss-type l1 \
  --hidden-irreps '128x0e + 128x1o' \
  --num-interactions 2 \
  --max-ell 3 \
  --correlation 3 \
  --film-scale 0.1 \
  --film-use-tanh \
  --magmom-input-clip 2.0 \
  --film-element-embedding-dim 8 \
  --device cuda \
  --use-zbl

