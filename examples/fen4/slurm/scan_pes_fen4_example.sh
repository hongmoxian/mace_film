#!/bin/bash
# Example FeN4-O2 PES scan with a trained MACE-FiLM checkpoint.

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J fen4_pes_example
#SBATCH -o slurm-scan-fen4-example-%j.out

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PYTHON=${PYTHON:-python}
CKPT=${CKPT:-${ROOT}/examples/fen4/checkpoints/best.pt}
OUTROOT=${OUTROOT:-${ROOT}/outputs/fen4_pes_example}

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTROOT}"

"${PYTHON}" "${ROOT}/mace_film/scan_fe_o2_adsorption_mace_film.py" \
  --checkpoint "${CKPT}" \
  --fe-index 54 \
  --o-indices 59 60 \
  --scan-distance-range 1.8:4.0:0.1 \
  --device cuda \
  --restraint-k 50 \
  --opt-fmax 0.05 \
  --opt-steps 200 \
  --progress-every 5 \
  --init-magmom-mode template \
  --model-spec "AFM|${ROOT}/examples/fen4/data/fen4_afm_aimd_secondframe.extxyz|0|0" \
  --model-spec "FM|${ROOT}/examples/fen4/data/fen4_fm_aimd_secondframe.extxyz|0|1" \
  --output-csv "${OUTROOT}/fe_o2_scan_mace_film.csv" \
  --output-plot "${OUTROOT}/fe_o2_scan_mace_film.png" \
  --output-structures-dir "${OUTROOT}/structures"

