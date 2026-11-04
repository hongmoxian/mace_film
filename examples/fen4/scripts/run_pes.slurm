#!/bin/bash
#SBATCH --job-name=fen4_film_pes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=fen4_film_pes-%j.out

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MACE_SOURCE="${MACE_SOURCE:-}"
PYTHON="${PYTHON:-python}"
OUT="${ROOT}/outputs/fen4/pes"
export PYTHONPATH="${MACE_SOURCE}:${ROOT}/mace_film:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
mkdir -p "${OUT}"

"${PYTHON}" "${ROOT}/mace_film/scan_fe_o2_adsorption_mace_film.py" \
  --checkpoint "${ROOT}/examples/fen4/models/pes.pt" \
  --atomic-energy-linref "${ROOT}/examples/fen4/inputs/reference_energies_pes.json" \
  --model-spec "AFM|${ROOT}/examples/fen4/inputs/afm_template.extxyz|0|0" \
  --model-spec "FM|${ROOT}/examples/fen4/inputs/fm_template.extxyz|0|1" \
  --fe-index 54 \
  --o-indices 59 60 \
  --scan-distance-range 1.8:4.0:0.1 \
  --init-magmom-mode template \
  --restraint-k 50 \
  --opt-fmax 0.05 \
  --opt-steps 200 \
  --device cuda \
  --output-csv "${OUT}/pes_scan.csv" \
  --output-plot "${OUT}/pes_scan.png" \
  --output-structures-dir "${OUT}/structures"
