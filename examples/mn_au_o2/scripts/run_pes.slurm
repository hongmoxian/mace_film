#!/bin/bash
#SBATCH --job-name=mn_au_film_pes
#SBATCH --partition=8v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=mn_au_film_pes-%j.out

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MACE_SOURCE="${MACE_SOURCE:-/home/wuzhihong/software/mace}"
PYTHON="${PYTHON:-/home/wuzhihong/app/miniconda/envs/mace/bin/python}"
OUT="${ROOT}/outputs/mn_au_o2/pes"
export PYTHONPATH="${MACE_SOURCE}:${ROOT}/mace_film/systems/mn_au_o2:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
mkdir -p "${OUT}"

"${PYTHON}" "${ROOT}/mace_film/systems/mn_au_o2/scan_mn_au_o2_adsorption_mace_film.py" \
  --checkpoint "${ROOT}/examples/mn_au_o2/models/pes_md_spp_previous_best.pt" \
  --atomic-energy-linref "${ROOT}/examples/mn_au_o2/inputs/atomic_energy_linref.json" \
  --model-spec "AFM|${ROOT}/examples/mn_au_o2/inputs/afm_template.extxyz|0|0" \
  --model-spec "FM|${ROOT}/examples/mn_au_o2/inputs/fm_template.extxyz|0|1" \
  --mn-index 15 \
  --o-indices 16 17 \
  --scan-distance-range 1.8:4.0:0.1 \
  --init-magmom-mode template \
  --distance-k 50 \
  --use-angle-restraint \
  --angle-k 20 \
  --opt-fmax 0.05 \
  --opt-steps 200 \
  --device cuda \
  --output-csv "${OUT}/pes_scan.csv" \
  --output-plot "${OUT}/pes_scan.png" \
  --output-structures-dir "${OUT}/structures"
