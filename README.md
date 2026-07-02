# MACE-FiLM magnetic-moment potentials

This directory is a compact release package for the MACE-FiLM part of the paper.  It contains the custom magnetic-moment FiLM model, training code, MD rollout code, PES scan code, and one small FeN4-O2 example.

The example data are intentionally small so that the repository can be hosted on GitHub.  The full training datasets and raw ML-MD trajectories should be archived separately, for example with Zenodo or an institutional data repository.

File sizes and SHA256 checksums are listed in `MANIFEST.csv`.  A short note on omitted large files is provided in `DATA_AVAILABILITY.md`.

## Contents

```text
mace_film/
  mace_magmom_film.py                  # MACE model with initial-magnetic-moment FiLM conditioning
  train_fen4_mace_magmom_film.py       # energy/force/magnetic-moment training entry point
  run_md_mace_film.py                  # ML-MD rollout with magnetic-moment propagation
  scan_fe_o2_adsorption_mace_film.py   # constrained Fe-O2 PES scan
  build_mace_extxyz_from_pt_cache.py   # internal helper used to build extxyz datasets
  subset_extxyz_by_info.py             # small extxyz filtering utility
  torch_ema.py                         # EMA helper

examples/fen4/
  data/
    fen4_aimd10_opt_pes_previous_sample20.extxyz
    fen4_afm_aimd_secondframe.extxyz
    fen4_fm_aimd_secondframe.extxyz
    fen4_fm_aimd_frame5401.extxyz
    atomic_energy_linref.json
  checkpoints/
    best.pt
    metrics.csv
  pes_outputs/
    fe_o2_scan_mace_aimd10_opt_pes_zbl_bounded_elem8.csv
    fe_o2_scan_mace_aimd10_opt_pes_zbl_bounded_elem8.png
  figure_source_data/
    dft_pes_reference.csv
    fen4_bounded_film_dft_ml_pes_magmom_overlay_data.csv
    fen4_afm_fm_previous_rule_selected_normalized_wasserstein.csv
    figure2_previous_S0_merged.csv
    figure2_previous_S2_merged.csv
  figures/
    final FeN4 PES/magnetic-moment and ML-MD distribution figures
  slurm/
    original Slurm scripts plus portable example scripts

scripts/
  plotting scripts used for the FeN4 figures
```

## Environment

The code was developed with Python, PyTorch, ASE, and MACE.  A minimal environment is:

```bash
pip install -r requirements.txt
```

For GPU training and MD, install a PyTorch build matching the CUDA version on your cluster.  The custom scripts expect the upstream `mace` Python package to be importable.

## Quick Checks

From this directory:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python mace_film/train_fen4_mace_magmom_film.py --help
python mace_film/run_md_mace_film.py --help
python mace_film/scan_fe_o2_adsorption_mace_film.py --help
```

## FeN4 Example

The included `sample20` extxyz is only a lightweight example, not the full paper training set.

Train a short smoke-test model:

```bash
bash examples/fen4/slurm/train_fen4_example.sh
```

Run a PES scan using the bundled checkpoint:

```bash
bash examples/fen4/slurm/scan_pes_fen4_example.sh
```

Run an ML-MD rollout using the bundled checkpoint:

```bash
bash examples/fen4/slurm/md_fen4_example.sh
```

The original cluster submission scripts are also retained in `examples/fen4/slurm/`.  They use absolute paths from the original working directory and are kept for provenance.

## Paper FeN4 Figures

Two FeN4 paper figures are bundled:

1. `examples/fen4/figures/fen4_bounded_film_dft_ml_pes_magmom_overlay.*`
   - DFT input: `examples/fen4/figure_source_data/dft_pes_reference.csv`
   - ML input: `examples/fen4/figure_source_data/fe_o2_scan_mace_aimd10_opt_pes_zbl_bounded_elem8.csv`
   - merged plot data: `examples/fen4/figure_source_data/fen4_bounded_film_dft_ml_pes_magmom_overlay_data.csv`
   - model: `examples/fen4/checkpoints/best.pt`
   - scan script: `mace_film/scan_fe_o2_adsorption_mace_film.py`
   - original submission script: `examples/fen4/slurm/sub.scan_fen4_mace_aimd10_opt_pes_zbl_bounded_elem8`

2. `examples/fen4/figures/fen4_afm_fm_previous_rule_selected_statistics.*`
   - compact plotted error data: `examples/fen4/figure_source_data/fen4_afm_fm_previous_rule_selected_normalized_wasserstein.csv`
   - reference AIMD summaries: `figure2_previous_S0_merged.csv`, `figure2_previous_S2_merged.csv`
   - plotting script: `scripts/plot_fen4_afm_fm_previous_best3_rule_worst3_compact.py`
   - raw ML-MD trajectories are not bundled by default because they are larger and are better suited for data-archive deposition.

## What Is Not Included

The following files are intentionally excluded from the GitHub-sized package:

- full FeN4 training extxyz files
- full raw ML-MD trajectories
- all `last.pt` resume checkpoints
- Slurm stdout logs
- repeated exploratory PES/MD outputs
- VASP OUTCAR files

For strict full reproduction of every density panel from raw trajectories, archive the full `md_outputs` directories and VASP OUTCAR/AIMD trajectory files separately.
