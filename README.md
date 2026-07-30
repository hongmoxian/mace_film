# MACE-FiLM for spin-polarized surface dynamics

This repository is the compact code-and-data release accompanying the paper. It contains the magnetic-moment-conditioned MACE-FiLM implementation, training and ASE evaluation code, the final checkpoints used for the reported calculations, compressed training datasets, and the final paper figures with their plotted data.

The release is organized by chemical system:

```text
mace_film/                         # model, training, PES, and ASE-ML-MD source
examples/fen4/                     # FeN4-O2 data, models, figures, and figure data
examples/mn_au_o2/                 # Mn-Au-O2 data, models, figures, and figure data
examples/ceo2/                     # CeO2 data, model, figures, and figure data
figures/workflow.png               # overview figure
```

## Model and data provenance

The labels used below are descriptive names for the magnetic-conditioning protocols:

- **Spp (previous)** propagates the predicted magnetic moments from one MD step to the next.
- **Stp (rule)** resets a branch-consistent magnetic template at every step.

These labels describe the magnetic seed protocol, not a claim that one protocol is universally superior.

### FeN4-O2

The FeN4 PES and ML-MD figures were generated with different model families because the two calculations emphasize different regions of configuration space.

- **ML-MD:** `examples/fen4/models/md_spp_previous_best.pt` and `md_stp_rule_best.pt`, trained on the stride-4 AIMD datasets in `examples/fen4/data/md/`.
- **PES:** `pes_afm_energy_weighted_best.pt` and `pes_fm_opt_finetuned_best.pt`, trained/fine-tuned with the AIMD-plus-optimization data and the short-distance optimization data in `examples/fen4/data/pes/`. The final PES plot combines the AFM energy-weighted branch checkpoint and the FM optimization-finetuned branch checkpoint; it is therefore not represented by one universal FeN4 checkpoint.

### Mn-Au-O2

PES and ML-MD use the same model family. The release contains the previous/Spp and rule/Stp companion checkpoints:

- `examples/mn_au_o2/models/pes_md_spp_previous_best.pt`
- `examples/mn_au_o2/models/pes_md_stp_rule_best.pt`

Their associated AIMD, constrained-scan, and optimization data are in `examples/mn_au_o2/data/`. The PES scan uses the Mn-O distance restraint together with the angle restraint used for the reported calculation.

### CeO2

CeO2 uses one mixed training set and one final checkpoint:

- Model: `examples/ceo2/models/mixed_2O3O4O_softmax_best.pt`
- Dataset: `examples/ceo2/data/mixed_2O3O4O_softmax.extxyz.gz`

The mixed dataset contains 2O, 3O, and 4O configurations, including the cleaned 4O trajectory subset. The final main localization figure and the corresponding SI figure show the 2O system; this presentation choice does not mean that 3O and 4O were absent from training.

## Source code

The framework source is retained under `mace_film/`:

- `mace_magmom_film.py`: magnetic-moment FiLM MACE model;
- `train_fen4_mace_magmom_film.py`: FeN4 training entry point;
- `run_md_mace_film.py`: ASE ML-MD with optional magnetic-moment rollout;
- `scan_fe_o2_adsorption_mace_film.py`: constrained FeN4 PES scan;
- `systems/mn_au_o2/`: Mn-Au-O2 training, MD, and PES entry points;
- `systems/ceo2/`: CeO2 training and MD entry points.

The system-specific model files are kept with their corresponding entry points so that the released scripts preserve the implementations used for each system.

## Quick checks

Install the Python dependencies and make the package importable:

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Inspect the available command-line options:

```bash
python3 mace_film/train_fen4_mace_magmom_film.py --help
python3 mace_film/run_md_mace_film.py --help
python3 mace_film/scan_fe_o2_adsorption_mace_film.py --help
python3 mace_film/systems/mn_au_o2/train_mn_au_mace_magmom_film.py --help
python3 mace_film/systems/mn_au_o2/run_md_mace_film.py --help
python3 mace_film/systems/mn_au_o2/scan_mn_au_o2_adsorption_mace_film.py --help
python3 mace_film/systems/ceo2/train_ceo2_mace_magmom_film.py --help
python3 mace_film/systems/ceo2/run_md_mace_film.py --help
```

The `.extxyz.gz` files are compressed only for repository transport. Decompress the required file before passing it to ASE/MACE, for example:

```bash
gzip -dk examples/ceo2/data/mixed_2O3O4O_softmax.extxyz.gz
```

GPU training and production MD require a PyTorch installation compatible with the local CUDA driver and an importable upstream `mace` package. The supplied `.pt` files are the final checkpoints; the neighboring metrics and split-summary files record their evaluation metadata.

## Figures and plotted data

Each system directory contains:

- `figures/`: final main-text and SI figures in PNG and PDF formats;
- `figure_data/`: CSV files used to generate the figures;
- `models/`: checkpoints and compact training/test metadata;
- `data/`: the associated compressed datasets and dataset summaries.

The exact model/data mapping for the PES and ML-MD figures is documented above. `MANIFEST.csv` records the byte size and SHA256 hash of every other tracked release file.

## Deliberate exclusions

To keep this repository focused and practical to clone, it does not contain raw VASP `OUTCAR` files, raw AIMD trajectory directories, full ML-MD trajectory dumps, Slurm logs, resume checkpoints such as `last.pt`, or obsolete exploratory scan/plot outputs. These files are not required to inspect the final figures or use the released model/data package and should be archived separately when complete raw-data deposition is needed.

## Citation and license

Please cite the accompanying paper when using the code, checkpoints, datasets, or figures. Add the project license here before public release if a repository-level license is required.
