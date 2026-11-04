# MACE-FiLM

Magnetic-moment-conditioned machine-learning interatomic potentials based on MACE.
MACE-FiLM predicts potential energy, atomic forces, and local magnetic moments
while allowing magnetic configurations to influence the energy landscape.

![MACE-FiLM workflow](figures/workflow.png)

## Highlights

- Equivariant interatomic potential with magnetic-moment conditioning.
- Spin-dependent potential-energy scans and molecular dynamics.
- Magnetic-state feedback during MD.
- Optional global magnetic-reversal symmetry.

## Installation

Use Python 3.11 with a compatible MACE 0.3.x installation.

~~~bash
python -m pip install -r requirements.txt
export MACE_SOURCE=/path/to/mace
export PYTHONPATH=$MACE_SOURCE:$PWD:$PYTHONPATH
~~~

See [Environment](docs/DEPENDENCIES.md) for validated dependency versions.

## Repository structure

| Location | Contents |
| --- | --- |
| mace_film/ | Model, training, MD, and PES source code |
| examples/fen4/ | FeN4/O2 MD and PES examples |
| examples/mn_au_o2/ | Mn-Au-O2 MD and PES examples |
| examples/ceo2/ | CeO2 MD example |
| examples/even_pair/ | Optional magnetic-reversal symmetry example |
| tests/ | Validation scripts |
| figures/ | Architecture illustration |

## Examples

Each example provides a pretrained model, minimal input structures,
and runnable Slurm scripts. Set MACE_SOURCE to your MACE installation
and activate your Python environment before submitting a job.

| System | Model | Example |
| --- | --- | --- |
| FeN4/O2 | examples/fen4/models/md_mhc.pt | examples/fen4/scripts/run_md.slurm |
| FeN4/O2 | examples/fen4/models/pes.pt | examples/fen4/scripts/run_pes.slurm |
| Mn-Au-O2 | examples/mn_au_o2/models/model.pt | examples/mn_au_o2/scripts/run_md.slurm and run_pes.slurm |
| CeO2 | examples/ceo2/models/model.pt | examples/ceo2/scripts/run_md.slurm |

Example:

~~~bash
sbatch examples/fen4/scripts/run_md.slurm
sbatch examples/fen4/scripts/run_pes.slurm
~~~

These are demonstration workflows. See the system README files for inputs
and recommended usage.

## Training

The main training scripts are provided under mace_film/.
Small labeled samples in examples/*/inputs/sample.extxyz.gz illustrate the
expected input format. They are not the complete research datasets.

The optional symmetry model is supplied under examples/even_pair/.
See [Magnetic symmetry](docs/MAGNETIC_SYMMETRY.md) for the method and validation.

## Data and reproducibility

This repository contains source code, pretrained demonstration models,
and small examples. The research datasets and additional simulation outputs
are maintained separately. See [Data availability](DATA_AVAILABILITY.md)
and [Reproducibility](docs/REPRODUCIBILITY.md).

## License

Released under the [MIT License](LICENSE).
MACE-FiLM builds on [MACE](https://github.com/ACEsuit/mace).
See [Third-party notices](THIRD_PARTY_NOTICES.md).
