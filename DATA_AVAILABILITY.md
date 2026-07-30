# Data availability

This repository is a compact release of the MACE-FiLM code and the data used for the paper figures.

Included:

- compressed system datasets in `examples/*/data/`;
- final checkpoints and compact training/test metadata in `examples/*/models/`;
- final PNG/PDF figures and the CSV data used to plot them;
- the retained MACE-FiLM training, PES-scan, and ASE ML-MD source code.

The FeN4 release contains separate PES and ML-MD model families. The Mn-Au-O2 release contains the common model family used for both PES and ML-MD. The CeO2 release contains the mixed 2O/3O/4O training dataset and its final checkpoint.

Raw VASP files, raw AIMD/ML-MD trajectory directories, Slurm logs, resume checkpoints, and discarded exploratory outputs are intentionally omitted from this GitHub package. They should be deposited in a persistent research-data archive if complete raw-trajectory reproducibility is required.

SHA256 checksums for the released files are recorded in `MANIFEST.csv`.
