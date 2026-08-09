# Data availability

This GitHub repository is a compact software and model release. It includes:

- the MACE-FiLM model and training implementations;
- portable ASE MD and constrained-PES scripts;
- the checkpoints used by the retained examples;
- small structures, atomic-energy references, and training-schema samples;
- the paper workflow figure (Figure 1).

It intentionally excludes the full production datasets, raw VASP outputs,
complete AIMD/ML-MD trajectories, processed paper plotting tables, and all
paper figures other than the workflow. Those research artifacts should be
deposited in a persistent data repository rather than duplicated in the
software repository.

The `.extxyz.gz` files under `examples/*/inputs/` contain only a few frames
for inspecting the required arrays and metadata. They are not statistically
representative datasets and cannot reproduce reported training or paper
metrics.

`MANIFEST.csv` records the size and SHA256 checksum of every retained file.
