# Data availability note

This GitHub package is intentionally compact. It contains the code, a small FeN4-O2 example dataset, the final bounded MACE-FiLM checkpoint used by the included PES example, and the source CSV files needed for the bundled FeN4 paper figures.

Large files that should be deposited separately for full reproducibility are:

- full FeN4 training extxyz datasets
- full FeN4 raw ML-MD trajectory directories
- VASP OUTCAR and raw AIMD trajectories
- exploratory checkpoints such as `last.pt`
- repeated Slurm stdout/stderr logs

The bundled `examples/fen4/data/fen4_aimd10_opt_pes_previous_sample20.extxyz` file is a 20-frame smoke-test subset, not the full training set used in the manuscript.
