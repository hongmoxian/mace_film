# Mn-Au-O2 example

`models/pes_md_spp_previous_best.pt` is the retained previous-conditioned
checkpoint and is used by both example workflows:

```bash
sbatch scripts/run_md.slurm
sbatch scripts/run_pes.slurm
```

The PES example applies the Mn-O distance restraint and the O-Mn-O angular
restraint used by this workflow. For portability it initializes each branch
from the magnetic moments stored in its AIMD template. Full DFT scan
trajectories and paper plotting tables are not included.

`inputs/train_sample.extxyz.gz` contains six AFM and six FM frames and exists
only to document the training-data schema.
