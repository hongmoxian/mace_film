# Mn-Au-O2 example

The supplied model.pt supports MD and constrained PES demonstrations.

~~~bash
sbatch scripts/run_md.slurm
sbatch scripts/run_pes.slurm
~~~

The inputs folder contains AFM/FM starting structures, reference
energies, and a small labeled sample. The complete training data
and publication results are maintained separately.
