# FeN4/O2 example

Use md_mhc.pt for magnetic-history-conditioned MD and pes.pt
for a two-branch PES demonstration.

~~~bash
sbatch scripts/run_md.slurm
sbatch scripts/run_pes.slurm
~~~

The inputs folder contains AFM/FM starting structures,
reference energies, and a small labeled training sample.
The PES example evaluates both branches with the same checkpoint.
The full publication data are maintained separately.
