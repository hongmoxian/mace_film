# FeN4-O2 example

This example retains two task-specific checkpoints:

- `models/md_spp_previous_best.pt`: previous-conditioned ML-MD;
- `models/pes_afm_energy10_best.pt`: Old E10 PES model, used for both AFM and FM.

Run on one GPU and three CPU cores:

```bash
sbatch scripts/run_md.slurm
sbatch scripts/run_pes.slurm
```

`inputs/afm_template.extxyz` and `inputs/fm_template.extxyz` are equilibrated
AIMD structures. The portable PES example uses their stored magnetic moments
at every scan distance. The manuscript production scan instead selected
branch- and distance-matched early DFT-relaxation moments; those full DFT
trajectories are not distributed in this compact software repository.

`inputs/train_sample.extxyz.gz` contains six AFM and six FM frames and only
documents the required `REF_energy`, `REF_forces`, `initial_magmom`,
`target_magmom`, and `state_id` fields. It is not the production training set.
