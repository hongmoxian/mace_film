# CeO2 example

`models/mixed_2O3O4O_softmax_best.pt` is the retained mixed 2O/3O/4O
checkpoint. Run the 2O magnetic-localization MD example with:

```bash
sbatch scripts/run_md.slurm
```

The example enables magnetic-moment rollout and constrains the active Ce set
to two sites. The complete 2O/3O/4O production dataset is not included;
`inputs/train_sample.extxyz.gz` contains three frames from each family only
to document the input schema. No static CeO2 PES task was used in the paper,
so this compact release does not provide one.
