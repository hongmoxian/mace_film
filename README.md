# MACE-FiLM

MACE-FiLM conditions a MACE potential on atom-resolved initial magnetic
moments and predicts energies, forces, and local magnetic moments. This
repository is a compact, code-oriented release: it contains the model and
training implementations, portable ASE MD/PES examples, the checkpoints
needed by those examples, and the workflow figure. Details about included
example data and reproducibility are in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

![MagPES-MACE workflow including optional global magnetic-reversal symmetry](figures/workflow.png)

The workflow has an editable vector file at figures/workflow.svg and a
reproducible rendering script at tools/build_workflow.py.

## Repository layout

```text
mace_film/                    model, training, ASE MD, and PES source
examples/fen4/                FeN4-O2 MD and PES examples
examples/mn_au_o2/            Mn-Au-O2 MD and PES examples
examples/ceo2/                CeO2 magnetic-localization MD example
examples/even_pair/           optional single-forward FeN4 symmetry example
figures/workflow.png          updated framework and MHC feedback diagram
figures/workflow.svg          editable vector version
docs/MAGNETIC_SYMMETRY.md      mathematics, checkpoint and MD caveats
tests/                        compatibility and spin-symmetry tests
```

Each example contains only:

- checkpoint files needed for the demonstrated task;
- small structural inputs and atomic-energy references;
- a small `.extxyz.gz` sample documenting the training-data schema;
- Slurm scripts for the retained MD/PES workflows;
- a short system-specific README.

Optional trained symmetry example: examples/even_pair/.

The samples are not substitutes for the complete production datasets and
must not be used to reproduce paper statistics.

Decompress a schema sample before passing it to the training entry point:

```bash
gzip -dk examples/fen4/inputs/train_sample.extxyz.gz
```

## Installation

Install an upstream MACE version compatible with the source in this release,
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="/path/to/upstream/mace:$PWD:$PYTHONPATH"
```

The examples default to `/home/wuzhihong/software/mace` for the upstream MACE
source. Override this portable path at submission time when needed:

```bash
MACE_SOURCE=/path/to/mace sbatch examples/fen4/scripts/run_md.slurm
```

## Retained workflows

### FeN4-O2

- MD checkpoint: `examples/fen4/models/md_spp_previous_best.pt`
- PES checkpoint: `examples/fen4/models/pes_afm_energy10_best.pt`
- MD: `sbatch examples/fen4/scripts/run_md.slurm`
- PES: `sbatch examples/fen4/scripts/run_pes.slurm`

The PES checkpoint is the Old E10 model selected for the manuscript. The
same checkpoint is used for both AFM and FM PES branches; no branch-specific
FM model is used. The MD and PES checkpoints differ because those workflows
emphasize different regions of configuration space.

### Mn-Au-O2

- common MD/PES checkpoint:
  `examples/mn_au_o2/models/pes_md_spp_previous_best.pt`
- MD: `sbatch examples/mn_au_o2/scripts/run_md.slurm`
- PES: `sbatch examples/mn_au_o2/scripts/run_pes.slurm`

The same previous-conditioned checkpoint is retained for the two example
workflows.

### CeO2

- MD checkpoint: `examples/ceo2/models/mixed_2O3O4O_softmax_best.pt`
- MD: `sbatch examples/ceo2/scripts/run_md.slurm`

The checkpoint was trained on a mixed 2O/3O/4O dataset. The paper uses this
system for magnetic-site localization and ML-MD, not for a static constrained
PES scan; consequently no artificial CeO2 PES example is included.

## Source entry points

```text
mace_film/mace_magmom_film.py
mace_film/train_fen4_mace_magmom_film.py
mace_film/run_md_mace_film.py
mace_film/scan_fe_o2_adsorption_mace_film.py

mace_film/systems/mn_au_o2/mace_magmom_film.py
mace_film/systems/mn_au_o2/train_mn_au_mace_magmom_film.py
mace_film/systems/mn_au_o2/run_md_mace_film.py
mace_film/systems/mn_au_o2/scan_mn_au_o2_adsorption_mace_film.py

mace_film/systems/ceo2/mace_magmom_film.py
mace_film/systems/ceo2/train_ceo2_mace_magmom_film.py
mace_film/systems/ceo2/run_md_mace_film.py
```

All inference entry points accept `--atomic-energy-linref`. This overrides
the historical absolute path stored inside a checkpoint and makes the
released examples portable.

## Magnetic conditioning

The default model conditions the initial scalar node embedding with an
atom-wise FiLM transformation derived from normalized initial magnetic
moments. MD can optionally feed predicted moments into the next step with
`--rollout-magmom`; PES examples keep the supplied branch template fixed
unless explicitly configured otherwise.

### Optional single-forward magnetic-reversal symmetry

The main production models use the original signed magnetic FiLM input
(spin_feature_mode=legacy, the default). A separate and optional
single-forward mode, spin_feature_mode=even_pair, builds even descriptors:

    q_i = (s_i**2, s_i * sbar_i),

where s_i is the normalized atomic moment and sbar_i is a
distance-weighted local magnetic neighbor average.

For global magnetic inversion:

    E(R,m) = E(R,-m)
    F(R,m) = F(R,-m)
    m_pred(R,-m) = -m_pred(R,m).

The relative-alignment term s_i*sbar_i preserves sensitivity to
FM/AFM branches. This mode is not the older two-initialization
feature-averaging technique; --enforce-spin-inversion is a
separate historical option and must not be used together with even_pair.

The optional symmetry code is *integrated* into the public FeN4
model and trainer (not a separate module copy). It is disabled by default
for all released manuscript PES and MD examples. Legacy checkpoints
cannot be used in even_pair mode without retraining.

See docs/MAGNETIC_SYMMETRY.md and examples/even_pair/README.md.
The Mn-Au-O2 and CeO2 system-specific model copies still implement
their original legacy workflows.

## Training quickstart (small schema sample)

The GitHub repository includes three labeled but tiny extxyz samples:
12 FeN4/O2, 12 Mn-Au-O2, and 9 CeO2 frames. The data carries REF_energy,
REF_forces, initial_magmom, target_magmom and state_id fields. These are
for checking the input interface, not for reproducing scientific MAEs.

From the repository root with a compatible upstream MACE on PYTHONPATH:

~~~bash
gzip -dk examples/fen4/inputs/train_sample.extxyz.gz
python mace_film/train_fen4_mace_magmom_film.py \
  --extxyz examples/fen4/inputs/train_sample.extxyz \
  --save-dir outputs/fen4/quickstart \
  --atomic-energy-linref examples/fen4/inputs/atomic_energy_linref_md.json \
  --max-epochs 1 --batch-size 2 --num-workers 0 \
  --device cpu --conditioning-mode film
~~~

To train an even_pair model on a properly prepared, sufficiently
large labeled dataset, add the following parameters:

~~~text
--spin-feature-mode even_pair
--magmom-condition-elements Fe O
--magmom-supervise-elements Fe O
~~~

To load the ready-to-test trained symmetry demonstration instead,
use examples/even_pair/models/best.pt and its provided reference
energies. Run the validation test:

~~~bash
PYTHONPATH="/path/to/upstream/mace:$PWD/mace_film:$PYTHONPATH" \
  python tests/validate_trained_symmetry.py
~~~

## Tests and version compatibility

Validated environment: Python 3.11, torch 2.3.1, mace-torch 0.3.14,
e3nn 0.4.4 and ASE 3.28; see docs/DEPENDENCIES.md for details.
This repository imports MACE 0.3.x internals and has NOT been
validated against the MACE 1.x API.

The legacy FeN4 MD and PES demo checkpoints were tested against
the updated default model: energy, force, and predicted moment
differences were exactly zero. The trained even_pair checkpoint
passed reversal and relative-magnetic-alignment checks.
These tests are documented in docs/INTEGRATION_NOTES_20261030.md.

The MHC rollout updates magnetic inputs externally at each accepted
MD step. Therefore, total-energy conservation cannot be presumed
for an MHC-NVE run. The updated FeN4 MD driver accounts for
magnetic-state update work and prevents stale ASE energy/force caches.

Do not load untrusted PyTorch checkpoints. Preserve the same DFT
local-moment projection convention at training and prediction time.

## Data availability

This repository provides compact training-schema samples and selected example
checkpoints, rather than the full datasets used for the paper. See
[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for details and
[MANIFEST.csv](MANIFEST.csv) for the included files and SHA256 checksums.

<!-- Maintainer comment: release curation (not shown in the rendered README).
Removed from the GitHub software release:
- paper PES/MD/statistical figures other than Figure 1;
- figure-data CSV files;
- full production training datasets;
- raw VASP outputs and full AIMD/ML-MD trajectories;
- exploratory checkpoints, training histories, and duplicated metadata.

The optional even_pair pretrained example is retained for testing the symmetry
extension; tiny schema samples are not substitutes for scientific training sets.
Complete paper datasets should be archived separately in a persistent research
data repository. The retained contents and checksums are documented in
DATA_AVAILABILITY.md and MANIFEST.csv.
-->

## License and attribution

MACE-FiLM is released under the **MIT License**. See
[LICENSE](LICENSE) for the permissions, conditions, and disclaimer.
Copyright (c) 2026 hongmoxian and contributors.

This project builds on the [MACE](https://github.com/ACEsuit/mace)
framework, which is distributed under the MIT License. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
Third-party dependencies retain their original licenses. The repository
contains small example datasets and pretrained checkpoints; their
provenance and scope are described in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).
