# Optional even_pair magnetic-reversal symmetry: FeN4/O2

This example demonstrates the single-forward, global magnetic-reversal
symmetry option. It is an **optional extension**. The main-manuscript
FeN4/O2, Mn–Au–O2, and CeO2 production calculations used the original
signed magnetic FiLM conditioning (spin_feature_mode=legacy).

## Contents

- configs/config_evenpair.json: model and training configuration used
  for the symmetry study. Data paths should be adjusted for your system.
- inputs/first_real_frame.extxyz: one FeN4/O2 AIMD example geometry
  (including initial and target moment arrays).
- inputs/atomic_energy_linref.json: per-element reference energies.
- models/best.pt: trained symmetry-mode example checkpoint.

The example frame and trained checkpoint are **not** a complete training
dataset, and are not sufficient to reproduce reported paper performance.

## Mathematical properties

For normalized magnetic input s_i and distance-weighted neighbor
mean sbar_i, the optional FiLM descriptor is

    q_i = (s_i**2, s_i * sbar_i).

For global magnetic reversal m -> -m:

    E(R,m) = E(R,-m)
    F(R,m) = F(R,-m)
    m_pred(R,-m) = -m_pred(R,m).

The product s_i*sbar_i responds to local relative alignment changes,
allowing different AFM/FM branch predictions.

Use --spin-feature-mode even_pair for NEW training. The original
legacy checkpoint cannot be converted to this representation simply
by changing a CLI flag.

## Test this pretrained example

From the GitHub repository root:

    PYTHONPATH="/path/to/upstream/mace:$PWD/mace_film:$PYTHONPATH" python tests/validate_trained_symmetry.py

The script evaluates the original and globally reversed magnetic
inputs at identical positions, and one local spin reversal to verify
that relative magnetic alignment remains distinguishable.

## Training with your own data

Unpack a labeled extxyz dataset with REF_energy, REF_forces,
initial_magmom, target_magmom and state_id fields. Run the FeN4
training entry point and add these options:

    --spin-feature-mode even_pair
    --magmom-condition-elements Fe O
    --magmom-supervise-elements Fe O

Do not combine with the historical --enforce-spin-inversion switch.
MD/PES results for this model require a separately validated workflow.
MD magnetic feedback is not an NVE-conservative force field when magnetic
inputs are externally updated every integration step.
