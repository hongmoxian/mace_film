# Magnetic conditioning and optional global spin-reversal symmetry

The original MACE-FiLM conditions an atomwise MACE embedding on a
normalized signed magnetic moment (legacy mode):

    x'_i = gamma(s_i) * x_i + beta(s_i)

This is the default implementation. It was used for the main production
PES and MD workflows. It can distinguish different magnetic branches,
but does not impose global spin-reversal symmetry exactly.

## Optional even_pair

The optional single-forward even_pair mode computes

    s_i = m_i / scale[Z_i]
    sbar_i = sum_j w_ij * s_j / (sum_j w_ij + epsilon)
    w_ij = max(0, 1 - r_ij / r_c)**2
    q_i = (s_i**2, s_i * sbar_i)

The neighbor sum includes only elements selected for magnetic
conditioning; its weights depend on geometry and remain differentiable
with respect to atomic coordinates. Globally reversing all input moments
leaves q_i unchanged; changing relative FM/AFM alignment can change q_i.

The magnetic-moment head uses an odd readout on the same odd bases:

    m_pred_i = a_i * s_i + b_i * sbar_i

where a_i and b_i are learned functions of the equivariant MACE features.
For the enabled symmetry option at fixed positions R and magnetic inputs m:

    E(R,m) = E(R,-m)
    F(R,m) = F(R,-m)
    m_pred(R,-m) = -m_pred(R,m)

Under global reversal, these equations are architectural properties
of the model, not post-processing averaged over two full MACE forwards.

## Code organization

Core model:
    mace_film/mace_magmom_film.py

FeN4 trainer and checkpoint-argument restoration:
    mace_film/train_fen4_mace_magmom_film.py

Legacy-compatible ASE MD:
    mace_film/run_md_mace_film.py

Optional trained example, configuration, and input:
    examples/even_pair/

Automated tests:
    tests/test_spin_features.py
    tests/validate_release_models.py
    tests/validate_trained_symmetry.py

The system-specific Mn-Au-O2 and CeO2 source copies remain legacy-only
at present. Do not claim symmetry has been implemented for every copy.

## Checkpoint compatibility and use

Original GitHub example weights belong to the legacy model.
The even_pair architecture has a two-component FiLM input and a
two-coefficient odd moment readout, requiring separately trained weights.
The repo includes a small (single checkpoint) trained demonstration.

Do not enable even_pair for old checkpoints or change a checkpoint's
saved architecture argument to force a different model type.
For externally supplied models use trusted checkpoints only; PyTorch
checkpoint pickle loading can execute arbitrary code.

For dynamics: external magnetic-moment feedback injects an additional
state-update work term. An MHC trajectory with updated moments
should not be claimed to conserve NVE total energy; MTC at fixed
magnetic inputs is the appropriate conservative force-field comparison.
MD drivers should only commit new magnetic predictions once per accepted
integration step and invalidate stale ASE Calculator cache entries.

## Known scope of evidence

- Trained global reversal symmetry verified for one FeN4 example.
- Old released FeN4 PES and MD checkpoint energy, forces and moment
  predictions verified numerically unchanged after integrating the new
  code with legacy as default.
- These checks do not replace full model retraining on a production
  dataset or a statistical DFT accuracy assessment of the even_pair mode.
