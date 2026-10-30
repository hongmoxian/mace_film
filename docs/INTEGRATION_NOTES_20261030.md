# Development integration (2026-10-30)

Base public repo: hongmoxian/mace_film, commit 9fce07e.
Branch: integration/even-pair-and-core-sync (not pushed).
The candidate integrates the tested 2026-10-27 single-forward
even_pair model and its training entry point from ocp_mag, while
retaining the legacy mode by default.

Critical compatibility distinction:
- Legacy: signed magnetic FiLM input; original example checkpoints.
- Even_pair: two even magnetic inputs and a two-coefficient odd moment
  head; NEW checkpoints required, not convertible by strict loading.
- Historical enforce_spin_inversion: distinct legacy mode, not the
  new single-forward representation. Do not combine options.

The first integration changes FeN4 model/training source and docs.
Mn-Au-O2 and CeO2 system-specific implementations, original example
checkpoint files, and MD/PES example scripts remain untouched.

Release blockers: checkpoint parity for original legacy models,
full new-model forward/force/oddness regression tests,
MTC/MHC dynamics test, and license/data/README audit.

## Validated on 2026-10-30 with Slurm CPU jobs

Job 718802, COMPLETED (exit 0):
- FeN4 MD published checkpoint: energy max difference 0 eV;
  forces max difference 0 eV/A; predicted moments max difference 0 muB.
- FeN4 PES published checkpoint: same three metrics exactly zero.
- General magnetic descriptor sign symmetry: energy inputs globally even,
  odd bases globally odd, zero moments stable, coordinate gradient nonzero.

Job 718803, COMPLETED (exit 0), preexisting trained even_pair checkpoint:
- Global reversal energy absolute difference: 0 eV.
- Global reversal force maximum difference: 3.5762787e-7 eV/A.
- Global reversal predicted magnetic moment oddness: 0 muB.
- Flipping the Fe moment at unchanged FeN4/O2 geometry changes the
  predicted energy by 0.130615234375 eV (one geometry; not DFT accuracy).

Audit logs, original/updated predictions and scheduler records:
  /home/wuzhihong/software/mace_film_validation_20261030/

Existing Mn-Au-O2/CeO2 codes and original checkpoints remain unchanged.
The optional even_pair feature is currently integrated into the primary
FeN4 model+training source; it has not yet been ported/tested for
the two system-specific modules.
Full train-from-scratch, multi-seed MD, and GitHub CI are not revalidated
by these smoke tests.

Public-release note: this repo currently has no explicit LICENSE file.
Confirm the desired software license and attribution before publishing
an official release tag.
