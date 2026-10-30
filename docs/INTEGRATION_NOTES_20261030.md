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

Initial integration preserved legacy source; the second integration
imports the newer FeN4 magnetic-update MD driver and the Mn-Au-O2 PES
scanner with portable in-repository imports and legacy CLI alias.
The existing system-specific Mn-Au-O2 model/train/MD and CeO2 code,
along with their old example checkpoints, remain unchanged.
The Mn-Au-O2 published PES example explicitly selects --mode mtc.

Pre-release open work: confirm the desired software LICENSE and
attribution; optional automated CI; optional wider MD/PES regression
across multiple trajectories and all material systems.
Checkpoint parity, feature-level symmetry, bundled trained-model
symmetry and short FeN4 MHC smoke tests were completed below.

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

Validation logs, original/updated predictions and scheduler records
are retained on the development server, outside this GitHub package.

Existing Mn-Au-O2/CeO2 model definitions and old checkpoints remain
unchanged; Mn-Au-O2 PES scanning and FeN4 MD drivers are updated.
The optional even_pair feature is currently integrated into the primary
FeN4 model+training source; it has not yet been ported/tested for
the two system-specific modules.
Full train-from-scratch, multi-seed MD, and GitHub CI are not revalidated
by these smoke tests.

Public-release note: this repo currently has no explicit LICENSE file.
Confirm the desired software license and attribution before publishing
an official release tag.

## Follow-up integration tests (2026-10-30)

Job 718814, COMPLETED (exit 0):
- FeN4 legacy MD and PES checkpoints: every tested energy, force and
  predicted moment equals the GitHub 9fce07e baseline exactly.
- Feature-level global reversal, relative spin and spatial gradients:
  pass.
- Bundled even_pair checkpoint and 1-geometry input: global energy
  even residual 0 eV, maximum force residual 3.5762787e-7 eV/Angstrom,
  predicted-moment odd residual 0 muB.
- Single-local-spin reversal changes model energy for the bundled
  example by 0.17926025390625 eV (not DFT accuracy evidence).
- Updated Mn-Au-O2 PES CLI: pass.
- Two-step FeN4 NVE magnetic-feedback MD smoke test: pass.
  This test demonstrates execution, NOT NVE conservation.

Job 718817, COMPLETED (exit 0):
- FeN4 12-frame sample training quickstart: one CPU training epoch,
  validation and test completed, with model and metrics saved.
- This tiny sample is a format/CLI test, not a credible training set.

The optional figure in figures/workflow.png was regenerated from the
user-supplied figure layout, with editable SVG and Python source
included. Any new visual changes can be made in tools/build_workflow.py.

The bundled even_pair config has placeholder paths, never the
absolute experimental filesystem paths. Full training data
is not included.

Job 718820, COMPLETED (exit 0):
- Updated Mn-Au-O2 PES script, with original released checkpoint and
  fixed AFM template under MTC mode, completed a single 2.0 Angstrom
  distance-point evaluation and wrote CSV, VASP and plot artifacts.
- The 0 optimizer-update check is a runnable end-to-end smoke test,
  NOT validation of a converged constrained PES.
- MTC now permits fixed branch templates as well as fixed MAGMOM
  rules, without allowing geometry-dependent per-scan DFT seeds.
