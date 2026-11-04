# Validation

The repository provides the following checks:

- tests/test_spin_features.py: feature construction and parity properties.
- tests/validate_release_models.py: energy, force, and moment predictions
  from the supplied FeN4 example checkpoints.
- tests/validate_trained_symmetry.py: global magnetic-reversal symmetry
  and relative-spin sensitivity of the optional trained example.

For the supported Python/MACE environment, tests can be run from the
repository root with upstream MACE available on PYTHONPATH.

~~~bash
python tests/test_spin_features.py
python tests/validate_trained_symmetry.py
python tests/validate_release_models.py --source . --output outputs/validation.json
~~~

The tests check software consistency; they do not replace
full scientific validation against the research datasets.
