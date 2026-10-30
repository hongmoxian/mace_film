# Validated scientific software environment

The non-destructive 2026-10-30 CPU validation used:
- Python 3.11.0
- PyTorch 2.3.1
- mace-torch 0.3.14 (MACE source installed separately)
- e3nn 0.4.4
- ASE 3.28.0
- NumPy 1.26.4

Do not assume forward compatibility with MACE v1.0 without changes.
The package imports MACE internals from the 0.3.x series. For release
reproducibility pin an upstream MACE revision or its verified package
version. Full production data is not included in GitHub examples.
