# Reproducibility

This repository is a software release with representative examples.
It does not include every trained model or numerical result used for
the manuscript figures. Reproducing a particular published result
requires its matching model weights, input configurations, simulation
parameters, and numerical data from the separate research archive.

For FeN4/O2, the example PES script evaluates two magnetic branches
using the same supplied checkpoint. The original manuscript PES data,
however, used separate fitted weights for the AFM and FM branches.
The example should not be described as an exact recreation of that plot.

The published FeN4 and Mn-Au-O2 MD distribution comparisons used
selected three-replica subsets from five available runs. The corresponding
selection and statistical protocol, along with source trajectories,
are necessary for independent verification. They are not supplied
in the software-only examples.

The optional symmetry checkpoint is distinct from production
MD and PES models. The source and methodology are described in
docs/MAGNETIC_SYMMETRY.md.
