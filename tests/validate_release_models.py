#!/usr/bin/env python3
"""Compare released legacy FeN4 checkpoint predictions across source revisions."""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.source.resolve()
sys.path.insert(0, str(root / "mace_film"))

import numpy as np
from ase.io import read
from run_md_mace_film import MACEFilmMDCalculator

out = {}
for suffix, reference in (
    ("md_spp_previous_best", "atomic_energy_linref_md.json"),
    ("pes_afm_energy10_best", "atomic_energy_linref_pes.json"),
):
    frame = read(str(root / "examples/fen4/inputs/afm_template.extxyz"), index=0)
    model = MACEFilmMDCalculator(
        checkpoint=str(root / "examples/fen4/models" / (suffix + ".pt")),
        device="cpu",
        init_magmom=np.asarray(frame.arrays["initial_magmom"]).copy(),
        rollout_magmom=False,
        atomic_energy_linref=str(root / "examples/fen4/inputs" / reference),
    )
    frame.calc = model
    energy = float(frame.get_potential_energy())
    forces = frame.get_forces().tolist()
    moments = np.asarray(model.results["magmoms"]).tolist()
    out[suffix] = {"energy": energy, "forces": forces, "moments": moments}
    print("OK", suffix, "energy", energy, flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out))
print("RESULT_PATH", args.output, flush=True)
