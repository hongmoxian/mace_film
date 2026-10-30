#!/usr/bin/env python3
"""Verify trained single-forward symmetry with unchanged DFT geometry."""
import sys
from pathlib import Path
import numpy as np
from ase.io import read
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'mace_film'))
from run_md_mace_film import MACEFilmMDCalculator
ckpt=root/'examples/even_pair/models/best.pt'
ref=str(root/'examples/even_pair/inputs/atomic_energy_linref.json')
initial=read(str(root/'examples/even_pair/inputs/first_real_frame.extxyz'),0)
m=np.asarray(initial.arrays['initial_magmom'],dtype=float)
def evaluate(mom):
 atoms=initial.copy()
 calc=MACEFilmMDCalculator(
  checkpoint=str(ckpt),device='cpu',init_magmom=mom,
  rollout_magmom=False,atomic_energy_linref=ref)
 atoms.calc=calc
 return float(atoms.get_potential_energy()),atoms.get_forces().copy(),np.asarray(calc.results['magmoms']).copy()
ep,fp,mp=evaluate(m)
en,fn,mn=evaluate(-m)
print('GLOBAL_ENERGY_EVEN_RESIDUAL_eV',abs(ep-en),flush=True)
print('GLOBAL_FORCE_EVEN_RESIDUAL_eV_A',np.max(np.abs(fp-fn)),flush=True)
print('GLOBAL_MAGMOM_ODD_RESIDUAL_muB',np.max(np.abs(mp+mn)),flush=True)
assert abs(ep-en)<2e-4
assert np.max(np.abs(fp-fn))<1e-4
assert np.max(np.abs(mp+mn))<2e-5
mnew=m.copy()
fe=np.where(initial.numbers==26)[0]
assert len(fe)==1
mnew[fe[0]]=-mnew[fe[0]]
ec,fc,mc=evaluate(mnew)
delta=ec-ep
print('FIXED_GEOMETRY_RELATIVE_SPIN_DELTA_eV',delta,flush=True)
assert abs(delta)>1e-8
print('TRAINED_SYMMETRY_PASS',flush=True)
