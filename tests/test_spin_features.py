#!/usr/bin/env python3
"""Species-agnostic, multiple-spin-center, sign-symmetry and spatial-gradient tests."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"mace_film"))
from mace_magmom_film import MACEInitialMagmomFiLM

torch.set_default_dtype(torch.float64)
class Fixture:
    magmom_key="initial_magmom"
    magmom_scale=torch.ones(4)
    magmom_input_clip=3.
    magmom_condition_element_mask=torch.tensor([True,True,True,True])
    spin_cutoff=5.0

fixture=Fixture()
pos=torch.tensor([[0.,0.,0.],[1.9,0.,0.],[0.3,2.2,0.],[0.5,0.3,2.5]],requires_grad=True)
idx_src=[];idx_dst=[]
for i in range(4):
 for j in range(4):
  if i!=j:idx_src.append(i);idx_dst.append(j)
edge_index=torch.tensor([idx_src,idx_dst],dtype=torch.long)
lengths=(pos[edge_index[0]]-pos[edge_index[1]]).norm(dim=-1)
node_attrs=torch.eye(4)
mom=torch.tensor([2.,-1.,1.5,0.4])
data={"node_attrs":node_attrs,"initial_magmom":mom,"edge_index":edge_index}
ev,odd=MACEInitialMagmomFiLM._compute_spin_features(fixture,data,lengths)
reversed_data={**data,"initial_magmom":-mom}
ev_rev,odd_rev=MACEInitialMagmomFiLM._compute_spin_features(fixture,reversed_data,lengths)
assert torch.allclose(ev,ev_rev,rtol=0,atol=1e-14)
assert torch.allclose(odd,-odd_rev,rtol=0,atol=1e-14)
mom_local=mom.clone();mom_local[1]*=-1
ev_local,_=MACEInitialMagmomFiLM._compute_spin_features(fixture,{**data,"initial_magmom":mom_local},lengths)
assert (ev_local-ev).abs().max()>1e-6
g=torch.autograd.grad(ev[:,1].square().sum(),pos)[0]
assert torch.isfinite(g).all() and g.norm()>1e-8
zeros={**data,"initial_magmom":torch.zeros_like(mom)}
ev_zero,odd_zero=MACEInitialMagmomFiLM._compute_spin_features(fixture,zeros,lengths)
assert torch.all(ev_zero==0) and torch.all(odd_zero==0)
print("MULTI_MAGNETIC_CENTERS_PASS")
print("GLOBAL_FLIP_EVEN_ERROR",float((ev-ev_rev).abs().max()))
print("GLOBAL_FLIP_ODD_ERROR",float((odd+odd_rev).abs().max()))
print("LOCAL_FLIP_FEATURE_CHANGE",float((ev-ev_local).abs().max()))
print("POSITION_GRADIENT_NORM",float(g.norm()))
print("ZERO_MOMENTS_PASS")
