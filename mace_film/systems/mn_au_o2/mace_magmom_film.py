from typing import Dict, List, Optional

import torch
from e3nn import o3

from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph


class MACEInitialMagmomFiLM(ScaleShiftMACE):
    """MACE with atom-wise initial magnetic moment FiLM conditioning.

    The base MACE species embedding is modulated as
        x0 <- gamma(m0) * x0 + beta(m0)
    where m0 is a per-atom initial magnetic moment. This keeps the rest of the
    MACE equivariant message passing unchanged and adds an atom-wise magnetic
    moment readout from the final scalar features.
    """

    def __init__(
        self,
        *args,
        magmom_hidden_dim: int = 64,
        magmom_key: str = "initial_magmom",
        target_magmom_key: str = "target_magmom",
        magmom_scale: Optional[List[float]] = None,
        film_scale: float = 1.0,
        film_use_tanh: bool = False,
        magmom_input_clip: Optional[float] = None,
        film_element_embedding_dim: int = 0,
        magmom_output_normalize: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.magmom_key = magmom_key
        self.target_magmom_key = target_magmom_key
        self.film_scale = float(film_scale)
        self.film_use_tanh = bool(film_use_tanh)
        self.magmom_input_clip = (
            None if magmom_input_clip is None else float(magmom_input_clip)
        )
        self.magmom_output_normalize = bool(magmom_output_normalize)

        # Per-element magnetic moment normalization scale. Each atom's
        # initial_magmom is divided by its element's scale before entering
        # the FiLM network. This unifies the magmom magnitude across element
        # types (e.g. Au ~0.01 vs Mn ~4.2) so the shared FiLM weights are not
        # dominated by large-magmom elements and extrapolation for small-magmom
        # elements stays controlled. The buffer travels with the checkpoint.
        num_elements = len(self.atomic_numbers)
        if magmom_scale is not None:
            if len(magmom_scale) != num_elements:
                raise ValueError(
                    f"magmom_scale length {len(magmom_scale)} != num_elements {num_elements}"
                )
            self.register_buffer(
                "magmom_scale",
                torch.as_tensor(magmom_scale, dtype=torch.get_default_dtype()),
            )
        else:
            self.register_buffer(
                "magmom_scale",
                torch.ones(num_elements, dtype=torch.get_default_dtype()),
            )

        scalar_dim = self.node_embedding.linear.irreps_out.count(o3.Irrep(0, 1))
        self._initial_scalar_dim = int(scalar_dim)
        self.film_element_embedding_dim = int(film_element_embedding_dim)
        if self.film_element_embedding_dim > 0:
            self.film_element_embedding = torch.nn.Embedding(
                num_elements,
                self.film_element_embedding_dim,
            )
        else:
            self.film_element_embedding = None
        film_input_dim = 1 + max(self.film_element_embedding_dim, 0)
        self.magmom_film = torch.nn.Sequential(
            torch.nn.Linear(film_input_dim, magmom_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(magmom_hidden_dim, 2 * self._initial_scalar_dim),
        )

        # Read out local magnetic moments from concatenated scalar features.
        self._readout_scalar_dims: List[int] = []
        final_scalar_dim = 0
        for readout_idx in range(len(self.readouts)):
            feat_idx = -1 if len(self.readouts) == 1 else readout_idx
            scalar_dim = int(
                self.products[feat_idx].linear.irreps_out.count(o3.Irrep(0, 1))
            )
            self._readout_scalar_dims.append(scalar_dim)
            final_scalar_dim += scalar_dim
        if final_scalar_dim <= 0:
            # Fallback for unusual MACE settings.
            final_scalar_dim = self._initial_scalar_dim
        self.magmom_readout = torch.nn.Sequential(
            torch.nn.Linear(int(final_scalar_dim), magmom_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(magmom_hidden_dim, 1),
        )

    def _apply_initial_magmom_film(
        self,
        node_feats: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.magmom_key not in data:
            raise KeyError(
                f"Missing atom-wise magnetic initialization field {self.magmom_key!r}."
            )
        initial_magmom = data[self.magmom_key].to(node_feats.dtype).reshape(-1, 1)
        # Normalize by per-element scale so all element types present a
        # comparable magmom magnitude to the shared FiLM MLP.
        element_idx = data["node_attrs"].argmax(dim=-1)
        scale = self.magmom_scale[element_idx].to(node_feats.dtype).reshape(-1, 1)
        normalized_magmom = initial_magmom / (scale + 1e-8)
        if self.magmom_input_clip is not None:
            normalized_magmom = torch.clamp(
                normalized_magmom,
                min=-self.magmom_input_clip,
                max=self.magmom_input_clip,
            )
        film_input = normalized_magmom
        if self.film_element_embedding is not None:
            elem_feat = self.film_element_embedding(element_idx).to(node_feats.dtype)
            film_input = torch.cat([normalized_magmom, elem_feat], dim=-1)
        gamma_beta = self.magmom_film(film_input)
        gamma_raw, beta_raw = gamma_beta.chunk(2, dim=-1)
        if self.film_use_tanh:
            gamma_raw = torch.tanh(gamma_raw)
            beta_raw = torch.tanh(beta_raw)
        gamma = 1.0 + self.film_scale * gamma_raw
        beta = self.film_scale * beta_raw
        conditioned = node_feats[:, : self._initial_scalar_dim] * gamma + beta
        if node_feats.shape[-1] == self._initial_scalar_dim:
            return conditioned
        return torch.cat([conditioned, node_feats[:, self._initial_scalar_dim :]], dim=-1)

    def _readout_magmom(
        self,
        node_feats_list,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        scalar_features = []
        for readout_idx, scalar_dim in enumerate(self._readout_scalar_dims):
            feat_idx = -1 if len(self._readout_scalar_dims) == 1 else readout_idx
            feats = node_feats_list[feat_idx]
            scalar_features.append(feats[:, :scalar_dim])
        readout = self.magmom_readout(torch.cat(scalar_features, dim=-1)).reshape(-1)
        if not self.magmom_output_normalize:
            return {"magmom": readout}
        element_idx = data["node_attrs"].argmax(dim=-1)
        scale = self.magmom_scale[element_idx].to(readout.dtype).reshape(-1)
        return {
            "magmom": readout * (scale + 1e-8),
            "magmom_normalized": readout,
        }

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # This mirrors ScaleShiftMACE.forward with a single insertion point:
        # the initial node embedding is FiLM-conditioned before message passing.
        from mace.tools.scatter import scatter_sum

        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(vectors.dtype)

        node_feats = self.node_embedding(data["node_attrs"])
        node_feats = self._apply_initial_magmom_film(node_feats, data)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths,
            data["node_attrs"],
            data["edge_index"],
            self.atomic_numbers,
        )

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths,
                data["node_attrs"],
                data["edge_index"],
                self.atomic_numbers,
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_es_list = [pair_node_energy]
        node_feats_list = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=inter_e,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )

        atomic_virials = None
        atomic_stresses = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )

        magmom_outputs = self._readout_magmom(node_feats_list, data)
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
            "model_magmom_scale": self.magmom_scale,
            **magmom_outputs,
        }
