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
        conditioning_mode: str = "film",
        film_scale: float = 1.0,
        film_use_tanh: bool = False,
        magmom_input_clip: Optional[float] = None,
        film_element_embedding_dim: int = 0,
        enforce_spin_inversion: bool = False,
        spin_feature_mode: str = "legacy",
        predict_magmom: bool = True,
        compact_conditioning: bool = False,
        magmom_condition_atomic_numbers: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.magmom_key = magmom_key
        self.target_magmom_key = target_magmom_key
        self.conditioning_mode = str(conditioning_mode or "film").strip().lower()
        self.film_scale = float(film_scale)
        self.film_use_tanh = bool(film_use_tanh)
        self.enforce_spin_inversion = bool(enforce_spin_inversion)
        self.spin_feature_mode = str(spin_feature_mode).strip().lower()
        if self.spin_feature_mode not in {"legacy", "even_pair"}:
            raise ValueError("spin_feature_mode must be legacy or even_pair")
        if self.spin_feature_mode == "even_pair" and self.enforce_spin_inversion:
            raise ValueError("even_pair already enforces global reversal: disable double-forward averaging")
        if self.spin_feature_mode == "even_pair" and self.conditioning_mode == "none":
            raise ValueError("even_pair requires active magnetic FiLM or concat conditioning")
        self.spin_cutoff = float(kwargs.get("r_max", 5.0))
        if self.spin_cutoff <= 0:
            raise ValueError("r_max must be positive")
        self.predict_magmom = bool(predict_magmom)
        self.compact_conditioning = bool(compact_conditioning)
        atomic_numbers = [int(z) for z in self.atomic_numbers]
        if magmom_condition_atomic_numbers is None:
            condition_element_mask = torch.ones(len(atomic_numbers), dtype=torch.bool)
        else:
            selected_numbers = {int(z) for z in magmom_condition_atomic_numbers}
            missing = sorted(selected_numbers.difference(atomic_numbers))
            if missing:
                raise ValueError(
                    "Magmom conditioning elements are absent from the model species: "
                    f"Z={missing}"
                )
            condition_element_mask = torch.tensor(
                [z in selected_numbers for z in atomic_numbers], dtype=torch.bool
            )
        self.register_buffer(
            "magmom_condition_element_mask", condition_element_mask, persistent=False
        )
        self.magmom_input_clip = (
            None if magmom_input_clip is None else float(magmom_input_clip)
        )
        if self.conditioning_mode not in {"none", "film", "residual_film", "concat"}:
            raise ValueError(
                f"Unsupported conditioning_mode={conditioning_mode!r}. "
                "Expected one of 'none', 'film', 'residual_film', 'concat'."
            )

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
        if self.film_element_embedding_dim > 0 and self.conditioning_mode != "none":
            self.film_element_embedding = torch.nn.Embedding(
                num_elements,
                self.film_element_embedding_dim,
            )
        else:
            self.film_element_embedding = None
        film_input_dim = (2 if self.spin_feature_mode == "even_pair" else 1) + max(self.film_element_embedding_dim, 0)
        # Keep legacy initialization order and checkpoint keys unless opted out.
        if self.conditioning_mode == "none" or (
            self.conditioning_mode == "concat" and self.compact_conditioning
        ):
            self.magmom_film = None
        else:
            self.magmom_film = torch.nn.Sequential(
                torch.nn.Linear(film_input_dim, magmom_hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(magmom_hidden_dim, 2 * self._initial_scalar_dim),
            )
        if self.conditioning_mode == "concat":
            self.magmom_concat_proj = torch.nn.Sequential(
                torch.nn.Linear(self._initial_scalar_dim + film_input_dim, magmom_hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(magmom_hidden_dim, self._initial_scalar_dim),
            )
        else:
            self.magmom_concat_proj = None

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
        if self.predict_magmom:
            self.magmom_readout = torch.nn.Sequential(
                torch.nn.Linear(int(final_scalar_dim), magmom_hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(magmom_hidden_dim, 2 if self.spin_feature_mode == "even_pair" else 1),
            )
        else:
            self.magmom_readout = None

    def _compute_spin_features(
        self,
        data: Dict[str, torch.Tensor],
        lengths: torch.Tensor,
    ):
        """Time-reversal EVEN FiLM features and ODD moment bases.

        No fixed chemical species or spin-reference atom is required.
        q_i = m_i * weighted_mean_j(m_j), both m normalized/optionally clipped.
        Distances and their gradients are retained for conservative forces.
        """
        node_attrs = data["node_attrs"]
        element_idx = node_attrs.argmax(dim=-1)
        m = data[self.magmom_key].reshape(-1).to(node_attrs.dtype)
        scale = self.magmom_scale[element_idx].to(m.dtype)
        norm = m / (scale + 1e-8)
        if self.magmom_input_clip is not None:
            norm = norm.clamp(-self.magmom_input_clip, self.magmom_input_clip)
        mask = self.magmom_condition_element_mask[element_idx].to(m.dtype)
        norm = norm * mask

        # edge_index[0] -> edge_index[1]; accumulate incoming neighbor moments.
        src, dst = data["edge_index"]
        r = lengths.reshape(-1).to(norm.dtype)
        if r.numel() != src.numel():
            raise ValueError("lengths/edge_index size mismatch")
        weight = torch.clamp(1.0 - r / self.spin_cutoff, min=0.0).square()
        weight = weight * mask[src] * mask[dst]
        weighted_m = weight * norm[src]
        n = norm.shape[0]
        neighbor_sum = torch.zeros(n, dtype=norm.dtype, device=norm.device)
        weight_sum = torch.zeros_like(neighbor_sum)
        neighbor_sum.index_add_(0, dst, weighted_m)
        weight_sum.index_add_(0, dst, weight)
        neighbor_m = neighbor_sum / (weight_sum + 1e-8)

        even_features = torch.stack((norm.square(), norm * neighbor_m), dim=-1)
        odd_bases = torch.stack((norm, neighbor_m), dim=-1)
        return even_features, odd_bases

    def _apply_initial_magmom_film(
        self,
        node_feats: torch.Tensor,
        data: Dict[str, torch.Tensor],
        even_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.conditioning_mode == "none":
            return node_feats
        if self.magmom_key not in data:
            raise KeyError(
                f"Missing atom-wise magnetic initialization field {self.magmom_key!r}."
            )
        initial_magmom = data[self.magmom_key].to(node_feats.dtype).reshape(-1, 1)
        element_idx = data["node_attrs"].argmax(dim=-1)
        scale = self.magmom_scale[element_idx].to(node_feats.dtype).reshape(-1, 1)
        normalized_magmom = initial_magmom / (scale + 1e-8)
        if self.magmom_input_clip is not None:
            normalized_magmom = torch.clamp(
                normalized_magmom,
                min=-self.magmom_input_clip,
                max=self.magmom_input_clip,
            )
        if self.spin_feature_mode == "even_pair":
            if even_features is None:
                raise ValueError("Missing even magnetic features")
            film_input = even_features.to(node_feats.dtype)
        else:
            film_input = normalized_magmom
        if self.film_element_embedding is not None:
            elem_feat = self.film_element_embedding(element_idx).to(node_feats.dtype)
            film_input = torch.cat([film_input, elem_feat], dim=-1)

        if self.conditioning_mode == "concat":
            conditioned_scalar = self.magmom_concat_proj(
                torch.cat([node_feats[:, : self._initial_scalar_dim], film_input], dim=-1)
            )
            selected_atoms = self.magmom_condition_element_mask[element_idx].unsqueeze(-1)
            conditioned_scalar = torch.where(
                selected_atoms,
                conditioned_scalar,
                node_feats[:, : self._initial_scalar_dim],
            )
            if node_feats.shape[-1] == self._initial_scalar_dim:
                return conditioned_scalar
            return torch.cat([conditioned_scalar, node_feats[:, self._initial_scalar_dim :]], dim=-1)

        gamma_beta = self.magmom_film(film_input)
        gamma_raw, beta_raw = gamma_beta.chunk(2, dim=-1)
        if self.film_use_tanh:
            gamma_raw = torch.tanh(gamma_raw)
            beta_raw = torch.tanh(beta_raw)
        gamma = 1.0 + self.film_scale * gamma_raw
        beta = self.film_scale * beta_raw
        if self.conditioning_mode == "residual_film":
            conditioned = node_feats[:, : self._initial_scalar_dim] + (
                node_feats[:, : self._initial_scalar_dim] * gamma + beta
            ) * self.film_scale
        else:
            conditioned = node_feats[:, : self._initial_scalar_dim] * gamma + beta
        selected_atoms = self.magmom_condition_element_mask[element_idx].unsqueeze(-1)
        conditioned = torch.where(
            selected_atoms, conditioned, node_feats[:, : self._initial_scalar_dim]
        )
        if node_feats.shape[-1] == self._initial_scalar_dim:
            return conditioned
        return torch.cat([conditioned, node_feats[:, self._initial_scalar_dim :]], dim=-1)

    def _readout_magmom(self, node_feats_list) -> torch.Tensor:
        scalar_features = []
        for readout_idx, scalar_dim in enumerate(self._readout_scalar_dims):
            feat_idx = -1 if len(self._readout_scalar_dims) == 1 else readout_idx
            feats = node_feats_list[feat_idx]
            scalar_features.append(feats[:, :scalar_dim])
        values = self.magmom_readout(torch.cat(scalar_features, dim=-1))
        return values if self.spin_feature_mode == "even_pair" else values.reshape(-1)

    def _forward_single(
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

        even_features = None
        odd_bases = None
        if self.spin_feature_mode == "even_pair":
            even_features, odd_bases = self._compute_spin_features(data, lengths)
        node_feats = self.node_embedding(data["node_attrs"])
        node_feats = self._apply_initial_magmom_film(node_feats, data, even_features)
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

        if getattr(self, "predict_magmom", True):
            magmom_values = self._readout_magmom(node_feats_list)
            if self.spin_feature_mode == "even_pair":
                # Even coefficients times odd magnetic bases -> strictly odd readout.
                magmom_values = torch.sum(magmom_values * odd_bases, dim=-1)
        else:
            magmom_values = None

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
            "magmom": magmom_values,
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
        """Evaluate with optional full-network global spin-inversion symmetry."""
        if self.spin_feature_mode == "even_pair":
            # One forward; even FiLM inputs ensure E,F are globally spin-inversion even.
            return self._forward_single(
                data,
                training=training,
                compute_force=compute_force,
                compute_virials=compute_virials,
                compute_stress=compute_stress,
                compute_displacement=compute_displacement,
                compute_hessian=compute_hessian,
                compute_edge_forces=compute_edge_forces,
                compute_atomic_stresses=compute_atomic_stresses,
                lammps_mliap=lammps_mliap,
            )
        if not self.enforce_spin_inversion:
            return self._forward_single(
                data,
                training=training,
                compute_force=compute_force,
                compute_virials=compute_virials,
                compute_stress=compute_stress,
                compute_displacement=compute_displacement,
                compute_hessian=compute_hessian,
                compute_edge_forces=compute_edge_forces,
                compute_atomic_stresses=compute_atomic_stresses,
                lammps_mliap=lammps_mliap,
            )
        if self.magmom_key not in data:
            raise KeyError(
                f"Missing atom-wise magnetic initialization field {self.magmom_key!r}."
            )
        inverted = dict(data)
        inverted[self.magmom_key] = -data[self.magmom_key]
        positive = self._forward_single(
            data,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            compute_atomic_stresses=compute_atomic_stresses,
            lammps_mliap=lammps_mliap,
        )
        negative = self._forward_single(
            inverted,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            compute_atomic_stresses=compute_atomic_stresses,
            lammps_mliap=lammps_mliap,
        )
        output = {}
        for key, value in positive.items():
            other = negative.get(key)
            if key == "magmom" and torch.is_tensor(value) and torch.is_tensor(other):
                output[key] = 0.5 * (value - other)
            elif torch.is_tensor(value) and torch.is_tensor(other):
                output[key] = 0.5 * (value + other)
            else:
                output[key] = value
        return output
