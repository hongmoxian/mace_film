#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase import io, units
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace import data as mace_data
from mace.tools import torch_geometric
from mace.tools.utils import AtomicNumberTable

from train_ceo2_mace_magmom_film import _build_model

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

try:
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
except ModuleNotFoundError:
    NoseHooverChainNVT = None


def _expand_magmom_rule(rule, expected_natoms):
    values = []
    for token in str(rule).split():
        if not token:
            continue
        if "*" in token:
            count_raw, value_raw = token.split("*", 1)
            values.extend([float(value_raw)] * int(count_raw))
        else:
            values.append(float(token))
    if len(values) != int(expected_natoms):
        raise ValueError(
            f"MAGMOM rule length mismatch: expected {expected_natoms}, "
            f"got {len(values)} from {rule!r}"
        )
    return np.asarray(values, dtype=float)


def _safe_get_frame_magmoms(atoms):
    try:
        magmoms = atoms.get_magnetic_moments()
        if magmoms is not None and len(magmoms) == len(atoms):
            magmoms = np.asarray(magmoms, dtype=float)
            if np.any(np.abs(magmoms) > 1e-12):
                return magmoms
    except Exception:
        pass
    for key in ("initial_magmom", "target_magmom", "magmoms"):
        if key in atoms.arrays and len(atoms.arrays[key]) == len(atoms):
            magmoms = np.asarray(atoms.arrays[key], dtype=float)
            if np.any(np.abs(magmoms) > 1e-12):
                return magmoms
    if "magmoms" in atoms.arrays and len(atoms.arrays["magmoms"]) == len(atoms):
        return np.asarray(atoms.arrays["magmoms"], dtype=float)
    try:
        magmoms = atoms.get_initial_magnetic_moments()
        if magmoms is not None and len(magmoms) == len(atoms):
            magmoms = np.asarray(magmoms, dtype=float)
            if np.any(np.abs(magmoms) > 1e-12):
                return magmoms
    except Exception:
        pass
    return None


def resolve_initial_magmoms(structure_path, structure_index):
    atoms = io.read(structure_path, index=structure_index)
    try:
        prev_index = int(structure_index) - 1
    except Exception:
        prev_index = None

    if prev_index is not None and prev_index >= 0:
        prev_atoms = io.read(structure_path, index=prev_index)
        current_state = atoms.info.get("state_id")
        prev_state = prev_atoms.info.get("state_id")
        same_state = (
            current_state is None
            or prev_state is None
            or int(current_state) == int(prev_state)
        )
        if same_state:
            prev_magmoms = _safe_get_frame_magmoms(prev_atoms)
            if prev_magmoms is not None and np.any(np.abs(prev_magmoms) > 1e-12):
                return prev_magmoms, f"previous frame {prev_index}"

    current_magmoms = _safe_get_frame_magmoms(atoms)
    if current_magmoms is not None and np.any(np.abs(current_magmoms) > 1e-12):
        return current_magmoms, f"current frame {structure_index}"

    return np.asarray(atoms.get_initial_magnetic_moments(), dtype=float), (
        "initial_magnetic_moments fallback"
    )


def resolve_initial_magmoms_with_override(
    structure_path,
    structure_index,
    init_magmom_mode="trajectory",
    init_magmom_spec=None,
):
    atoms = io.read(structure_path, index=structure_index)
    mode = str(init_magmom_mode or "trajectory").strip().lower()
    if mode in {"trajectory", "auto"}:
        return resolve_initial_magmoms(structure_path, structure_index)
    if mode in {"rule", "magmom_rule"}:
        if not init_magmom_spec:
            raise ValueError("--init-magmom-mode rule requires --init-magmom-spec")
        return _expand_magmom_rule(init_magmom_spec, expected_natoms=len(atoms)), (
            f"rule:{init_magmom_spec}"
        )
    if mode in {"constant", "fixed"}:
        if init_magmom_spec is None:
            raise ValueError("--init-magmom-mode constant requires --init-magmom-spec")
        value = float(init_magmom_spec)
        return np.full(len(atoms), value, dtype=float), f"constant:{value}"
    raise ValueError(f"Unsupported --init-magmom-mode: {init_magmom_mode}")


def _get_ce_indices_from_atoms(atoms):
    return [idx for idx, z in enumerate(atoms.get_atomic_numbers()) if int(z) == 58]


def _to_jsonable(value):
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _topk_indices(values, k):
    if k <= 0 or values.numel() == 0:
        return []
    order = torch.argsort(values, descending=True)
    return [int(idx.item()) for idx in order[: min(k, values.numel())]]


def _sum_abs_for_indices(values, indices):
    if not indices:
        return 0.0
    return float(torch.abs(values[indices]).sum().item())


class MACEFilmMDCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint,
        device="cuda",
        init_magmom=None,
        rollout_magmom=True,
        atomic_numbers=None,
        constrain_active_ce=False,
        fixed_active_count=None,
        active_count_init_threshold=0.8,
        active_set_stickiness=0.0,
        active_set_confirm_steps=0,
        active_magmom_mixing_alpha=0.7,
        active_magmom_smooth_threshold=None,
        transfer_magmom_on_active_hop=False,
        inactive_magmom_scale=0.05,
        active_magmom_target_mode="none",
        active_magmom_target_value=None,
        active_magmom_target_tolerance_rel=0.1,
        active_magmom_target_relax_power=0.35,
        ce_active_threshold=0.5,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location=self.device)
        args = SimpleNamespace(**payload["args"])
        z_values = atomic_numbers
        if z_values is None:
            z_values = getattr(args, "atomic_numbers", None)
        if z_values is None:
            z_values = getattr(args, "z_table", None)
        if z_values is None:
            z_values = [8, 58, 79]
        self.z_table = AtomicNumberTable([int(z) for z in z_values])
        self.model = _build_model(args, self.z_table).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.cutoff = float(args.cutoff)
        self.ce_active_loss_type = str(
            getattr(args, "ce_active_loss_type", "focal_bce")
        )
        self.rollout_magmom = bool(rollout_magmom)
        self.constrain_active_ce = bool(constrain_active_ce)
        self.fixed_active_count = fixed_active_count
        self.active_count_init_threshold = float(active_count_init_threshold)
        self.active_set_stickiness = float(active_set_stickiness)
        self.active_set_confirm_steps = int(active_set_confirm_steps)
        self.active_magmom_mixing_alpha = float(active_magmom_mixing_alpha)
        self.active_magmom_smooth_threshold = (
            None
            if active_magmom_smooth_threshold is None
            else float(active_magmom_smooth_threshold)
        )
        self.transfer_magmom_on_active_hop = bool(transfer_magmom_on_active_hop)
        self.inactive_magmom_scale = float(inactive_magmom_scale)
        self.active_magmom_target_mode = str(active_magmom_target_mode)
        self.active_magmom_target_value = active_magmom_target_value
        self.active_magmom_target_tolerance_rel = float(
            active_magmom_target_tolerance_rel
        )
        self.active_magmom_target_relax_power = float(active_magmom_target_relax_power)
        self.ce_active_threshold = float(ce_active_threshold)
        self.current_magmom = (
            None
            if init_magmom is None
            else torch.as_tensor(init_magmom, dtype=torch.get_default_dtype())
        )
        self.pending_magmom = None
        self.ce_indices = None
        self.active_set_local_indices = None
        self.pending_active_set_local_indices = None
        self.pending_active_set_counter = 0
        self.active_magmom_target_sum = None
        if self.active_magmom_target_mode == "fixed" and active_magmom_target_value is None:
            raise ValueError(
                "active_magmom_target_mode='fixed' requires "
                "--active-magmom-target-value."
            )

    def _initialize_active_constraint_state(self, atoms):
        if self.ce_indices is not None:
            return
        self.ce_indices = _get_ce_indices_from_atoms(atoms)
        if not self.ce_indices:
            raise ValueError("No Ce atoms found while initializing active constraints.")

        ce_init_mag = self.current_magmom[self.ce_indices].clone().to(torch.float32)
        init_abs = torch.abs(ce_init_mag)
        if self.fixed_active_count is None:
            inferred_count = int((init_abs >= self.active_count_init_threshold).sum().item())
            self.fixed_active_count = max(1, inferred_count)
        self.fixed_active_count = min(int(self.fixed_active_count), len(self.ce_indices))

        if self.fixed_active_count > 0:
            self.active_set_local_indices = _topk_indices(init_abs, self.fixed_active_count)
        else:
            self.active_set_local_indices = []

        if self.active_magmom_target_mode == "init_sum_abs":
            self.active_magmom_target_sum = _sum_abs_for_indices(
                ce_init_mag, self.active_set_local_indices
            )
        elif self.active_magmom_target_mode == "fixed":
            self.active_magmom_target_sum = float(self.active_magmom_target_value)
        else:
            self.active_magmom_target_sum = None

    def _candidate_active_set_local_indices(self, ce_probs):
        scores = ce_probs.clone()
        if self.active_set_local_indices:
            for idx in self.active_set_local_indices:
                scores[idx] += self.active_set_stickiness
        return _topk_indices(scores, self.fixed_active_count)

    def _update_active_set_local_indices(self, candidate_local_indices):
        candidate_key = tuple(sorted(candidate_local_indices))
        current_key = tuple(sorted(self.active_set_local_indices or []))
        previous_local_indices = list(self.active_set_local_indices or [])
        previous_set = set(previous_local_indices)
        candidate_set = set(candidate_local_indices)
        old_local_indices = sorted(previous_set - candidate_set)
        new_local_indices = sorted(candidate_set - previous_set)

        if candidate_key == current_key:
            self.pending_active_set_local_indices = None
            self.pending_active_set_counter = 0
            return {
                "changed": False,
                "pending": False,
                "transition": "stable",
                "active_local_indices": previous_local_indices,
                "previous_local_indices": previous_local_indices,
                "candidate_local_indices": list(candidate_local_indices),
                "old_local_indices": [],
                "new_local_indices": [],
                "rejected": False,
            }

        if len(old_local_indices) > 1 or len(new_local_indices) > 1:
            self.pending_active_set_local_indices = None
            self.pending_active_set_counter = 0
            return {
                "changed": False,
                "pending": False,
                "transition": "rejected_multi_change",
                "active_local_indices": previous_local_indices,
                "previous_local_indices": previous_local_indices,
                "candidate_local_indices": list(candidate_local_indices),
                "old_local_indices": old_local_indices,
                "new_local_indices": new_local_indices,
                "rejected": True,
            }

        if self.active_set_confirm_steps <= 0:
            self.active_set_local_indices = list(candidate_local_indices)
            self.pending_active_set_local_indices = None
            self.pending_active_set_counter = 0
            if len(old_local_indices) == 1 and len(new_local_indices) == 1:
                transition = "single_replacement"
            elif len(old_local_indices) == 0 and len(new_local_indices) == 1:
                transition = "single_addition"
            elif len(old_local_indices) == 1 and len(new_local_indices) == 0:
                transition = "single_removal"
            else:
                transition = "set_change"
            return {
                "changed": True,
                "pending": False,
                "transition": transition,
                "active_local_indices": list(candidate_local_indices),
                "previous_local_indices": previous_local_indices,
                "candidate_local_indices": list(candidate_local_indices),
                "old_local_indices": old_local_indices,
                "new_local_indices": new_local_indices,
                "rejected": False,
            }

        if self.pending_active_set_local_indices == candidate_key:
            self.pending_active_set_counter += 1
        else:
            self.pending_active_set_local_indices = candidate_key
            self.pending_active_set_counter = 1

        if self.pending_active_set_counter > self.active_set_confirm_steps:
            self.active_set_local_indices = list(candidate_local_indices)
            self.pending_active_set_local_indices = None
            self.pending_active_set_counter = 0
            if len(old_local_indices) == 1 and len(new_local_indices) == 1:
                transition = "single_replacement"
            elif len(old_local_indices) == 0 and len(new_local_indices) == 1:
                transition = "single_addition"
            elif len(old_local_indices) == 1 and len(new_local_indices) == 0:
                transition = "single_removal"
            else:
                transition = "set_change"
            return {
                "changed": True,
                "pending": False,
                "transition": transition,
                "active_local_indices": list(candidate_local_indices),
                "previous_local_indices": previous_local_indices,
                "candidate_local_indices": list(candidate_local_indices),
                "old_local_indices": old_local_indices,
                "new_local_indices": new_local_indices,
                "rejected": False,
            }

        return {
            "changed": False,
            "pending": True,
            "transition": "pending_set_change",
            "active_local_indices": previous_local_indices,
            "previous_local_indices": previous_local_indices,
            "candidate_local_indices": list(candidate_local_indices),
            "old_local_indices": old_local_indices,
            "new_local_indices": new_local_indices,
            "rejected": False,
        }

    def _project_rollout_magmom(self, atoms, pred_magmom, ce_probs):
        self._initialize_active_constraint_state(atoms)
        candidate_local_indices = self._candidate_active_set_local_indices(ce_probs)
        transition_meta = self._update_active_set_local_indices(candidate_local_indices)

        active_local_indices = list(transition_meta["active_local_indices"])
        previous_local_indices = list(transition_meta["previous_local_indices"])
        old_local_indices = list(transition_meta["old_local_indices"])
        new_local_indices = list(transition_meta["new_local_indices"])
        transition = str(transition_meta["transition"])

        next_magmom = pred_magmom.clone()
        prev_ce_mag = self.current_magmom[self.ce_indices].clone().to(torch.float32)
        ce_pred = pred_magmom[self.ce_indices].clone().to(torch.float32)
        ce_next = ce_pred.clone() * self.inactive_magmom_scale
        active_sum_before_target = _sum_abs_for_indices(ce_next, active_local_indices)
        lower_bound = None
        upper_bound = None
        target_scale = 1.0
        target_applied = False
        smooth_applied_count = 0

        active_mask = torch.zeros(len(self.ce_indices), dtype=torch.bool)
        if active_local_indices:
            active_mask[active_local_indices] = True
            if (
                self.transfer_magmom_on_active_hop
                and transition == "single_replacement"
                and len(old_local_indices) == 1
                and len(new_local_indices) == 1
            ):
                keep_local_indices = sorted(
                    set(active_local_indices) & set(previous_local_indices)
                )
                if keep_local_indices:
                    keep_mask = torch.zeros(len(self.ce_indices), dtype=torch.bool)
                    keep_mask[keep_local_indices] = True
                    ce_next[keep_mask] = ce_pred[keep_mask]

                old_idx = old_local_indices[0]
                new_idx = new_local_indices[0]
                old_abs = torch.abs(prev_ce_mag[old_idx])
                new_pred = ce_pred[new_idx]
                new_sign = torch.sign(new_pred)
                if float(torch.abs(new_sign).item()) < 1e-12:
                    new_sign = torch.sign(prev_ce_mag[old_idx])
                if float(torch.abs(new_sign).item()) < 1e-12:
                    new_sign = new_pred.new_tensor(1.0)
                ce_next[new_idx] = new_sign * old_abs
            else:
                ce_next[active_mask] = ce_pred[active_mask]
                if self.active_magmom_smooth_threshold is not None:
                    active_diff = torch.abs(ce_pred[active_mask] - prev_ce_mag[active_mask])
                    smooth_mask = active_diff > self.active_magmom_smooth_threshold
                    smooth_applied_count = int(smooth_mask.sum().item())
                    if smooth_applied_count > 0:
                        blended = (
                            self.active_magmom_mixing_alpha * ce_pred[active_mask]
                            + (1.0 - self.active_magmom_mixing_alpha)
                            * prev_ce_mag[active_mask]
                        )
                        ce_next_active = ce_next[active_mask]
                        ce_next_active[smooth_mask] = blended[smooth_mask]
                        ce_next[active_mask] = ce_next_active

                if (
                    self.active_magmom_target_sum is not None
                    and self.active_magmom_target_sum > 1e-12
                ):
                    active_sum = torch.abs(ce_next[active_mask]).sum()
                    lower_bound = self.active_magmom_target_sum * (
                        1.0 - self.active_magmom_target_tolerance_rel
                    )
                    upper_bound = self.active_magmom_target_sum * (
                        1.0 + self.active_magmom_target_tolerance_rel
                    )
                    active_sum_before_target = float(active_sum.item())
                    if float(active_sum.item()) > 1e-12 and (
                        active_sum_before_target < lower_bound
                        or active_sum_before_target > upper_bound
                    ):
                        raw_scale = self.active_magmom_target_sum / active_sum_before_target
                        target_scale = raw_scale ** self.active_magmom_target_relax_power
                        ce_next[active_mask] = ce_next[active_mask] * target_scale
                        target_applied = True

        next_magmom[self.ce_indices] = ce_next
        active_global_indices = [self.ce_indices[idx] for idx in active_local_indices]
        candidate_global_indices = [self.ce_indices[idx] for idx in candidate_local_indices]
        previous_global_indices = [self.ce_indices[idx] for idx in previous_local_indices]
        old_global_indices = [self.ce_indices[idx] for idx in old_local_indices]
        new_global_indices = [self.ce_indices[idx] for idx in new_local_indices]
        return next_magmom, {
            "active_set_indices_1based": [idx + 1 for idx in active_global_indices],
            "active_set_candidate_indices_1based": [idx + 1 for idx in candidate_global_indices],
            "active_set_previous_indices_1based": [idx + 1 for idx in previous_global_indices],
            "active_set_changed": bool(transition_meta["changed"]),
            "active_set_pending": bool(transition_meta["pending"]),
            "active_set_transition": transition,
            "active_set_rejected": bool(transition_meta["rejected"]),
            "active_set_old_indices_1based": [idx + 1 for idx in old_global_indices],
            "active_set_new_indices_1based": [idx + 1 for idx in new_global_indices],
            "active_set_confirm_counter": self.pending_active_set_counter,
            "active_count_fixed": self.fixed_active_count,
            "active_target_sum_abs_magmom": self.active_magmom_target_sum,
            "active_sum_abs_magmom_pred": _sum_abs_for_indices(ce_pred, active_local_indices),
            "active_sum_abs_magmom_input": _sum_abs_for_indices(ce_next, active_local_indices),
            "active_target_lower_bound": lower_bound,
            "active_target_upper_bound": upper_bound,
            "active_target_scale_applied": target_scale,
            "active_target_correction_applied": target_applied,
            "active_sum_abs_magmom_before_target": active_sum_before_target,
            "active_magmom_smooth_threshold": self.active_magmom_smooth_threshold,
            "active_magmom_smooth_applied_count": smooth_applied_count,
        }

    def check_state(self, atoms, tol=1e-15):
        # Only structural changes should invalidate the calculator cache.
        # Extra extxyz arrays such as target_magmom/initial_magmom are not part
        # of the dynamical state. Returning all_changes here forces ASE to
        # recalculate even for repeated get_energy/get_forces calls at the same
        # geometry; with magnetic rollout enabled that advances current_magmom
        # multiple times within one MD step and can create artificial feedback.
        if self.atoms is None:
            return all_changes
        changes = []
        if len(atoms) != len(self.atoms):
            return all_changes
        if not np.array_equal(atoms.numbers, self.atoms.numbers):
            changes.append("numbers")
        if not np.allclose(atoms.positions, self.atoms.positions, rtol=0.0, atol=tol):
            changes.append("positions")
        if not np.allclose(atoms.cell.array, self.atoms.cell.array, rtol=0.0, atol=tol):
            changes.append("cell")
        if not np.array_equal(atoms.pbc, self.atoms.pbc):
            changes.append("pbc")
        return changes

    def _atoms_to_batch(self, atoms):
        if self.current_magmom is None:
            init_mag = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
            self.current_magmom = torch.as_tensor(
                init_mag, dtype=torch.get_default_dtype()
            )
        atoms = atoms.copy()
        atoms.arrays["initial_magmom"] = (
            self.current_magmom.detach().cpu().numpy().astype(float)
        )
        config = mace_data.config_from_atoms(
            atoms,
            key_specification=mace_data.KeySpecification(
                info_keys={"head": "head"},
                arrays_keys={"initial_magmom": "initial_magmom"},
            ),
            head_name="Default",
        )
        data = mace_data.AtomicData.from_config(
            config,
            z_table=self.z_table,
            cutoff=self.cutoff,
            heads=["Default"],
        )
        if "initial_magmom" not in config.properties:
            raise KeyError("config_from_atoms did not preserve initial_magmom")
        init_magmom = np.asarray(config.properties["initial_magmom"], dtype=float).reshape(-1)
        if len(init_magmom) != len(atoms):
            raise ValueError(
                f"initial_magmom length {len(init_magmom)} != natoms {len(atoms)}"
            )
        data.initial_magmom = torch.as_tensor(
            init_magmom, dtype=torch.get_default_dtype()
        )
        loader = torch_geometric.dataloader.DataLoader([data], batch_size=1)
        batch = next(iter(loader)).to(self.device)
        if "initial_magmom" not in batch.to_dict():
            raise KeyError("AtomicData batch is missing initial_magmom")
        model_dtype = next(self.model.parameters()).dtype
        for key in batch.keys:
            value = batch[key]
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=model_dtype)
        return batch

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        batch = self._atoms_to_batch(atoms)
        out = self.model(
            batch.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=False,
        )
        next_magmom = out["magmom"].reshape(-1).detach().cpu().to(torch.float32)
        if "ce_active_prob" in out and out["ce_active_prob"] is not None:
            ce_active_prob = out["ce_active_prob"].reshape(-1).detach().cpu().to(torch.float32)
        elif "ce_active_logits" in out and out["ce_active_logits"] is not None:
            logits = out["ce_active_logits"].reshape(-1).detach().cpu().to(torch.float32)
            if self.ce_active_loss_type == "softmax_ce_soft_target":
                ce_active_prob = torch.zeros_like(logits)
                ce_indices = _get_ce_indices_from_atoms(atoms)
                if ce_indices:
                    ce_active_prob[ce_indices] = torch.softmax(logits[ce_indices], dim=0)
            else:
                ce_active_prob = torch.sigmoid(logits)
        else:
            ce_active_prob = None
        self.results["energy"] = float(out["energy"].reshape(-1)[0].detach().cpu())
        self.results["forces"] = out["forces"].detach().cpu().numpy()
        self.results["magmoms_pred"] = next_magmom.numpy()
        if ce_active_prob is not None:
            self.results["ce_active_prob"] = ce_active_prob.numpy()
            ce_indices = _get_ce_indices_from_atoms(atoms)
            ce_probs = ce_active_prob[ce_indices] if ce_indices else torch.tensor([])
            if len(ce_probs) > 0:
                top_order = torch.argsort(ce_probs, descending=True)
                top1_local = int(top_order[0].item())
                top2_local = int(top_order[1].item()) if len(top_order) > 1 else top1_local
                self.results["ce_active_probs_ce"] = ce_probs.numpy()
                self.results["ce_active_ce_indices_1based"] = [idx + 1 for idx in ce_indices]
                self.results["ce_active_prob_mean"] = float(ce_probs.mean().item())
                self.results["ce_active_prob_max"] = float(ce_probs.max().item())
                self.results["ce_active_top1_index_1based"] = ce_indices[top1_local] + 1
                self.results["ce_active_top2_index_1based"] = ce_indices[top2_local] + 1
                self.results["ce_active_count_prob_gt_0p5"] = int(
                    (ce_probs >= 0.5).sum().item()
                )
                self.results["ce_active_count_prob_gt_threshold"] = int(
                    (ce_probs >= self.ce_active_threshold).sum().item()
                )
                if len(top_order) > 1:
                    self.results["ce_active_prob_gap_top1_top2"] = (
                        float(ce_probs[top_order[0]].item())
                        - float(ce_probs[top_order[1]].item())
                    )
                else:
                    self.results["ce_active_prob_gap_top1_top2"] = None
        self.results["residual_total_energy"] = float(
            out["interaction_energy"].reshape(-1)[0].detach().cpu()
        )
        self.results["linref_total_energy"] = (
            self.results["energy"] - self.results["residual_total_energy"]
        )
        rollout_magmom = next_magmom
        if (
            self.rollout_magmom
            and self.constrain_active_ce
            and ce_active_prob is not None
        ):
            ce_indices = _get_ce_indices_from_atoms(atoms)
            if ce_indices:
                rollout_magmom, active_meta = self._project_rollout_magmom(
                    atoms,
                    next_magmom,
                    ce_active_prob[ce_indices],
                )
                self.results.update(active_meta)
        self.results["magmoms"] = rollout_magmom.numpy()
        self.pending_magmom = rollout_magmom
        if self.rollout_magmom:
            # The MD driver commits this once per accepted MD step. Updating
            # inside calculate() is unsafe because ASE may call energy/forces
            # several times at the same or intermediate geometry.
            pass

    def commit_magmom_rollout(self):
        if self.rollout_magmom and self.pending_magmom is not None:
            self.current_magmom = self.pending_magmom.detach().clone().to(torch.float32)


def build_dynamics(atoms, args):
    timestep = args.timestep_fs * units.fs
    if args.thermostat == "langevin":
        return Langevin(
            atoms,
            timestep,
            temperature_K=args.temperature_k,
            friction=args.friction,
        )
    if args.thermostat == "nose-hoover":
        if NoseHooverChainNVT is None:
            raise RuntimeError(
                "This ASE environment does not provide ase.md.nose_hoover_chain. "
                "Use --thermostat langevin or nve."
            )
        return NoseHooverChainNVT(
            atoms,
            timestep=timestep,
            temperature_K=args.temperature_k,
            tdamp=args.tdamp_fs * units.fs,
            tchain=args.tchain,
            tloop=args.tloop,
        )
    if args.thermostat == "nve":
        return VelocityVerlet(atoms, timestep=timestep)
    raise ValueError(f"Unsupported thermostat: {args.thermostat}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASE MD with MACE-FiLM model.")
    parser.add_argument("--structure", required=True)
    parser.add_argument("--structure-index", default="1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atomic-numbers", type=int, nargs="+", default=None)
    parser.add_argument("--temperature-k", type=float, default=50.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument(
        "--thermostat",
        choices=["langevin", "nose-hoover", "nve"],
        default="nose-hoover",
    )
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--tdamp-fs", type=float, default=100.0)
    parser.add_argument("--tchain", type=int, default=3)
    parser.add_argument("--tloop", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--loginterval", type=int, default=20)
    parser.add_argument(
        "--stop-temperature-k",
        type=float,
        default=1500.0,
        help="Abort MD if the instantaneous temperature exceeds this value.",
    )
    parser.add_argument("--velocity-seed", type=int, default=None)
    parser.add_argument("--rollout-magmom", action="store_true")
    parser.add_argument(
        "--constrain-active-ce",
        action="store_true",
        help="Use the Ce active head to project magnetic rollout onto active Ce sites.",
    )
    parser.add_argument(
        "--fixed-active-count",
        type=int,
        default=None,
        help="Fixed number of active Ce sites. Default infers from initial |magmom|.",
    )
    parser.add_argument(
        "--active-count-init-threshold",
        type=float,
        default=0.8,
        help="Initial |Ce magmom| threshold used to infer active count.",
    )
    parser.add_argument(
        "--ce-active-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used only for logging active Ce count.",
    )
    parser.add_argument(
        "--active-set-stickiness",
        type=float,
        default=0.0,
        help="Score bonus added to currently active Ce sites before top-k selection.",
    )
    parser.add_argument(
        "--active-set-confirm-steps",
        type=int,
        default=0,
        help="Require this many consecutive candidate steps before switching active set.",
    )
    parser.add_argument(
        "--active-magmom-mixing-alpha",
        type=float,
        default=0.7,
        help="Blend factor for optional active-site smoothing.",
    )
    parser.add_argument(
        "--active-magmom-smooth-threshold",
        type=float,
        default=0.0,
        help="If >0, smooth active Ce magmom when |pred-prev| exceeds this value.",
    )
    parser.add_argument(
        "--transfer-magmom-on-active-hop",
        action="store_true",
        help="On single-site active Ce hop, transfer old active |magmom| to new site.",
    )
    parser.add_argument(
        "--inactive-magmom-scale",
        type=float,
        default=0.05,
        help="Scale predicted magmom on inactive Ce sites during constrained rollout.",
    )
    parser.add_argument(
        "--active-magmom-target-mode",
        choices=["none", "init_sum_abs", "fixed"],
        default="none",
        help="Optionally constrain total |magmom| on active Ce set.",
    )
    parser.add_argument("--active-magmom-target-value", type=float, default=None)
    parser.add_argument(
        "--active-magmom-target-tolerance-rel",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--active-magmom-target-relax-power",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--init-magmom-mode",
        choices=["trajectory", "rule", "constant"],
        default="trajectory",
    )
    parser.add_argument("--init-magmom-spec", default=None)
    parser.add_argument("--output-dir", default="md_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = io.read(args.structure, index=args.structure_index)
    init_magmoms, init_mag_source = resolve_initial_magmoms_with_override(
        args.structure,
        args.structure_index,
        init_magmom_mode=args.init_magmom_mode,
        init_magmom_spec=args.init_magmom_spec,
    )
    print(f"Using initial magnetic moments from: {init_mag_source}", flush=True)
    calc = MACEFilmMDCalculator(
        checkpoint=args.checkpoint,
        device=args.device,
        init_magmom=init_magmoms,
        rollout_magmom=args.rollout_magmom,
        atomic_numbers=args.atomic_numbers,
        constrain_active_ce=args.constrain_active_ce,
        fixed_active_count=args.fixed_active_count,
        active_count_init_threshold=args.active_count_init_threshold,
        active_set_stickiness=args.active_set_stickiness,
        active_set_confirm_steps=args.active_set_confirm_steps,
        active_magmom_mixing_alpha=args.active_magmom_mixing_alpha,
        active_magmom_smooth_threshold=args.active_magmom_smooth_threshold
        if args.active_magmom_smooth_threshold > 0
        else None,
        transfer_magmom_on_active_hop=args.transfer_magmom_on_active_hop,
        inactive_magmom_scale=args.inactive_magmom_scale,
        active_magmom_target_mode=args.active_magmom_target_mode,
        active_magmom_target_value=args.active_magmom_target_value,
        active_magmom_target_tolerance_rel=args.active_magmom_target_tolerance_rel,
        active_magmom_target_relax_power=args.active_magmom_target_relax_power,
        ce_active_threshold=args.ce_active_threshold,
    )
    atoms.calc = calc

    velocity_rng = (
        np.random.default_rng(args.velocity_seed)
        if args.velocity_seed is not None
        else None
    )
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=args.temperature_k,
        rng=velocity_rng,
    )
    dyn = build_dynamics(atoms, args)

    summary_path = output_dir / "md_log.csv"
    magmom_path = output_dir / "md_magmom.jsonl"
    traj_path = output_dir / "md.traj"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "time_fs",
                "temperature_K",
                "potential_energy_eV",
                "residual_total_energy_eV",
                "linref_total_energy_eV",
                "kinetic_energy_eV",
                "total_energy_eV",
                "thermostat",
                "mean_magmom",
                "mean_abs_magmom",
                "max_abs_magmom",
                "ce_active_prob_mean",
                "ce_active_prob_max",
                "ce_active_top1_index_1based",
                "ce_active_top2_index_1based",
                "ce_active_count_prob_gt_0p5",
                "ce_active_count_prob_gt_threshold",
                "ce_active_prob_gap_top1_top2",
                "constrain_active_ce",
                "active_count_fixed",
                "active_set_indices_1based",
                "active_set_candidate_indices_1based",
                "active_set_previous_indices_1based",
                "active_set_changed",
                "active_set_pending",
                "active_set_transition",
                "active_set_rejected",
                "active_set_old_indices_1based",
                "active_set_new_indices_1based",
                "active_set_confirm_counter",
                "active_target_sum_abs_magmom",
                "active_sum_abs_magmom_pred",
                "active_sum_abs_magmom_input",
                "active_target_lower_bound",
                "active_target_upper_bound",
                "active_target_scale_applied",
                "active_target_correction_applied",
                "active_sum_abs_magmom_before_target",
                "active_magmom_smooth_applied_count",
            ]
        )
    magmom_path.write_text("", encoding="utf-8")
    traj = Trajectory(str(traj_path), "w", atoms)
    state = {"step": -args.loginterval}

    def log_status():
        state["step"] += args.loginterval
        if "energy" not in calc.results or "forces" not in calc.results:
            atoms.get_forces()
        temp = atoms.get_temperature()
        epot = float(calc.results["energy"])
        residual_epot = calc.results.get("residual_total_energy", epot)
        linref_epot = calc.results.get("linref_total_energy", 0.0)
        ekin = atoms.get_kinetic_energy()
        magmoms = [float(x) for x in calc.results.get("magmoms", [])]
        magmoms_pred = [float(x) for x in calc.results.get("magmoms_pred", [])]
        ce_active_prob = [float(x) for x in calc.results.get("ce_active_prob", [])]
        mean_mag = sum(magmoms) / len(magmoms)
        abs_mags = [abs(x) for x in magmoms]
        mean_abs_mag = sum(abs_mags) / len(abs_mags)
        max_abs_mag = max(abs_mags)
        ce_active_prob_mean = calc.results.get("ce_active_prob_mean")
        ce_active_prob_max = calc.results.get("ce_active_prob_max")
        ce_active_top1_index_1based = calc.results.get("ce_active_top1_index_1based")
        ce_active_top2_index_1based = calc.results.get("ce_active_top2_index_1based")
        ce_active_count_prob_gt_0p5 = calc.results.get("ce_active_count_prob_gt_0p5")
        ce_active_count_prob_gt_threshold = calc.results.get(
            "ce_active_count_prob_gt_threshold"
        )
        ce_active_prob_gap_top1_top2 = calc.results.get("ce_active_prob_gap_top1_top2")
        active_set_indices_1based = calc.results.get("active_set_indices_1based")
        active_set_candidate_indices_1based = calc.results.get(
            "active_set_candidate_indices_1based"
        )
        active_set_previous_indices_1based = calc.results.get(
            "active_set_previous_indices_1based"
        )
        active_set_changed = calc.results.get("active_set_changed")
        active_set_pending = calc.results.get("active_set_pending")
        active_set_transition = calc.results.get("active_set_transition")
        active_set_rejected = calc.results.get("active_set_rejected")
        active_set_old_indices_1based = calc.results.get("active_set_old_indices_1based")
        active_set_new_indices_1based = calc.results.get("active_set_new_indices_1based")
        active_set_confirm_counter = calc.results.get("active_set_confirm_counter")
        active_count_fixed = calc.results.get("active_count_fixed")
        active_target_sum_abs_magmom = calc.results.get("active_target_sum_abs_magmom")
        active_sum_abs_magmom_pred = calc.results.get("active_sum_abs_magmom_pred")
        active_sum_abs_magmom_input = calc.results.get("active_sum_abs_magmom_input")
        active_target_lower_bound = calc.results.get("active_target_lower_bound")
        active_target_upper_bound = calc.results.get("active_target_upper_bound")
        active_target_scale_applied = calc.results.get("active_target_scale_applied")
        active_target_correction_applied = calc.results.get(
            "active_target_correction_applied"
        )
        active_sum_abs_magmom_before_target = calc.results.get(
            "active_sum_abs_magmom_before_target"
        )
        active_magmom_smooth_applied_count = calc.results.get(
            "active_magmom_smooth_applied_count"
        )
        with summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    state["step"],
                    state["step"] * args.timestep_fs,
                    temp,
                    epot,
                    residual_epot,
                    linref_epot,
                    ekin,
                    epot + ekin,
                    args.thermostat,
                    mean_mag,
                    mean_abs_mag,
                    max_abs_mag,
                    ce_active_prob_mean,
                    ce_active_prob_max,
                    ce_active_top1_index_1based,
                    ce_active_top2_index_1based,
                    ce_active_count_prob_gt_0p5,
                    ce_active_count_prob_gt_threshold,
                    ce_active_prob_gap_top1_top2,
                    args.constrain_active_ce,
                    active_count_fixed,
                    json.dumps(_to_jsonable(active_set_indices_1based)),
                    json.dumps(_to_jsonable(active_set_candidate_indices_1based)),
                    json.dumps(_to_jsonable(active_set_previous_indices_1based)),
                    active_set_changed,
                    active_set_pending,
                    active_set_transition,
                    active_set_rejected,
                    json.dumps(_to_jsonable(active_set_old_indices_1based)),
                    json.dumps(_to_jsonable(active_set_new_indices_1based)),
                    active_set_confirm_counter,
                    active_target_sum_abs_magmom,
                    active_sum_abs_magmom_pred,
                    active_sum_abs_magmom_input,
                    active_target_lower_bound,
                    active_target_upper_bound,
                    active_target_scale_applied,
                    active_target_correction_applied,
                    active_sum_abs_magmom_before_target,
                    active_magmom_smooth_applied_count,
                ]
            )
        with magmom_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": state["step"],
                        "time_fs": state["step"] * args.timestep_fs,
                        "magmoms": magmoms,
                        "magmoms_pred": magmoms_pred,
                        "ce_active_prob": ce_active_prob,
                        "ce_active_probs_ce": calc.results.get("ce_active_probs_ce"),
                        "ce_indices_1based": calc.results.get(
                            "ce_active_ce_indices_1based"
                        ),
                        "ce_active_prob_mean": ce_active_prob_mean,
                        "ce_active_prob_max": ce_active_prob_max,
                        "ce_active_top1_index_1based": ce_active_top1_index_1based,
                        "ce_active_top2_index_1based": ce_active_top2_index_1based,
                        "ce_active_count_prob_gt_0p5": ce_active_count_prob_gt_0p5,
                        "ce_active_count_prob_gt_threshold": ce_active_count_prob_gt_threshold,
                        "ce_active_prob_gap_top1_top2": ce_active_prob_gap_top1_top2,
                        "constrain_active_ce": args.constrain_active_ce,
                        "active_count_fixed": active_count_fixed,
                        "active_set_indices_1based": active_set_indices_1based,
                        "active_set_candidate_indices_1based": active_set_candidate_indices_1based,
                        "active_set_previous_indices_1based": active_set_previous_indices_1based,
                        "active_set_changed": active_set_changed,
                        "active_set_pending": active_set_pending,
                        "active_set_transition": active_set_transition,
                        "active_set_rejected": active_set_rejected,
                        "active_set_old_indices_1based": active_set_old_indices_1based,
                        "active_set_new_indices_1based": active_set_new_indices_1based,
                        "active_set_confirm_counter": active_set_confirm_counter,
                        "active_target_sum_abs_magmom": active_target_sum_abs_magmom,
                        "active_sum_abs_magmom_pred": active_sum_abs_magmom_pred,
                        "active_sum_abs_magmom_input": active_sum_abs_magmom_input,
                        "active_target_lower_bound": active_target_lower_bound,
                        "active_target_upper_bound": active_target_upper_bound,
                        "active_target_scale_applied": active_target_scale_applied,
                        "active_target_correction_applied": active_target_correction_applied,
                        "active_sum_abs_magmom_before_target": active_sum_abs_magmom_before_target,
                        "active_magmom_smooth_applied_count": active_magmom_smooth_applied_count,
                    }
                , default=_to_jsonable)
                + "\n"
            )
        traj.write(atoms)
        print(
            f"step={state['step']} thermostat={args.thermostat} T={temp:.2f}K "
            f"Epot={epot:.6f}eV Ekin={ekin:.6f}eV <|m|>={mean_abs_mag:.6f}",
            flush=True,
        )
        finite_values = [temp, epot, residual_epot, linref_epot, ekin, mean_mag, mean_abs_mag, max_abs_mag]
        if not all(np.isfinite(value) for value in finite_values):
            raise RuntimeError(
                f"Stopping MD at step={state['step']}: non-finite value detected."
            )
        if temp > args.stop_temperature_k:
            raise RuntimeError(
                f"Stopping MD at step={state['step']}: temperature {temp:.2f} K "
                f"> {args.stop_temperature_k:.2f} K."
            )

    log_status()
    dyn.attach(calc.commit_magmom_rollout, interval=1)
    dyn.attach(log_status, interval=args.loginterval)
    dyn.run(args.steps)
    traj.close()


if __name__ == "__main__":
    main()
