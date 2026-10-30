#!/usr/bin/env python3
"""Au-Mn-O2 PES scan with the MACE-FiLM checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import io
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
from mace.tools.utils import AtomicNumberTable


def _working_modules():
    # The Mn and Fe directories use identical bare module names. Load the
    # mask-aware working implementation without depending on sys.path order.
    source = Path(__file__).resolve().parents[2]
    saved = {name: sys.modules.get(name) for name in (
        "mace_magmom_film", "train_fen4_mace_magmom_film"
    )}

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, source / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        network = load("_mn_pes_working_network", "mace_magmom_film.py")
        sys.modules["mace_magmom_film"] = network
        trainer = load("_mn_pes_working_trainer", "train_fen4_mace_magmom_film.py")
        sys.modules["train_fen4_mace_magmom_film"] = trainer
        driver = load("_mn_pes_working_driver", "run_md_mace_film.py")
        return trainer, driver
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


_trainer, _driver = _working_modules()
MACEFilmMDCalculator = _driver.MACEFilmMDCalculator


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    args: SimpleNamespace
    z_table: AtomicNumberTable
    device: torch.device
    metadata: dict


def load_checkpoint(checkpoint, device="cuda", atomic_numbers=None, atomic_reference=None):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; PES will not silently fall back to CPU.")
    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**payload["args"])
    state = payload["model_state_dict"]
    zs = [int(z) for z in state["atomic_numbers"].tolist()]
    if atomic_numbers is not None and list(atomic_numbers) != zs:
        raise ValueError(f"Atomic-number order {atomic_numbers} differs from checkpoint {zs}.")
    original_reference = getattr(args, "atomic_energy_linref", None)
    if atomic_reference is not None:
        reference = Path(atomic_reference).resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
        args.atomic_energy_linref = str(reference)
    elif original_reference is not None and not Path(original_reference).is_file():
        candidates = [parent / name for parent in (checkpoint.parent, checkpoint.parent.parent)
                      for name in (Path(original_reference).name, "atomic_energy_linref.json")]
        reference = next((p for p in candidates if p.is_file()), None)
        if reference is None:
            raise FileNotFoundError(
                f"Training reference {original_reference} is unavailable; pass --atomic-reference."
            )
        args.atomic_energy_linref = str(reference.resolve())
    table = AtomicNumberTable(zs)
    dtype = state["node_embedding.linear.weight"].dtype
    model = _trainer._build_model(args, table).to(dtype=dtype)
    for name, value in model.state_dict().items():
        if name.startswith("atomic_energies_fn.") or name == "magmom_scale":
            if not torch.equal(value.cpu(), state[name]):
                raise ValueError(f"Reference/normalization mismatch with checkpoint: {name}")
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value.cpu(), state[name]):
            raise ValueError(f"Checkpoint tensor changed during loading: {name}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "selected_epoch_1based": int(payload["epoch"]) + 1,
        "atomic_numbers": zs, "atomic_reference": getattr(args, "atomic_energy_linref", None),
        "magmom_condition_atomic_numbers": getattr(args, "magmom_condition_atomic_numbers", None),
        "magmom_scale": getattr(args, "magmom_scale", None),
        "magmom_input_clip": getattr(args, "magmom_input_clip", None),
        "film_scale": getattr(args, "film_scale", 1.0),
        "enforce_spin_inversion": getattr(args, "enforce_spin_inversion", False),
        "strict_state_dict_exact": True, "working_source": str(Path(_driver.__file__).resolve()),
        "torch_version": torch.__version__, "device": str(device),
    }
    if device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(device)
    return LoadedCheckpoint(model, args, table, device, metadata)


@dataclass
class ModelSpec:
    label: str
    template: str
    template_index: str
    state_id: int


def parse_distance_range(raw: str) -> list[float]:
    start, stop, step = [float(x) for x in raw.split(":")]
    if not all(math.isfinite(x) for x in (start, stop, step)) or step <= 0 or start > stop or start <= 0:
        raise ValueError("Scan range requires 0 < start <= stop and a positive finite step.")
    values = []
    current = start
    while current <= stop + 1e-8:
        values.append(round(current, 10))
        current += step
    return values


def parse_model_spec(raw: str) -> ModelSpec:
    parts = [x.strip() for x in raw.split("|")]
    if len(parts) != 4:
        raise ValueError("model spec must be LABEL|TEMPLATE|TEMPLATE_INDEX|STATE_ID")
    return ModelSpec(parts[0], parts[1], parts[2], int(parts[3]))


def safe_get_magmoms(atoms) -> np.ndarray:
    for key in ("target_magmom", "initial_magmom", "magmoms"):
        if key in atoms.arrays and len(atoms.arrays[key]) == len(atoms):
            values = np.asarray(atoms.arrays[key], dtype=float)
            if np.any(np.abs(values) > 1e-12):
                return values.copy()
    try:
        values = np.asarray(atoms.get_magnetic_moments(), dtype=float)
        if len(values) == len(atoms) and np.any(np.abs(values) > 1e-12):
            return values.copy()
    except Exception:
        pass
    values = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
    if len(values) != len(atoms) or not np.any(np.abs(values) > 1e-12):
        raise ValueError("Template does not contain usable initial magnetic moments.")
    return values.copy()


def expand_magmom_rule(rule: str, natoms: int) -> np.ndarray:
    values: list[float] = []
    for token in str(rule).split():
        token = token.strip()
        if not token:
            continue
        if "*" in token:
            count_raw, value_raw = token.split("*", 1)
            values.extend([float(value_raw)] * int(count_raw))
        else:
            values.append(float(token))
    if len(values) != int(natoms):
        raise ValueError(f"MAGMOM rule expands to {len(values)} atoms, expected {natoms}: {rule!r}")
    return np.asarray(values, dtype=float)


def load_magmom_overrides(path: str | None) -> dict[tuple[int, float], dict[int, float]]:
    if not path:
        return {}
    overrides: dict[tuple[int, float], dict[int, float]] = {}
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        required = {"state_id", "distance", "atom_index", "magmom"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing magmom override columns in {path}: {sorted(missing)}")
        for row in reader:
            key = (int(row["state_id"]), round(float(row["distance"]), 10))
            overrides.setdefault(key, {})[int(row["atom_index"])] = float(row["magmom"])
    return overrides


def apply_magmom_override(
    magmoms: np.ndarray,
    overrides: dict[tuple[int, float], dict[int, float]],
    state_id: int,
    distance: float,
) -> tuple[np.ndarray, str]:
    patch = overrides.get((int(state_id), round(float(distance), 10)))
    if not patch:
        return magmoms, ""
    updated = np.asarray(magmoms, dtype=float).copy()
    for atom_index, value in patch.items():
        if atom_index < 0 or atom_index >= len(updated):
            raise IndexError(f"Magmom override atom index out of range: {atom_index} for natoms={len(updated)}")
        updated[int(atom_index)] = float(value)
    detail = ",".join(f"{idx}:{value:g}" for idx, value in sorted(patch.items()))
    return updated, f"override[state={state_id},distance={distance:.1f},{detail}]"


def format_scan_distance(distance: float) -> str:
    return f"{float(distance):.1f}"


def resolve_scan_frame_path(spec: ModelSpec, distance: float, pattern: str) -> Path:
    template_path = Path(spec.template)
    # Current templates are e.g. .../afm-o/md_optlike/OUTCAR and .../fm-o/md/OUTCAR.
    # The corresponding DFT scan directories live one level above md/md_optlike.
    scan_root = template_path.parent.parent
    distance_text = format_scan_distance(distance)
    return scan_root / pattern.format(distance=distance_text, distance_float=float(distance))


def get_scan_init_magmoms(
    atoms,
    spec: ModelSpec,
    mode: str,
    afm_rule: str,
    fm_rule: str,
    distance: float | None = None,
    scan_frame_pattern: str = "scan_mno_{distance}A/OUTCAR",
    scan_frame_index: str = "0",
) -> tuple[np.ndarray, str]:
    if mode == "template":
        template_atoms = io.read(spec.template, index=spec.template_index)
        return safe_get_magmoms(template_atoms), f"template:{spec.template}@{spec.template_index}"
    if mode == "rule":
        rule = afm_rule if int(spec.state_id) == 0 else fm_rule
        return expand_magmom_rule(rule, len(atoms)), f"rule:{rule}"
    if mode == "scan_frame":
        if distance is None:
            raise ValueError("scan_frame init mode requires a scan distance.")
        scan_path = resolve_scan_frame_path(spec, distance, scan_frame_pattern)
        if not scan_path.exists():
            raise FileNotFoundError(
                f"Missing scan-frame magnetic initialization source: {scan_path}"
            )
        scan_atoms = io.read(scan_path, index=scan_frame_index)
        magmoms = safe_get_magmoms(scan_atoms)
        if not np.array_equal(scan_atoms.numbers, atoms.numbers):
            raise ValueError(
                f"scan_frame atom count/order mismatch for {scan_path}"
            )
        return magmoms, f"scan_frame:{scan_path}@{scan_frame_index}"
    raise ValueError(f"Unsupported init magmom mode: {mode!r}")


def retarget_o2_distance(atoms, mn_index: int, o_indices: Sequence[int], target_distance: float):
    atoms = atoms.copy()
    o1, o2 = [int(x) for x in o_indices]
    distances = atoms.get_distances(mn_index, [o1, o2], mic=True)
    anchor = o1 if distances[0] <= distances[1] else o2
    direction = atoms.get_distance(mn_index, anchor, mic=True, vector=True)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("Degenerate Mn-anchor O distance in template.")
    delta = (float(target_distance) - norm) * direction / norm
    atoms.positions[[o1, o2]] += delta
    return atoms, anchor


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def structure_metrics(atoms, mn_index: int, o_indices: Sequence[int], anchor_o_index: int):
    o1, o2 = [int(x) for x in o_indices]
    mn_o1 = float(atoms.get_distance(mn_index, o1, mic=True))
    mn_o2 = float(atoms.get_distance(mn_index, o2, mic=True))
    anchor = float(atoms.get_distance(mn_index, anchor_o_index, mic=True))
    oo = float(atoms.get_distance(o1, o2, mic=True))
    other = o2 if anchor_o_index == o1 else o1
    angle = _angle_between(
        atoms.get_distance(anchor_o_index, mn_index, mic=True, vector=True),
        atoms.get_distance(anchor_o_index, other, mic=True, vector=True),
    )
    centroid = 0.5 * (
        atoms.get_distances(mn_index, [o1], mic=True, vector=True)[0]
        + atoms.get_distances(mn_index, [o2], mic=True, vector=True)[0]
    )
    return {
        "mn_o1_distance_A": mn_o1,
        "mn_o2_distance_A": mn_o2,
        "mn_anchor_o_distance_A": anchor,
        "mn_o_min_distance_A": min(mn_o1, mn_o2),
        "mn_o_mean_distance_A": 0.5 * (mn_o1 + mn_o2),
        "mn_o2_centroid_distance_A": float(np.linalg.norm(centroid)),
        "o_o_distance_A": oo,
        "mn_anchor_o_other_o_angle_deg": angle,
    }


def harmonic_distance_restraint(atoms, i: int, j: int, target_distance: float, k: float):
    vector = torch.tensor(atoms.get_distance(i, j, mic=True, vector=True),
                          dtype=torch.float64, requires_grad=True)
    distance = torch.linalg.norm(vector)
    energy = 0.5 * float(k) * (distance - float(target_distance)) ** 2
    grad = torch.autograd.grad(energy, vector)[0].detach().numpy()
    forces = np.zeros((len(atoms), 3))
    forces[i] += grad
    forces[j] -= grad
    return float(energy.item()), forces


def harmonic_angle_restraint(atoms, mn_index: int, anchor_o_index: int, other_o_index: int, target_angle_deg: float, k: float):
    vectors = torch.tensor(atoms.get_distances(
        anchor_o_index, [mn_index, other_o_index], mic=True, vector=True),
        dtype=torch.float64, requires_grad=True)
    v1, v2 = vectors
    norms = torch.linalg.norm(v1) * torch.linalg.norm(v2)
    if float(norms.detach()) < 1e-12:
        raise ValueError("Degenerate bond in angle restraint.")
    cosang = torch.dot(v1, v2) / norms
    angle = torch.acos(torch.clamp(cosang, -0.999999, 0.999999))
    target = torch.tensor(math.radians(float(target_angle_deg)), dtype=vectors.dtype)
    energy = 0.5 * float(k) * (angle - target) ** 2
    grad = torch.autograd.grad(energy, vectors)[0].detach().numpy()
    forces = np.zeros((len(atoms), 3))
    forces[mn_index] -= grad[0]
    forces[other_o_index] -= grad[1]
    forces[anchor_o_index] += grad.sum(axis=0)
    return float(energy.item()), forces


class MACEFilmMnO2ScanCalculator(MACEFilmMDCalculator):
    def __init__(
        self,
        checkpoint=None,
        device="cuda",
        init_magmom=None,
        rollout_magmom=True,
        atomic_numbers=None,
        atomic_reference=None,
        loaded_checkpoint=None,
        *,
        mn_index,
        anchor_o_index,
        other_o_index,
        target_distance,
        distance_k,
        target_angle_deg=None,
        angle_k=0.0,
    ):
        Calculator.__init__(self)
        loaded = loaded_checkpoint or load_checkpoint(
            checkpoint, device, atomic_numbers, atomic_reference
        )
        self.model, self.device, self.z_table = loaded.model, loaded.device, loaded.z_table
        self.cutoff = float(loaded.args.cutoff)
        self.rollout_magmom = bool(rollout_magmom)
        condition = getattr(loaded.args, "magmom_condition_atomic_numbers", None)
        self.magmom_condition_atomic_numbers = None if condition is None else frozenset(condition)
        if init_magmom is None or np.ndim(init_magmom) != 1 or not np.isfinite(init_magmom).all():
            raise ValueError("Explicit finite initial magnetic moments are required.")
        self.current_magmom = torch.as_tensor(init_magmom, dtype=torch.float32).clone()
        self.pending_magmom = None
        self.metadata = loaded.metadata
        self.mn_index = int(mn_index)
        self.anchor_o_index = int(anchor_o_index)
        self.other_o_index = int(other_o_index)
        self.target_distance = float(target_distance)
        self.distance_k = float(distance_k)
        self.target_angle_deg = None if target_angle_deg is None else float(target_angle_deg)
        self.angle_k = float(angle_k)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e_dist, f_dist = harmonic_distance_restraint(
            atoms,
            self.mn_index,
            self.anchor_o_index,
            self.target_distance,
            self.distance_k,
        )
        e_angle = 0.0
        f_angle = np.zeros_like(f_dist)
        if self.target_angle_deg is not None and self.angle_k > 0:
            e_angle, f_angle = harmonic_angle_restraint(
                atoms,
                self.mn_index,
                self.anchor_o_index,
                self.other_o_index,
                self.target_angle_deg,
                self.angle_k,
            )
        self.results["model_energy"] = self.results["energy"]
        self.results["model_forces"] = self.results["forces"].copy()
        self.results["energy"] += e_dist + e_angle
        self.results["forces"] = self.results["forces"] + f_dist + f_angle
        self.results["distance_restraint_energy"] = e_dist
        self.results["angle_restraint_energy"] = e_angle


class AcceptedStepMagmomUpdate:
    """One explicit magnetic commit per accepted BFGS step, never at step zero."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_step = 0
        self.updates = 0
        self.cumulative_switch_energy = 0.0
        self.records = []

    def __call__(self):
        step = self.optimizer.nsteps
        atoms = self.optimizer.atoms
        calc = atoms.calc
        if step <= self.last_step:
            return
        # ASE evaluates the new geometry before observers. Ensure pending
        # moments belong to this geometry even when called without that log.
        atoms.get_potential_energy()
        before = float(calc.results["model_energy"])
        old_input = calc.current_magmom.clone()
        committed = calc.commit_magmom_rollout()
        atoms.get_potential_energy()
        delta = float(calc.results["model_energy"]) - before
        if committed:
            self.updates += 1
            self.cumulative_switch_energy += delta
        self.records.append({
            "step": step, "magmom_committed": int(committed),
            "switch_energy_eV": delta, "cumulative_switch_energy_eV": self.cumulative_switch_energy,
            "input_magmom_before_muB": old_input.tolist(),
            "input_magmom_after_muB": calc.current_magmom.tolist(),
            "predicted_magmom_muB": calc.results["magmoms"].tolist(),
            "model_energy_eV": float(calc.results["model_energy"]),
            "optimizer_energy_eV": float(calc.results["energy"]),
        })
        self.last_step = step


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label) or "model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["mhc", "mtc"], default="mtc")
    parser.add_argument("--atomic-reference", "--atomic-energy-linref", dest="atomic_reference", help="Portable atomic reference JSON; verified against checkpoint.")
    parser.add_argument("--model-spec", action="append", required=True)
    parser.add_argument("--mn-index", type=int, default=15)
    parser.add_argument("--o-indices", type=int, nargs=2, default=[16, 17])
    parser.add_argument("--atomic-numbers", type=int, nargs="+", default=None,
                        help="Optional validation of checkpoint species order (otherwise inferred).")
    parser.add_argument("--scan-distance-range", default="1.8:4.0:0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distance-k", type=float, default=50.0)
    parser.add_argument("--use-angle-restraint", action="store_true")
    parser.add_argument("--angle-k", type=float, default=20.0)
    parser.add_argument(
        "--init-magmom-mode",
        choices=["template", "rule", "scan_frame"],
        default=None,
        help=(
            "Magnetic initialization for the ML PES scan. "
            "'template' reuses the branch template magmom for all distances; "
            "'rule' uses the AFM/FM MAGMOM rules; "
            "'scan_frame' reads the corresponding DFT scan OUTCAR for each distance."
        ),
    )
    parser.add_argument(
        "--afm-init-magmom-mode",
        choices=["template", "rule", "scan_frame"],
        default=None,
        help="Optional AFM-only override for --init-magmom-mode.",
    )
    parser.add_argument(
        "--fm-init-magmom-mode",
        choices=["template", "rule", "scan_frame"],
        default=None,
        help="Optional FM-only override for --init-magmom-mode.",
    )
    parser.add_argument("--afm-rule", default="15*0 1*-4 2*1")
    parser.add_argument("--fm-rule", default="15*0 1*4 2*1")
    parser.add_argument(
        "--scan-frame-pattern",
        default="scan_mno_{distance}A/OUTCAR",
        help=(
            "Relative path below the branch root used by --init-magmom-mode scan_frame. "
            "Available fields: {distance} and {distance_float}."
        ),
    )
    parser.add_argument(
        "--scan-frame-index",
        default="0",
        help="Frame index read from each DFT scan OUTCAR for scan_frame initialization.",
    )
    parser.add_argument(
        "--magmom-override-csv",
        default=None,
        help=(
            "Optional CSV with columns state_id,distance,atom_index,magmom. "
            "Applied after template/rule/scan_frame initialization."
        ),
    )
    parser.add_argument("--opt-fmax", type=float, default=0.05)
    parser.add_argument("--opt-steps", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-plot", required=True)
    parser.add_argument("--output-structures-dir", required=True)
    args = parser.parse_args()

    if args.opt_steps < 0 or args.opt_fmax <= 0 or args.distance_k < 0 or args.angle_k < 0:
        parser.error("Require opt-steps >= 0, opt-fmax > 0, and nonnegative restraint coefficients.")
    if args.mode == "mtc":
        # Both a fixed branch template and an explicit fixed MAGMOM rule
        # are valid MTC initializations; only per-scan-frame DFT seeds and
        # distance-specific overrides violate the constant-template protocol.
        if any(mode not in (None, "rule", "template") for mode in (
            args.init_magmom_mode, args.afm_init_magmom_mode, args.fm_init_magmom_mode
        )) or args.magmom_override_csv:
            parser.error("MTC requires a fixed branch template or rule; per-frame initialization and overrides are not allowed.")
    args.init_magmom_mode = args.init_magmom_mode or ("rule" if args.mode == "mtc" else "template")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    distances = parse_distance_range(args.scan_distance_range)
    specs = [parse_model_spec(raw) for raw in args.model_spec]
    if len({sanitize_label(spec.label) for spec in specs}) != len(specs):
        parser.error("Model labels must be unique after filename sanitization.")
    if any(spec.state_id not in (0, 1) for spec in specs):
        parser.error("State IDs must be 0 (AFM) or 1 (FM).")
    loaded = load_checkpoint(args.checkpoint, args.device, args.atomic_numbers, args.atomic_reference)
    magmom_overrides = load_magmom_overrides(args.magmom_override_csv)
    out_struct_dir = Path(args.output_structures_dir)
    out_struct_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(loaded.metadata, mode=args.mode, seed=args.seed,
                    command=sys.argv, parameters=vars(args),
                    energy_convention="Relative model energy excludes harmonic restraints; optimizer energy includes them.",
                    magnetic_update="One commit after each accepted optimizer step; none at step 0; MTC fixed.")
    manifest_path = Path(args.output_csv).with_suffix(".metadata.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    energy_by_label = {spec.label: {} for spec in specs}
    total_jobs = len(specs) * len(distances)
    job = 0

    for spec in specs:
        template = io.read(spec.template, index=spec.template_index)
        if (len(set([args.mn_index, *args.o_indices])) != 3
                or any(i < 0 or i >= len(template) for i in [args.mn_index, *args.o_indices])
                or template.numbers[args.mn_index] != 25
                or any(template.numbers[i] != 8 for i in args.o_indices)):
            raise ValueError("Mn/O indices do not match the template species.")
        for distance in distances:
            job += 1
            atoms, anchor = retarget_o2_distance(
                template,
                mn_index=args.mn_index,
                o_indices=args.o_indices,
                target_distance=distance,
            )
            state_init_mode = args.init_magmom_mode
            if int(spec.state_id) == 0 and args.afm_init_magmom_mode is not None:
                state_init_mode = args.afm_init_magmom_mode
            if int(spec.state_id) == 1 and args.fm_init_magmom_mode is not None:
                state_init_mode = args.fm_init_magmom_mode

            init_magmoms, init_magmom_source = get_scan_init_magmoms(
                atoms,
                spec,
                mode=state_init_mode,
                afm_rule=args.afm_rule,
                fm_rule=args.fm_rule,
                distance=distance,
                scan_frame_pattern=args.scan_frame_pattern,
                scan_frame_index=args.scan_frame_index,
            )
            init_magmoms, override_source = apply_magmom_override(
                init_magmoms,
                magmom_overrides,
                state_id=spec.state_id,
                distance=distance,
            )
            if override_source:
                init_magmom_source = f"{init_magmom_source};{override_source}"
            other = int(args.o_indices[1]) if anchor == int(args.o_indices[0]) else int(args.o_indices[0])
            initial_metrics = structure_metrics(atoms, args.mn_index, args.o_indices, anchor)
            target_angle = (
                float(initial_metrics["mn_anchor_o_other_o_angle_deg"])
                if args.use_angle_restraint
                else None
            )
            calc = MACEFilmMnO2ScanCalculator(
                loaded_checkpoint=loaded,
                init_magmom=init_magmoms,
                rollout_magmom=args.mode == "mhc",
                mn_index=args.mn_index,
                anchor_o_index=anchor,
                other_o_index=other,
                target_distance=distance,
                distance_k=args.distance_k,
                target_angle_deg=target_angle,
                angle_k=args.angle_k,
            )
            atoms.calc = calc
            optimizer = BFGS(atoms, logfile=None)
            update_magmoms = AcceptedStepMagmomUpdate(optimizer)
            optimizer.attach(update_magmoms, interval=1)
            label = f"[{job}/{total_jobs}] model={spec.label} scan_distance={distance:.2f}A"
            print(f"{label} start", flush=True)
            def log_progress():
                step = optimizer.nsteps
                if step == 0 or step % max(args.progress_every, 1) == 0:
                    e = float(atoms.get_potential_energy())
                    fmax = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max()))
                    r = float(atoms.get_distance(args.mn_index, anchor, mic=True))
                    print(
                        f"{label} opt_step={step} energy={e:.6f} eV "
                        f"r(Mn-O_anchor)={r:.4f} A dr={r - distance:+.4f} A fmax={fmax:.4f}",
                        flush=True,
                    )

            optimizer.attach(log_progress, interval=1)
            converged = optimizer.run(fmax=args.opt_fmax, steps=args.opt_steps)
            total_energy = float(atoms.get_potential_energy())
            model_energy = float(calc.results["model_energy"])
            final_forces = atoms.get_forces()
            magmoms = np.asarray(calc.results.get("magmoms"), dtype=float)
            input_magmoms = np.asarray(calc.results["input_magmom"], dtype=float)
            selected = (np.ones(len(atoms), dtype=bool) if calc.magmom_condition_atomic_numbers is None
                        else np.isin(atoms.numbers, list(calc.magmom_condition_atomic_numbers)))
            if not selected.any():
                raise ValueError("No atoms selected for magnetic conditioning.")
            metrics = structure_metrics(atoms, args.mn_index, args.o_indices, anchor)
            path = out_struct_dir / f"{sanitize_label(spec.label)}_d{distance:.2f}_constrained_opt.vasp"
            io.write(path, atoms, format="vasp")
            trace_path = path.with_suffix(".magnetic_trace.json")
            trace_path.write_text(json.dumps(update_magmoms.records, indent=2) + "\n")
            energy_by_label[spec.label][float(distance)] = model_energy
            row = {
                "mode": "constrained_opt",
                "magnetic_mode": args.mode,
                "scan_init_mode": "independent",
                "constraint_mode": "harmonic_distance_plus_angle" if args.use_angle_restraint else "harmonic_distance",
                "distance_k_eV_per_A2": float(args.distance_k),
                "use_angle_restraint": int(bool(args.use_angle_restraint)),
                "angle_k_eV_per_rad2": float(args.angle_k) if args.use_angle_restraint else "",
                "scan_displace_mode": "o2_rigid",
                "init_magmom_mode": state_init_mode,
                "init_magmom_source": init_magmom_source,
                "model_label": spec.label,
                "state_id": int(spec.state_id),
                "scan_distance": float(distance),
                "template": spec.template,
                "template_index": spec.template_index,
                "saved_structure_path": str(path),
                "magnetic_trace_path": str(trace_path),
                "checkpoint_sha256": loaded.metadata["checkpoint_sha256"],
                "converged": int(bool(converged)),
                "optimizer_steps": optimizer.nsteps,
                "magnetic_updates": update_magmoms.updates,
                "cumulative_switch_energy_eV": update_magmoms.cumulative_switch_energy,
                "final_fmax_eV_per_A": float(np.linalg.norm(final_forces, axis=1).max()),
                "magnetic_residual_selected_rms_muB": float(np.sqrt(np.mean((magmoms[selected] - input_magmoms[selected]) ** 2))),
                "model_energy_eV": model_energy,
                "optimizer_energy_eV": total_energy,
                "total_energy_eV": total_energy,
                "residual_total_energy_eV": float(calc.results.get("residual_total_energy", np.nan)),
                "linref_total_energy_eV": float(calc.results.get("linref_total_energy", np.nan)),
                "distance_restraint_energy_eV": float(calc.results.get("distance_restraint_energy", np.nan)),
                "angle_restraint_energy_eV": float(calc.results.get("angle_restraint_energy", np.nan)),
                "total_magmom": float(np.sum(magmoms)),
                "max_abs_magmom": float(np.max(np.abs(magmoms))),
                "mn_magmom": float(magmoms[args.mn_index]),
                "o1_magmom": float(magmoms[int(args.o_indices[0])]),
                "o2_magmom": float(magmoms[int(args.o_indices[1])]),
                "o_sum_magmom": float(magmoms[args.o_indices].sum()),
                "input_mn_magmom": float(input_magmoms[args.mn_index]),
                "input_o1_magmom": float(input_magmoms[int(args.o_indices[0])]),
                "input_o2_magmom": float(input_magmoms[int(args.o_indices[1])]),
                "init_total_magmom": float(np.sum(init_magmoms)),
                "init_mn_magmom": float(init_magmoms[int(args.mn_index)]),
                "init_o1_magmom": float(init_magmoms[int(args.o_indices[0])]),
                "init_o2_magmom": float(init_magmoms[int(args.o_indices[1])]),
                "distance_error_A": metrics["mn_anchor_o_distance_A"] - distance,
                **metrics,
                **{f"initial_{k}": v for k, v in initial_metrics.items()},
            }
            rows.append(row)
            print(f"{label} done converged={bool(converged)} model_energy={model_energy:.6f} eV "
                  f"optimizer_energy={total_energy:.6f} eV updates={update_magmoms.updates} saved={path}", flush=True)

    for spec in specs:
        reference_distance = max(energy_by_label[spec.label])
        reference_energy = energy_by_label[spec.label][reference_distance]
        for row in rows:
            if row["model_label"] == spec.label:
                row["reference_distance"] = reference_distance
                row["reference_model_energy_eV"] = reference_energy
                reference_row = next(r for r in rows if r["model_label"] == spec.label
                                     and r["scan_distance"] == reference_distance)
                row["reference_total_energy_eV"] = reference_row["total_energy_eV"]
                row["optimizer_energy_relative_eV"] = row["total_energy_eV"] - reference_row["total_energy_eV"]
                row["adsorption_energy_relative_eV"] = row["model_energy_eV"] - reference_energy
                row["model_energy_relative_eV"] = row["adsorption_energy_relative_eV"]

    rows.sort(key=lambda x: (str(x["model_label"]), float(x["scan_distance"])))
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_plot = Path(args.output_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    for spec in specs:
        model_rows = [row for row in rows if row["model_label"] == spec.label]
        xs = [float(row["mn_anchor_o_distance_A"]) for row in model_rows]
        ys = [float(row["adsorption_energy_relative_eV"]) for row in model_rows]
        plt.plot(xs, ys, marker="o", label=spec.label)
    plt.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    plt.xlabel("Actual Mn-anchor O distance (A)")
    plt.ylabel("Relative adsorption energy (eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Wrote plot to {out_plot}")


if __name__ == "__main__":
    main()
