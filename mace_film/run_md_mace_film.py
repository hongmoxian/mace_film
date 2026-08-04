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

from train_fen4_mace_magmom_film import _build_model

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


class MACEFilmMDCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint,
        device="cuda",
        init_magmom=None,
        rollout_magmom=True,
        atomic_numbers=None,
        enforce_spin_inversion=None,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location=self.device)
        args = SimpleNamespace(**payload["args"])
        if enforce_spin_inversion is not None:
            args.enforce_spin_inversion = bool(enforce_spin_inversion)
        z_values = atomic_numbers
        if z_values is None:
            z_values = getattr(args, "atomic_numbers", None)
        if z_values is None:
            z_values = getattr(args, "z_table", None)
        if z_values is None:
            z_values = [6, 7, 8, 26]
        self.z_table = AtomicNumberTable([int(z) for z in z_values])
        self.model = _build_model(args, self.z_table).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.cutoff = float(args.cutoff)
        self.rollout_magmom = bool(rollout_magmom)
        self.current_magmom = (
            None
            if init_magmom is None
            else torch.as_tensor(init_magmom, dtype=torch.get_default_dtype())
        )
        self.pending_magmom = None

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
        self.results["energy"] = float(out["energy"].reshape(-1)[0].detach().cpu())
        self.results["forces"] = out["forces"].detach().cpu().numpy()
        self.results["magmoms"] = next_magmom.numpy()
        self.results["residual_total_energy"] = float(
            out["interaction_energy"].reshape(-1)[0].detach().cpu()
        )
        self.results["linref_total_energy"] = (
            self.results["energy"] - self.results["residual_total_energy"]
        )
        self.pending_magmom = next_magmom
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
    parser.add_argument("--velocity-seed", type=int, default=None)
    parser.add_argument("--rollout-magmom", action="store_true")
    parser.add_argument(
        "--enforce-spin-inversion",
        dest="enforce_spin_inversion",
        action="store_true",
        default=None,
        help=(
            "Override the checkpoint and average the FiLM conditioning map for "
            "m and -m before message passing."
        ),
    )
    parser.add_argument(
        "--no-enforce-spin-inversion",
        dest="enforce_spin_inversion",
        action="store_false",
        help="Override the checkpoint and disable spin-inversion symmetrization.",
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
        enforce_spin_inversion=args.enforce_spin_inversion,
        atomic_numbers=args.atomic_numbers,
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
        mean_mag = sum(magmoms) / len(magmoms)
        abs_mags = [abs(x) for x in magmoms]
        mean_abs_mag = sum(abs_mags) / len(abs_mags)
        max_abs_mag = max(abs_mags)
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
                ]
            )
        with magmom_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": state["step"],
                        "time_fs": state["step"] * args.timestep_fs,
                        "magmoms": magmoms,
                    }
                )
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
        if temp > 1000.0:
            raise RuntimeError(
                f"Stopping MD at step={state['step']}: temperature {temp:.2f} K > 1000 K."
            )

    log_status()
    dyn.attach(calc.commit_magmom_rollout, interval=1)
    dyn.attach(log_status, interval=args.loginterval)
    dyn.run(args.steps)
    traj.close()


if __name__ == "__main__":
    main()
