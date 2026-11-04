#!/usr/bin/env python3
"""FeN4-O2 PES scan with the MACE-FiLM checkpoint.

This mirrors the GemOC scan settings used previously: independent distance
points, rigid O2 displacement, harmonic Fe-anchor-O restraint, and BFGS
constrained optimization.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import io
from ase.calculators.calculator import all_changes
from ase.optimize import BFGS

from run_md_mace_film import MACEFilmMDCalculator


@dataclass
class ModelSpec:
    label: str
    template: str
    template_index: str
    state_id: int


def parse_distance_range(raw: str) -> list[float]:
    start, stop, step = [float(x) for x in raw.split(":")]
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
            vals = np.asarray(atoms.arrays[key], dtype=float)
            if np.any(np.abs(vals) > 1e-12):
                return vals.copy()
    try:
        vals = np.asarray(atoms.get_magnetic_moments(), dtype=float)
        if len(vals) == len(atoms) and np.any(np.abs(vals) > 1e-12):
            return vals.copy()
    except Exception:
        pass
    vals = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
    if len(vals) != len(atoms) or not np.any(np.abs(vals) > 1e-12):
        raise ValueError("Template does not contain usable initial magnetic moments.")
    return vals.copy()


def expand_magmom_rule(rule: str, natoms: int) -> np.ndarray:
    values: list[float] = []
    for raw_part in str(rule).split():
        part = raw_part.strip()
        if not part:
            continue
        if "*" in part:
            count_raw, value_raw = part.split("*", 1)
            values.extend([float(value_raw)] * int(count_raw))
        else:
            values.append(float(part))
    if len(values) != int(natoms):
        raise ValueError(f"Magmom rule expands to {len(values)} atoms, expected {natoms}: {rule!r}")
    return np.asarray(values, dtype=float)


def format_scan_distance(distance: float) -> str:
    return f"{float(distance):.2f}"


def resolve_scan_frame_path(spec: ModelSpec, distance: float, pattern: str) -> Path:
    branch = "S0" if int(spec.state_id) == 0 else "S2"
    distance_text = format_scan_distance(distance)
    distance_short = f"{float(distance):.1f}"
    return Path(pattern.format(
        branch=branch,
        state=branch,
        distance=distance_text,
        distance_short=distance_short,
        distance_float=float(distance),
    ))


def resolve_scan_structure_path(spec: ModelSpec, distance: float, pattern: str) -> Path:
    branch = "S0" if int(spec.state_id) == 0 else "S2"
    distance_text = format_scan_distance(distance)
    distance_short = f"{float(distance):.1f}"
    return Path(pattern.format(
        branch=branch,
        state=branch,
        distance=distance_text,
        distance_short=distance_short,
        distance_float=float(distance),
    ))


def read_outcar_magnetization_block(path: Path, block_index: int, natoms: int) -> np.ndarray:
    blocks: list[np.ndarray] = []
    lines = path.read_text(errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        if "magnetization (x)" not in lines[i]:
            i += 1
            continue
        i += 1
        values: list[float] = []
        while i < len(lines):
            parts = lines[i].split()
            if len(parts) >= 5 and parts[0].isdigit():
                try:
                    values.append(float(parts[4]))
                except ValueError:
                    pass
            elif values and (parts[:1] == ["tot"] or "--------" in lines[i]):
                if len(values) == natoms:
                    blocks.append(np.asarray(values, dtype=float))
                break
            i += 1
    if not blocks:
        raise ValueError(f"No complete magnetization blocks found in {path}")
    idx = int(block_index)
    if idx < 0:
        idx = len(blocks) + idx
    if idx < 0 or idx >= len(blocks):
        raise IndexError(
            f"OUTCAR magnetization block index {block_index} out of range for {path}; "
            f"found {len(blocks)} complete blocks"
        )
    return blocks[idx].copy()


def _strip_vasp_comment(line: str) -> str:
    return re.split(r"[!#]", line, maxsplit=1)[0].strip()


def _expand_magmom_rule(rule: str, natoms: int) -> np.ndarray:
    values: list[float] = []
    for token in str(rule).replace("=", " ").split():
        token = token.strip()
        if not token:
            continue
        if "*" in token:
            count_raw, value_raw = token.split("*", 1)
            values.extend([float(value_raw)] * int(float(count_raw)))
        else:
            values.append(float(token))
    if len(values) != int(natoms):
        raise ValueError(f"MAGMOM expands to {len(values)} atoms, expected {natoms}: {rule!r}")
    return np.asarray(values, dtype=float)


def read_vasp_input_magmom(path: Path, natoms: int) -> tuple[np.ndarray, str]:
    """Read the input MAGMOM guess from INCAR or an OUTCAR nearby."""
    path = Path(path)
    search_dirs: list[Path] = []
    if path.is_dir():
        search_dirs.append(path)
    else:
        search_dirs.append(path.parent)
    if path.name == "OUTCAR":
        search_dirs.append(path.parent)

    for directory in search_dirs:
        for name in ("INCAR", "OUTCAR"):
            incar = directory / name
            if not incar.exists():
                continue
            lines = incar.read_text(encoding="utf-8", errors="ignore").splitlines()
            for idx, raw in enumerate(lines):
                line = _strip_vasp_comment(raw)
                if not re.match(r"^\s*MAGMOM\s*=", line, flags=re.IGNORECASE):
                    continue
                rhs = line.split("=", 1)[1].strip()
                for extra in lines[idx + 1 : idx + 8]:
                    candidate = _strip_vasp_comment(extra)
                    if not candidate:
                        continue
                    if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_ -]*\s*=", candidate):
                        break
                    rhs = f"{rhs} {candidate}"
                    try:
                        expanded = _expand_magmom_rule(rhs, natoms)
                    except ValueError:
                        continue
                    return expanded, f"vasp_input:{incar}"
                return _expand_magmom_rule(rhs, natoms), f"vasp_input:{incar}"
    raise FileNotFoundError(f"Could not find VASP MAGMOM near {path}")


@lru_cache(maxsize=8)
def _load_atoms_frames(path_str: str) -> tuple:
    path = Path(path_str)
    frames = io.read(path, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    return tuple(frames)


def read_opt_second_frame_magmom(
    path: Path,
    atoms,
    state_id: int,
    distance: float,
    missing_policy: str = "error",
) -> tuple[np.ndarray, str]:
    """Read magmoms from the source_step=2 frame in a merged opt extxyz."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing opt extxyz magnetic initialization source: {path}")

    target_distance = round(float(distance), 2)
    target_state = int(state_id)
    candidates = []
    fallback_candidates = []
    for frame in _load_atoms_frames(str(path)):
        frame_state = frame.info.get("state_id", frame.info.get("state", None))
        frame_distance = frame.info.get("scan_distance", frame.info.get("distance", None))
        frame_step = frame.info.get("source_step", None)
        if frame_state is None or frame_distance is None or frame_step is None:
            continue
        try:
            if int(frame_state) != target_state:
                continue
            if int(frame_step) != 2:
                continue
            if round(float(frame_distance), 2) != target_distance:
                fallback_candidates.append((abs(round(float(frame_distance), 2) - target_distance), frame))
                continue
        except Exception:
            continue
        candidates.append(frame)

    if not candidates:
        if missing_policy != "fallback_nearest" or not fallback_candidates:
            raise FileNotFoundError(
                f"Could not find source_step=2 frame in {path} for state_id={target_state} distance={target_distance:.2f}"
            )
        fallback_candidates.sort(key=lambda item: (item[0], float(item[1].info["scan_distance"])))
        chosen = fallback_candidates[0][1]
        candidates = [chosen]
        source_distance = round(float(chosen.info["scan_distance"]), 2)
        source_suffix = f":fallback_nearest_d{source_distance:.2f}"
    else:
        source_suffix = ""
    if len(candidates) > 1:
        # Keep the most recent match if duplicated; the dataset should normally be unique.
        candidates = [candidates[-1]]

    magmoms = safe_get_magmoms(candidates[0])
    if len(magmoms) != len(atoms):
        raise ValueError(
            f"opt_second_frame magmom length mismatch for {path}: {len(magmoms)} != {len(atoms)}"
        )
    return magmoms, f"opt_second_frame:{path}@state{target_state}:d{target_distance:.2f}:step2{source_suffix}"


def get_scan_init_magmoms(
    atoms,
    spec: ModelSpec,
    mode: str,
    s0_rule: str,
    s2_rule: str,
    distance: float | None = None,
    scan_frame_pattern: str = "",
    scan_frame_index: str = "0",
    opt_second_magmom_pattern: str = "",
    opt_second_missing_policy: str = "error",
    outcar_mag_block_index: int = 0,
) -> tuple[np.ndarray, str]:
    if mode == "template":
        return safe_get_magmoms(atoms), f"template:{spec.template}@{spec.template_index}"
    if mode == "rule":
        rule = s0_rule if int(spec.state_id) == 0 else s2_rule
        return expand_magmom_rule(rule, len(atoms)), f"rule:{rule}"
    if mode == "scan_frame":
        if distance is None:
            raise ValueError("scan_frame init mode requires a scan distance.")
        scan_path = resolve_scan_frame_path(spec, distance, scan_frame_pattern)
        if not scan_path.exists():
            raise FileNotFoundError(f"Missing scan-frame magnetic initialization source: {scan_path}")
        scan_atoms = io.read(scan_path, index=scan_frame_index)
        magmoms = safe_get_magmoms(scan_atoms)
        if len(magmoms) != len(atoms):
            raise ValueError(
                f"scan_frame magmom length mismatch for {scan_path}: {len(magmoms)} != {len(atoms)}"
            )
        return magmoms, f"scan_frame:{scan_path}@{scan_frame_index}"
    if mode == "opt_second_frame":
        if distance is None:
            raise ValueError("opt_second_frame init mode requires a scan distance.")
        pattern = opt_second_magmom_pattern or scan_frame_pattern
        if not pattern:
            raise ValueError("opt_second_frame init mode requires --opt-second-magmom-pattern or --scan-frame-pattern.")
        scan_path = resolve_scan_frame_path(spec, distance, pattern)
        magmoms, source = read_opt_second_frame_magmom(
            scan_path,
            atoms,
            spec.state_id,
            distance,
            missing_policy=opt_second_missing_policy,
        )
        return magmoms, source
    if mode == "outcar_mag_block":
        if distance is None:
            raise ValueError("outcar_mag_block init mode requires a scan distance.")
        scan_path = resolve_scan_frame_path(spec, distance, scan_frame_pattern)
        if not scan_path.exists():
            raise FileNotFoundError(f"Missing OUTCAR magnetic initialization source: {scan_path}")
        magmoms = read_outcar_magnetization_block(scan_path, outcar_mag_block_index, len(atoms))
        return magmoms, f"outcar_mag_block:{scan_path}#{outcar_mag_block_index}"
    if mode == "vasp_input_magmom":
        if distance is None:
            raise ValueError("vasp_input_magmom init mode requires a scan distance.")
        scan_path = resolve_scan_frame_path(spec, distance, scan_frame_pattern)
        magmoms, source = read_vasp_input_magmom(scan_path, len(atoms))
        return magmoms, source
    raise ValueError(f"Unsupported init magmom mode: {mode!r}")


def get_scan_init_magmoms_piecewise(
    atoms,
    spec: ModelSpec,
    distance: float,
    default_mode: str,
    lowdist_mode: str | None,
    lowdist_interval: tuple[float, float] | None,
    s0_rule: str,
    s2_rule: str,
    scan_frame_pattern: str = "",
    scan_frame_index: str = "0",
    opt_second_magmom_pattern: str = "",
    opt_second_missing_policy: str = "error",
    outcar_mag_block_index: int = 0,
) -> tuple[np.ndarray, str]:
    active_mode = default_mode
    if lowdist_mode and lowdist_interval is not None:
        lo, hi = lowdist_interval
        if lo <= float(distance) <= hi:
            active_mode = lowdist_mode
    return get_scan_init_magmoms(
        atoms,
        spec,
        mode=active_mode,
        s0_rule=s0_rule,
        s2_rule=s2_rule,
        distance=distance,
        scan_frame_pattern=scan_frame_pattern,
        scan_frame_index=scan_frame_index,
        opt_second_magmom_pattern=opt_second_magmom_pattern,
        opt_second_missing_policy=opt_second_missing_policy,
        outcar_mag_block_index=outcar_mag_block_index,
    )


def apply_fe_magmom_shift(
    magmoms: np.ndarray,
    fe_index: int,
    distance: float,
    state_id: int,
    fe_magmom_shift: float,
    afm_interval: tuple[float, float] | None,
    fm_interval: tuple[float, float] | None,
) -> tuple[np.ndarray, bool]:
    shifted = magmoms.copy()
    active = False
    interval = afm_interval if int(state_id) == 0 else fm_interval
    if interval is not None and float(interval[0]) <= float(distance) <= float(interval[1]):
        shifted[int(fe_index)] = float(shifted[int(fe_index)] + float(fe_magmom_shift))
        active = True
    return shifted, active


def apply_o2_magmom_shift(
    magmoms: np.ndarray,
    o_indices: Sequence[int],
    distance: float,
    state_id: int,
    o2_magmom_shift: float,
    afm_interval: tuple[float, float] | None,
    fm_interval: tuple[float, float] | None,
) -> tuple[np.ndarray, bool]:
    shifted = magmoms.copy()
    active = False
    interval = afm_interval if int(state_id) == 0 else fm_interval
    if interval is not None and float(interval[0]) <= float(distance) <= float(interval[1]):
        delta = float(o2_magmom_shift) / 2.0
        for idx in o_indices:
            shifted[int(idx)] = float(shifted[int(idx)] + delta)
        active = True
    return shifted, active


def get_scan_structure_piecewise(
    spec: ModelSpec,
    distance: float,
    default_mode: str,
    lowdist_mode: str | None,
    lowdist_interval: tuple[float, float] | None,
    scan_structure_pattern: str = "",
) -> tuple[Path | None, str]:
    active_mode = default_mode
    if lowdist_mode and lowdist_interval is not None:
        lo, hi = lowdist_interval
        if lo <= float(distance) <= hi:
            active_mode = lowdist_mode
    if active_mode == "template":
        return None, f"template:{spec.template}@{spec.template_index}"
    if active_mode == "opt_source_structure":
        structure_path = resolve_scan_structure_path(spec, distance, scan_structure_pattern)
        if not structure_path.exists():
            raise FileNotFoundError(f"Missing scan structure source: {structure_path}")
        return structure_path, f"opt_source_structure:{structure_path}"
    raise ValueError(f"Unsupported scan structure mode: {active_mode!r}")


def _parse_distance_interval(raw: str) -> tuple[float, float] | None:
    raw = str(raw).strip()
    if not raw:
        return None
    parts = [x.strip() for x in raw.split(":")]
    if len(parts) != 2:
        raise ValueError(f"Distance interval must be START:STOP, got {raw!r}")
    return float(parts[0]), float(parts[1])


def retarget_o2_distance(atoms, fe_index: int, o_indices: Sequence[int], target_distance: float):
    atoms = atoms.copy()
    fe_pos = atoms.positions[int(fe_index)].copy()
    o1, o2 = [int(x) for x in o_indices]
    distances = [
        float(np.linalg.norm(atoms.positions[o1] - fe_pos)),
        float(np.linalg.norm(atoms.positions[o2] - fe_pos)),
    ]
    anchor = o1 if distances[0] <= distances[1] else o2
    direction = atoms.positions[anchor] - fe_pos
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("Degenerate Fe-anchor O distance in template.")
    delta = (float(target_distance) - norm) * direction / norm
    atoms.positions[[o1, o2]] += delta
    return atoms, anchor


def structure_metrics(atoms, fe_index: int, o_indices: Sequence[int], anchor_o_index: int):
    o1, o2 = [int(x) for x in o_indices]
    fe_o1 = float(atoms.get_distance(fe_index, o1, mic=True))
    fe_o2 = float(atoms.get_distance(fe_index, o2, mic=True))
    anchor = float(atoms.get_distance(fe_index, anchor_o_index, mic=True))
    oo = float(atoms.get_distance(o1, o2, mic=True))
    vec1 = atoms.positions[fe_index] - atoms.positions[anchor_o_index]
    other = o2 if anchor_o_index == o1 else o1
    vec2 = atoms.positions[other] - atoms.positions[anchor_o_index]
    angle = ""
    n1 = float(np.linalg.norm(vec1))
    n2 = float(np.linalg.norm(vec2))
    if n1 > 1e-12 and n2 > 1e-12:
        cosang = float(np.dot(vec1, vec2) / (n1 * n2))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    return {
        "fe_o1_distance_A": fe_o1,
        "fe_o2_distance_A": fe_o2,
        "fe_anchor_o_distance_A": anchor,
        "fe_o_min_distance_A": min(fe_o1, fe_o2),
        "o_o_distance_A": oo,
        "fe_anchor_o_other_o_angle_deg": angle,
    }


def harmonic_restraint(atoms, fe_index: int, anchor_o_index: int, target_distance: float, k: float):
    positions = torch.tensor(atoms.positions, dtype=torch.float32, requires_grad=True)
    vec = positions[anchor_o_index] - positions[fe_index]
    distance = torch.linalg.norm(vec)
    energy = 0.5 * float(k) * (distance - float(target_distance)) ** 2
    grad = torch.autograd.grad(energy, positions)[0]
    return float(energy.item()), -grad.detach().numpy()


def harmonic_angle_restraint(
    atoms,
    fe_index: int,
    anchor_o_index: int,
    other_o_index: int,
    target_angle_deg: float,
    k: float,
):
    positions = torch.tensor(atoms.positions, dtype=torch.float32, requires_grad=True)
    vec1 = positions[fe_index] - positions[anchor_o_index]
    vec2 = positions[other_o_index] - positions[anchor_o_index]
    n1 = torch.linalg.norm(vec1)
    n2 = torch.linalg.norm(vec2)
    cosang = torch.clamp(torch.dot(vec1, vec2) / (n1 * n2), -1.0, 1.0)
    angle = torch.acos(cosang)
    target = math.radians(float(target_angle_deg))
    energy = 0.5 * float(k) * (angle - target) ** 2
    grad = torch.autograd.grad(energy, positions)[0]
    return float(energy.item()), -grad.detach().numpy()


class MACEFilmScanCalculator(MACEFilmMDCalculator):
    def __init__(
        self,
        *args,
        fe_index,
        anchor_o_index,
        other_o_index,
        target_distance,
        restraint_k,
        angle_target_deg=None,
        angle_restraint_k=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fe_index = int(fe_index)
        self.anchor_o_index = int(anchor_o_index)
        self.other_o_index = int(other_o_index)
        self.target_distance = float(target_distance)
        self.restraint_k = float(restraint_k)
        self.angle_target_deg = None if angle_target_deg is None else float(angle_target_deg)
        self.angle_restraint_k = float(angle_restraint_k)

    def check_state(self, atoms, tol=1e-15):
        return all_changes

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e_rest, f_rest = harmonic_restraint(
            atoms,
            self.fe_index,
            self.anchor_o_index,
            self.target_distance,
            self.restraint_k,
        )
        self.results["energy"] += e_rest
        self.results["forces"] = self.results["forces"] + f_rest
        self.results["restraint_energy"] = e_rest
        if self.angle_target_deg is not None and self.angle_restraint_k > 0.0:
            e_angle, f_angle = harmonic_angle_restraint(
                atoms,
                self.fe_index,
                self.anchor_o_index,
                self.other_o_index,
                self.angle_target_deg,
                self.angle_restraint_k,
            )
            self.results["energy"] += e_angle
            self.results["forces"] = self.results["forces"] + f_angle
            self.results["angle_restraint_energy"] = e_angle


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label) or "model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-spec", action="append", required=True)
    parser.add_argument("--fe-index", type=int, default=54)
    parser.add_argument("--o-indices", type=int, nargs=2, default=[59, 60])
    parser.add_argument("--scan-distance-range", default="1.8:4.0:0.1")
    parser.add_argument(
        "--scan-distance-exclude",
        default="",
        help="Comma-separated scan distances to skip, e.g. 3.9 or 3.9,4.1.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--atomic-energy-linref",
        default=None,
        help="Portable override for the atomic-energy reference path stored in the checkpoint.",
    )
    parser.add_argument("--restraint-k", type=float, default=50.0)
    parser.add_argument(
        "--angle-restraint-k",
        type=float,
        default=0.0,
        help="Optional harmonic restraint on the Fe-anchor-O-other angle in eV/rad^2.",
    )
    parser.add_argument(
        "--angle-restraint-interval",
        default="",
        help="Optional distance interval START:STOP to enable angle restraint, e.g. 2.0:3.0 or 1.8:2.0.",
    )
    parser.add_argument(
        "--rollout-magmom",
        action="store_true",
        help=(
            "If set, feed the model-predicted magmom from each BFGS step back "
            "as the next optimization-step initial_magmom. Default keeps magmom "
            "fixed per scan point, matching the original PES behavior."
        ),
    )
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
        "--rollout-magmom-interval",
        type=int,
        default=1,
        help="How often to commit the rollout magmom during BFGS optimization.",
    )
    parser.add_argument(
        "--init-magmom-mode",
        choices=["template", "rule", "scan_frame", "opt_second_frame", "outcar_mag_block", "vasp_input_magmom"],
        default="template",
    )
    parser.add_argument("--s0-rule", default="54*0.1 1*-2 4*0.1 2*1")
    parser.add_argument("--s2-rule", default="54*0.1 1*2 4*0 2*1")
    parser.add_argument(
        "--scan-frame-pattern",
        default="data/scans/{state}/d_{distance}/OUTCAR",
        help=(
            "Path pattern used by --init-magmom-mode scan_frame. Available fields: "
            "{state}/{branch}=S0 or S2, {distance}=%%.2f, {distance_short}=%%.1f."
        ),
    )
    parser.add_argument(
        "--scan-structure-mode",
        choices=["template", "opt_source_structure"],
        default="template",
        help=(
            "Geometry source for each scan point. template keeps the current fixed "
            "model-spec template; opt_source_structure reads the per-distance DFT opt "
            "source_structure.vasp."
        ),
    )
    parser.add_argument(
        "--scan-structure-pattern",
        default="data/scans/{state}/d_{distance}/source_structure.vasp",
        help=(
            "Path pattern used by --scan-structure-mode opt_source_structure. "
            "Available fields: {state}/{branch}=S0 or S2, {distance}=%%.2f, "
            "{distance_short}=%%.1f."
        ),
    )
    parser.add_argument(
        "--lowdist-init-structure-mode-afm",
        default="",
        choices=["", "template", "opt_source_structure"],
        help="Optional alternate structure mode for AFM low-distance interval.",
    )
    parser.add_argument(
        "--lowdist-init-structure-interval-afm",
        default="",
        help="AFM distance interval START:STOP to use --lowdist-init-structure-mode-afm.",
    )
    parser.add_argument(
        "--lowdist-init-structure-mode-fm",
        default="",
        choices=["", "template", "opt_source_structure"],
        help="Optional alternate structure mode for FM low-distance interval.",
    )
    parser.add_argument(
        "--lowdist-init-structure-interval-fm",
        default="",
        help="FM distance interval START:STOP to use --lowdist-init-structure-mode-fm.",
    )
    parser.add_argument(
        "--fe-magmom-shift",
        type=float,
        default=0.0,
        help="Additive shift applied only to Fe initial magmom inside the selected AFM/FM short-distance window.",
    )
    parser.add_argument(
        "--fe-magmom-shift-interval-afm",
        default="",
        help="AFM distance interval START:STOP where the Fe initial magmom shift is active.",
    )
    parser.add_argument(
        "--fe-magmom-shift-interval-fm",
        default="",
        help="FM distance interval START:STOP where the Fe initial magmom shift is active.",
    )
    parser.add_argument(
        "--o2-magmom-shift",
        type=float,
        default=0.0,
        help="Additive shift applied to the O2 total initial magmom inside the selected AFM/FM short-distance window.",
    )
    parser.add_argument(
        "--o2-magmom-shift-interval-afm",
        default="",
        help="AFM distance interval START:STOP where the O2 total magmom shift is active.",
    )
    parser.add_argument(
        "--o2-magmom-shift-interval-fm",
        default="",
        help="FM distance interval START:STOP where the O2 total magmom shift is active.",
    )
    parser.add_argument(
        "--vasp-magmom-pattern",
        default="",
        help=(
            "Optional override path pattern used by --init-magmom-mode vasp_input_magmom. "
            "If omitted, scan-frame pattern is reused."
        ),
    )
    parser.add_argument(
        "--opt-second-magmom-pattern",
        default="",
        help=(
            "Optional path pattern used by --init-magmom-mode opt_second_frame. "
            "It should point to the merged extxyz that contains source_step metadata."
        ),
    )
    parser.add_argument(
        "--opt-second-missing-policy",
        choices=["error", "fallback_nearest"],
        default="error",
        help="How to handle a missing exact source_step=2 opt frame for opt_second_frame initialization.",
    )
    parser.add_argument(
        "--lowdist-init-magmom-mode",
        default="",
        choices=["", "template", "rule", "scan_frame", "opt_second_frame", "outcar_mag_block", "vasp_input_magmom"],
        help="Optional alternate init magmom mode for a low-distance interval.",
    )
    parser.add_argument(
        "--lowdist-init-distance-interval",
        default="",
        help="Distance interval START:STOP to use --lowdist-init-magmom-mode, e.g. 1.8:2.0.",
    )
    parser.add_argument(
        "--scan-frame-index",
        default="0",
        help="Frame index read from each DFT scan trajectory/OUTCAR for scan_frame initialization.",
    )
    parser.add_argument(
        "--outcar-mag-block-index",
        type=int,
        default=0,
        help=(
            "Magnetization block index read from OUTCAR for --init-magmom-mode "
            "outcar_mag_block. Use 1 for the second self-consistent magnetization block."
        ),
    )
    parser.add_argument("--opt-fmax", type=float, default=0.05)
    parser.add_argument("--opt-steps", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-plot", required=True)
    parser.add_argument("--output-structures-dir", required=True)
    args = parser.parse_args()

    distances = parse_distance_range(args.scan_distance_range)
    if args.scan_distance_exclude.strip():
        exclude = {
            round(float(item.strip()), 10)
            for item in args.scan_distance_exclude.split(",")
            if item.strip()
        }
        distances = [d for d in distances if round(float(d), 10) not in exclude]
    lowdist_interval = _parse_distance_interval(args.lowdist_init_distance_interval)
    lowdist_structure_interval_afm = _parse_distance_interval(args.lowdist_init_structure_interval_afm)
    lowdist_structure_interval_fm = _parse_distance_interval(args.lowdist_init_structure_interval_fm)
    fe_shift_interval_afm = _parse_distance_interval(args.fe_magmom_shift_interval_afm)
    fe_shift_interval_fm = _parse_distance_interval(args.fe_magmom_shift_interval_fm)
    o2_shift_interval_afm = _parse_distance_interval(args.o2_magmom_shift_interval_afm)
    o2_shift_interval_fm = _parse_distance_interval(args.o2_magmom_shift_interval_fm)
    angle_interval = _parse_distance_interval(args.angle_restraint_interval)
    specs = [parse_model_spec(raw) for raw in args.model_spec]
    out_struct_dir = Path(args.output_structures_dir)
    out_struct_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    energy_by_label = {spec.label: {} for spec in specs}
    total_jobs = len(specs) * len(distances)
    job = 0

    for spec in specs:
        for distance in distances:
            job += 1
            structure_path, structure_source_desc = get_scan_structure_piecewise(
                spec=spec,
                distance=distance,
                default_mode=args.scan_structure_mode,
                lowdist_mode=(
                    args.lowdist_init_structure_mode_afm if int(spec.state_id) == 0 else args.lowdist_init_structure_mode_fm
                )
                or None,
                lowdist_interval=(
                    lowdist_structure_interval_afm if int(spec.state_id) == 0 else lowdist_structure_interval_fm
                ),
                scan_structure_pattern=args.scan_structure_pattern,
            )
            if structure_path is None:
                template = io.read(spec.template, index=spec.template_index)
            else:
                template = io.read(structure_path)
            atoms, anchor = retarget_o2_distance(
                template,
                fe_index=args.fe_index,
                o_indices=args.o_indices,
                target_distance=distance,
            )
            other_o = int(args.o_indices[0]) if anchor == int(args.o_indices[1]) else int(args.o_indices[1])
            angle_target = None
            if args.angle_restraint_k > 0.0:
                enabled = angle_interval is None or (angle_interval[0] <= float(distance) <= angle_interval[1])
                if enabled:
                    angle_target = structure_metrics(
                        atoms,
                        args.fe_index,
                        args.o_indices,
                        anchor,
                    )["fe_anchor_o_other_o_angle_deg"]
            init_magmoms, init_magmom_source = get_scan_init_magmoms_piecewise(
                atoms,
                spec,
                distance=distance,
                default_mode=args.init_magmom_mode,
                lowdist_mode=args.lowdist_init_magmom_mode or None,
                lowdist_interval=lowdist_interval,
                s0_rule=args.s0_rule,
                s2_rule=args.s2_rule,
                scan_frame_pattern=args.vasp_magmom_pattern or args.scan_frame_pattern,
                scan_frame_index=args.scan_frame_index,
                opt_second_magmom_pattern=args.opt_second_magmom_pattern,
                opt_second_missing_policy=args.opt_second_missing_policy,
                outcar_mag_block_index=args.outcar_mag_block_index,
            )
            init_magmoms, fe_shift_active = apply_fe_magmom_shift(
                init_magmoms,
                fe_index=args.fe_index,
                distance=distance,
                state_id=spec.state_id,
                fe_magmom_shift=args.fe_magmom_shift,
                afm_interval=fe_shift_interval_afm,
                fm_interval=fe_shift_interval_fm,
            )
            init_magmoms, o2_shift_active = apply_o2_magmom_shift(
                init_magmoms,
                o_indices=args.o_indices,
                distance=distance,
                state_id=spec.state_id,
                o2_magmom_shift=args.o2_magmom_shift,
                afm_interval=o2_shift_interval_afm,
                fm_interval=o2_shift_interval_fm,
            )
            calc = MACEFilmScanCalculator(
                checkpoint=args.checkpoint,
                device=args.device,
                init_magmom=init_magmoms,
                rollout_magmom=args.rollout_magmom,
                atomic_energy_linref=args.atomic_energy_linref,
                enforce_spin_inversion=args.enforce_spin_inversion,
                fe_index=args.fe_index,
                anchor_o_index=anchor,
                other_o_index=other_o,
                target_distance=distance,
                restraint_k=args.restraint_k,
                angle_target_deg=angle_target,
                angle_restraint_k=args.angle_restraint_k,
            )
            atoms.calc = calc
            optimizer = BFGS(atoms, logfile=None)
            label = f"[{job}/{total_jobs}] model={spec.label} scan_distance={distance:.2f}A"
            print(f"{label} start", flush=True)
            step_counter = {"value": 0}

            def log_progress():
                step_counter["value"] += 1
                step = step_counter["value"]
                if step == 1 or step % max(args.progress_every, 1) == 0:
                    e = float(atoms.get_potential_energy())
                    fmax = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max()))
                    r = float(atoms.get_distance(args.fe_index, anchor, mic=True))
                    print(
                        f"{label} opt_step={step} energy={e:.6f} eV "
                        f"r(Fe-O_anchor)={r:.4f} A dr={r - distance:+.4f} A fmax={fmax:.4f}",
                        flush=True,
                    )

            optimizer.attach(log_progress, interval=1)
            if args.rollout_magmom:
                optimizer.attach(
                    calc.commit_magmom_rollout,
                    interval=max(int(args.rollout_magmom_interval), 1),
                )
            optimizer.run(fmax=args.opt_fmax, steps=args.opt_steps)
            total_energy = float(atoms.get_potential_energy())
            magmoms = np.asarray(calc.results.get("magmoms"), dtype=float)
            metrics = structure_metrics(atoms, args.fe_index, args.o_indices, anchor)
            path = out_struct_dir / f"{sanitize_label(spec.label)}_d{distance:.2f}_constrained_opt.vasp"
            io.write(path, atoms, format="vasp")
            energy_by_label[spec.label][float(distance)] = total_energy
            row = {
                "mode": "constrained_opt",
                "scan_init_mode": "independent",
                "constraint_mode": "harmonic",
                "restraint_k_eV_per_A2": float(args.restraint_k),
                "angle_restraint_k_eV_per_rad2": float(args.angle_restraint_k),
                "scan_displace_mode": "o2_rigid",
                "init_magmom_mode": args.init_magmom_mode,
                "init_magmom_source": init_magmom_source,
                "fe_magmom_shift": float(args.fe_magmom_shift),
                "fe_magmom_shift_active": bool(fe_shift_active),
                "o2_magmom_shift": float(args.o2_magmom_shift),
                "o2_magmom_shift_active": bool(o2_shift_active),
                "scan_structure_mode": args.scan_structure_mode,
                "scan_structure_source": structure_source_desc,
                "model_label": spec.label,
                "state_id": int(spec.state_id),
                "scan_distance": float(distance),
                "template": spec.template,
                "saved_structure_path": str(path),
                "total_energy_eV": total_energy,
                "residual_total_energy_eV": float(calc.results.get("residual_total_energy", np.nan)),
                "linref_total_energy_eV": float(calc.results.get("linref_total_energy", np.nan)),
                "total_magmom": float(np.sum(magmoms)),
                "max_abs_magmom": float(np.max(np.abs(magmoms))),
                "fe_magmom": float(magmoms[args.fe_index]),
                "o1_magmom": float(magmoms[int(args.o_indices[0])]),
                "o2_magmom": float(magmoms[int(args.o_indices[1])]),
                "init_total_magmom": float(np.sum(init_magmoms)),
                "init_fe_magmom": float(init_magmoms[args.fe_index]),
                "init_o1_magmom": float(init_magmoms[int(args.o_indices[0])]),
                "init_o2_magmom": float(init_magmoms[int(args.o_indices[1])]),
                **metrics,
            }
            if angle_target is not None:
                row["angle_restraint_target_deg"] = float(angle_target)
            rows.append(row)
            print(f"{label} done total_energy={total_energy:.6f} eV saved={path}", flush=True)

    for spec in specs:
        reference_distance = max(energy_by_label[spec.label])
        reference_energy = energy_by_label[spec.label][reference_distance]
        for row in rows:
            if row["model_label"] == spec.label:
                row["reference_distance"] = reference_distance
                row["reference_total_energy_eV"] = reference_energy
                row["adsorption_energy_relative_eV"] = row["total_energy_eV"] - reference_energy

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
        xs = [float(row["fe_anchor_o_distance_A"]) for row in model_rows]
        ys = [float(row["adsorption_energy_relative_eV"]) for row in model_rows]
        plt.plot(xs, ys, marker="o", label=spec.label)
    plt.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    plt.xlabel("Actual Fe-anchor O distance (A)")
    plt.ylabel("Relative adsorption energy (eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Wrote plot to {out_plot}")


if __name__ == "__main__":
    main()
