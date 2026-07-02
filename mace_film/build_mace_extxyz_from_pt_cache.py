#!/usr/bin/env python3
import argparse
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms


def _data_index(path):
    stem = Path(path).stem
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _state_id_from_frameid(frameid):
    text = str(frameid)
    if text.startswith("src0:") or text.startswith("src2:"):
        return 0
    if text.startswith("src1:") or text.startswith("src3:"):
        return 1
    raise ValueError(f"Cannot infer state_id from frameid={frameid!r}")


def _sample_group_from_frameid(frameid):
    text = str(frameid)
    if text.startswith("src0:"):
        return "aimd_s0"
    if text.startswith("src1:"):
        return "aimd_s2"
    if text.startswith("src2:"):
        return "opt_s0"
    if text.startswith("src3:"):
        return "opt_s2"
    return "unknown"


def _trajectory_key_from_frameid(frameid):
    text = str(frameid)
    prefix, sep, suffix = text.rpartition("-")
    if sep and suffix.isdigit():
        return prefix
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Convert a GemOC/GemNet PT cache into MACE-FiLM extxyz."
    )
    parser.add_argument("--pt-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--rebuild-previous-inmag",
        action="store_true",
        help=(
            "Ignore cached Data.inmag and rebuild initial_magmom from the previous "
            "target_magmom within each trajectory key parsed from frameid. This "
            "avoids target leakage when a PT cache stores inmag == y."
        ),
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to read PT cache Data objects") from exc

    cache = Path(args.pt_cache)
    paths = sorted(cache.glob("data_*.pt"), key=_data_index)
    if not paths:
        raise FileNotFoundError(f"No data_*.pt files found in {cache}")

    frames = []
    counts = {}
    previous_target_by_traj = {}
    equal_initial_target = 0
    for path in paths:
        data = torch.load(path, map_location="cpu")
        frameid = str(getattr(data, "frameid", path.stem))
        atomic_numbers = data.atomic_numbers.detach().cpu().numpy().astype(int)
        pos = data.pos.detach().cpu().numpy().astype(float)
        cell = data.cell.detach().cpu().numpy().astype(float)
        if cell.ndim == 3:
            cell = cell[0]
        pbc = data.pbc.detach().cpu().numpy().astype(bool)
        atoms = Atoms(numbers=atomic_numbers, positions=pos, cell=cell, pbc=pbc)

        if not hasattr(data, "raw_total_energy"):
            raise ValueError(f"{path} is missing raw_total_energy")
        if not hasattr(data, "force"):
            raise ValueError(f"{path} is missing force")
        if not hasattr(data, "y"):
            raise ValueError(f"{path} is missing target magmom y")
        if not hasattr(data, "inmag"):
            raise ValueError(f"{path} is missing initial magmom inmag")

        atoms.info["REF_energy"] = float(data.raw_total_energy.reshape(-1)[0].item())
        atoms.info["state_id"] = _state_id_from_frameid(frameid)
        atoms.info["frameid"] = frameid
        atoms.info["sample_group"] = _sample_group_from_frameid(frameid)
        atoms.arrays["REF_forces"] = data.force.detach().cpu().numpy().astype(float)
        target_magmom = data.y.detach().cpu().numpy().astype(float).reshape(-1)
        if args.rebuild_previous_inmag:
            traj_key = _trajectory_key_from_frameid(frameid)
            init_magmom = previous_target_by_traj.get(traj_key, target_magmom).copy()
            previous_target_by_traj[traj_key] = target_magmom.copy()
        else:
            init_magmom = data.inmag.detach().cpu().numpy().astype(float).reshape(-1)
        atoms.arrays["target_magmom"] = target_magmom
        atoms.arrays["initial_magmom"] = init_magmom

        if len(atoms.arrays["target_magmom"]) != len(atoms):
            raise ValueError(f"{path} target_magmom length does not match atoms")
        if len(atoms.arrays["initial_magmom"]) != len(atoms):
            raise ValueError(f"{path} initial_magmom length does not match atoms")
        if not np.any(np.abs(atoms.arrays["target_magmom"]) > 0):
            raise ValueError(f"{path} has all-zero target_magmom")
        equal_initial_target += int(
            np.allclose(
                atoms.arrays["initial_magmom"],
                atoms.arrays["target_magmom"],
                rtol=0.0,
                atol=1e-12,
            )
        )

        group = atoms.info["sample_group"]
        counts[group] = counts.get(group, 0) + 1
        frames.append(atoms)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(out, frames, format="extxyz")
    print(f"Wrote {len(frames)} frames to {out}")
    for group, count in sorted(counts.items()):
        print(f"  {group}: {count}")
    print(
        "  initial==target: "
        f"{equal_initial_target}/{len(frames)} "
        f"({equal_initial_target / max(len(frames), 1):.4%})"
    )


if __name__ == "__main__":
    main()
