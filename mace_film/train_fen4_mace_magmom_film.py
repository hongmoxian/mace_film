#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from e3nn import o3

from mace import data as mace_data
from mace import modules as mace_modules
from mace.tools import torch_geometric
from mace.tools.utils import AtomicNumberTable

from mace_magmom_film import MACEInitialMagmomFiLM


class EnergyForceMagmomLoss(torch.nn.Module):
    def __init__(
        self,
        energy_w=1.0,
        force_w=100.0,
        magmom_w=100.0,
        total_magmom_w=1.0,
        loss_type="l1",
    ):
        super().__init__()
        self.energy_w = float(energy_w)
        self.force_w = float(force_w)
        self.magmom_w = float(magmom_w)
        self.total_magmom_w = float(total_magmom_w)
        loss_type = str(loss_type).strip().lower()
        if loss_type not in {"l1", "smoothl1"}:
            raise ValueError(f"Unsupported loss_type={loss_type!r}")
        self.loss_type = loss_type

    def _elem_loss(self, pred, target):
        if self.loss_type == "smoothl1":
            return torch.nn.functional.smooth_l1_loss(pred, target)
        return torch.mean(torch.abs(pred - target))

    def components(self, pred, ref):
        natoms = ref.ptr[1:] - ref.ptr[:-1]
        energy_loss = self._elem_loss(
            (pred["energy"] - ref.energy) / natoms,
            torch.zeros_like(ref.energy),
        )
        force_loss = self._elem_loss(pred["forces"], ref.forces)
        target_mag = ref.target_magmom.reshape(-1).to(pred["magmom"].dtype)
        magmom_loss = self._elem_loss(pred["magmom"].reshape(-1), target_mag)
        graph_index = torch.repeat_interleave(
            torch.arange(len(natoms), device=pred["magmom"].device),
            natoms.to(pred["magmom"].device),
        )
        pred_total = torch.zeros(len(natoms), device=pred["magmom"].device, dtype=pred["magmom"].dtype)
        targ_total = torch.zeros_like(pred_total)
        pred_total.index_add_(0, graph_index, pred["magmom"].reshape(-1))
        targ_total.index_add_(0, graph_index, target_mag)
        total_magmom_loss = self._elem_loss(pred_total, targ_total)
        total = (
            self.energy_w * energy_loss
            + self.force_w * force_loss
            + self.magmom_w * magmom_loss
            + self.total_magmom_w * total_magmom_loss
        )
        return total, energy_loss, force_loss, magmom_loss, total_magmom_loss

    def forward(self, pred, ref):
        return self.components(pred, ref)[0]


def _load_configs(
    path,
    train_fraction,
    val_fraction,
    test_fraction,
    seed,
    max_initial_equals_target_fraction,
):
    key_spec = mace_data.KeySpecification.from_defaults()
    key_spec.arrays_keys["initial_magmom"] = "initial_magmom"
    key_spec.arrays_keys["target_magmom"] = "target_magmom"
    key_spec.info_keys["state_id"] = "state_id"
    _, configs = mace_data.load_from_xyz(
        file_path=str(path),
        key_specification=key_spec,
        head_name="Default",
        config_type_weights={"Default": 1.0},
        extract_atomic_energies=False,
        keep_isolated_atoms=True,
    )
    _check_magmom_initialization(
        configs,
        max_equal_fraction=max_initial_equals_target_fraction,
    )
    by_state = {}
    for idx, cfg in enumerate(configs):
        state = cfg.properties.get("state_id", "unknown")
        try:
            state = int(state)
        except (TypeError, ValueError):
            state = "unknown"
        by_state.setdefault(state, []).append((idx, cfg))

    train, val, test = [], [], []
    split_summary = {}
    for state, indexed_configs in sorted(by_state.items(), key=lambda item: str(item[0])):
        rng = np.random.default_rng(seed + (0 if state == "unknown" else int(state) * 1009))
        order = np.arange(len(indexed_configs))
        rng.shuffle(order)
        n_total = len(indexed_configs)
        n_train, n_val, n_test = _split_counts(
            n_total, train_fraction, val_fraction, test_fraction
        )
        train_local = [indexed_configs[i][1] for i in order[:n_train]]
        val_local = [indexed_configs[i][1] for i in order[n_train : n_train + n_val]]
        test_local = [
            indexed_configs[i][1]
            for i in order[n_train + n_val : n_train + n_val + n_test]
        ]
        train.extend(train_local)
        val.extend(val_local)
        test.extend(test_local)
        split_summary[str(state)] = {
            "num_train": len(train_local),
            "num_val": len(val_local),
            "num_test": len(test_local),
            "num_total": n_total,
        }
    return train, val, test, split_summary


def _check_magmom_initialization(configs, max_equal_fraction=0.05):
    equal_count = 0
    checked = 0
    max_diffs = []
    for cfg in configs:
        if "initial_magmom" not in cfg.properties or "target_magmom" not in cfg.properties:
            raise KeyError("Dataset must contain initial_magmom and target_magmom")
        init = np.asarray(cfg.properties["initial_magmom"], dtype=float).reshape(-1)
        target = np.asarray(cfg.properties["target_magmom"], dtype=float).reshape(-1)
        if init.shape != target.shape:
            raise ValueError(
                f"initial_magmom shape {init.shape} != target_magmom shape {target.shape}"
            )
        diff = np.abs(init - target)
        max_diffs.append(float(diff.max()) if diff.size else 0.0)
        equal_count += int(np.allclose(init, target, rtol=0.0, atol=1e-12))
        checked += 1
    equal_fraction = equal_count / max(checked, 1)
    print(
        "Magmom initialization audit: "
        f"initial==target {equal_count}/{checked} ({equal_fraction:.4%}), "
        f"mean max|diff|={np.mean(max_diffs) if max_diffs else 0.0:.6f}, "
        f"max max|diff|={np.max(max_diffs) if max_diffs else 0.0:.6f}",
        flush=True,
    )
    if equal_fraction > float(max_equal_fraction):
        raise ValueError(
            "Too many frames have initial_magmom == target_magmom "
            f"({equal_fraction:.2%} > {max_equal_fraction:.2%}). "
            "This usually means the previous-input dataset was built from the "
            "target moments instead of the previous frame moments."
        )


def _split_counts(n_total, train_fraction, val_fraction, test_fraction):
    n_train = int(round(n_total * train_fraction))
    n_val = int(round(n_total * val_fraction))
    n_test = int(round(n_total * test_fraction))
    if n_train + n_val + n_test != n_total:
        n_train += n_total - (n_train + n_val + n_test)
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = max(1, n_test)
    if n_train + n_val + n_test > n_total:
        n_train = max(1, n_total - n_val - n_test)
    return n_train, n_val, n_test


def _infer_atomic_number_table(config_groups):
    zs = set()
    for configs in config_groups:
        for cfg in configs:
            numbers = getattr(cfg, "atomic_numbers", None)
            if numbers is None:
                numbers = cfg.atomic_numbers
            zs.update(int(z) for z in np.asarray(numbers, dtype=int).reshape(-1))
    if not zs:
        raise ValueError("Cannot infer atomic numbers from the extxyz dataset.")
    ordered = sorted(zs)
    print(f"Inferred atomic numbers: {ordered}", flush=True)
    return AtomicNumberTable(ordered)


def _attach_magmom_fields(data, cfg):
    # MACE AtomicData only keeps recognized fields by default. Preserve the
    # magnetic conditioning/target arrays explicitly for PyG batching.
    for key in ("initial_magmom", "target_magmom"):
        if key not in cfg.properties or cfg.properties[key] is None:
            raise KeyError(f"Configuration is missing required {key!r}")
        values = np.asarray(cfg.properties[key], dtype=float).reshape(-1)
        if len(values) != len(cfg.atomic_numbers):
            raise ValueError(
                f"{key} length {len(values)} != natoms {len(cfg.atomic_numbers)}"
            )
        setattr(data, key, torch.as_tensor(values, dtype=torch.get_default_dtype()))
    return data


def _to_atomic_dataset(configs, z_table, cutoff):
    dataset = []
    for cfg in configs:
        data = mace_data.AtomicData.from_config(
            cfg,
            z_table=z_table,
            cutoff=cutoff,
            heads=["Default"],
        )
        dataset.append(_attach_magmom_fields(data, cfg))
    return dataset


def _write_split_summary(path, train_ds, val_ds, test_ds, by_state):
    summary = {
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "num_total": len(train_ds) + len(val_ds) + len(test_ds),
        "by_state_id": by_state,
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _print_split_summary(summary):
    total = max(int(summary["num_total"]), 1)
    print("Dataset split summary:", flush=True)
    for state, item in sorted(summary["by_state_id"].items(), key=lambda kv: str(kv[0])):
        state_name = {"0": "AFM", "1": "FM"}.get(str(state), f"state_{state}")
        n_state = max(int(item["num_total"]), 1)
        train = int(item["num_train"])
        val = int(item["num_val"])
        test = int(item["num_test"])
        print(
            f"  {state_name} (state_id={state}): "
            f"train={train} ({train / n_state:.3f}), "
            f"val={val} ({val / n_state:.3f}), "
            f"test={test} ({test / n_state:.3f}), "
            f"total={n_state}",
            flush=True,
        )
    print(
        "  TOTAL: "
        f"train={summary['num_train']} ({summary['num_train'] / total:.3f}), "
        f"val={summary['num_val']} ({summary['num_val'] / total:.3f}), "
        f"test={summary['num_test']} ({summary['num_test'] / total:.3f}), "
        f"total={summary['num_total']}",
        flush=True,
    )


def _load_atomic_energies(path, z_table):
    if path is None:
        return np.zeros(len(z_table), dtype=np.float64)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "atomic_numbers" not in payload or "coefficients" not in payload:
        raise ValueError(
            f"{path} must contain 'atomic_numbers' and 'coefficients' fields."
        )
    numbers = [int(z) for z in payload["atomic_numbers"]]
    coeffs = [float(v) for v in payload["coefficients"]]
    if len(numbers) != len(coeffs):
        raise ValueError(
            f"{path} has {len(numbers)} atomic_numbers but {len(coeffs)} coefficients."
        )
    by_z = dict(zip(numbers, coeffs))
    missing = [int(z) for z in z_table.zs if int(z) not in by_z]
    if missing:
        raise ValueError(f"{path} is missing atomic reference energies for Z={missing}.")
    return np.asarray([by_z[int(z)] for z in z_table.zs], dtype=np.float64)


def _compute_magmom_scale(configs, z_table):
    by_z = {}
    for cfg in configs:
        numbers = np.asarray(cfg.atomic_numbers, dtype=int).reshape(-1)
        target = np.asarray(cfg.properties["target_magmom"], dtype=float).reshape(-1)
        for z, m in zip(numbers, target):
            by_z.setdefault(int(z), []).append(abs(float(m)))
    scale = []
    for z in z_table.zs:
        vals = by_z.get(int(z), [])
        scale.append(float(max(vals)) if vals else 1.0)
    return scale


def _build_model(args, z_table):
    hidden_irreps = o3.Irreps(args.hidden_irreps)
    atomic_energies = _load_atomic_energies(args.atomic_energy_linref, z_table)
    print(
        "Atomic reference energies:",
        {int(z): float(e0) for z, e0 in zip(z_table.zs, atomic_energies)},
        flush=True,
    )
    magmom_scale = getattr(args, "magmom_scale", None)
    if magmom_scale is not None:
        print(
            "Per-element magmom normalization scale:",
            {int(z): float(s) for z, s in zip(z_table.zs, magmom_scale)},
            flush=True,
        )
    print(
        "FiLM settings:",
        {
            "conditioning_mode": str(getattr(args, "conditioning_mode", "film")),
            "film_scale": float(getattr(args, "film_scale", 1.0)),
            "film_use_tanh": bool(getattr(args, "film_use_tanh", False)),
            "magmom_input_clip": getattr(args, "magmom_input_clip", None),
            "film_element_embedding_dim": int(
                getattr(args, "film_element_embedding_dim", 0)
            ),
        },
        flush=True,
    )
    return MACEInitialMagmomFiLM(
        r_max=args.cutoff,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=mace_modules.interaction_classes[args.interaction],
        interaction_cls_first=mace_modules.interaction_classes[args.interaction_first],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=hidden_irreps,
        MLP_irreps=o3.Irreps(args.mlp_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=args.correlation,
        gate=mace_modules.gate_dict[args.gate],
        pair_repulsion=args.use_zbl,
        atomic_inter_scale=1.0,
        atomic_inter_shift=[0.0],
        radial_MLP=[64, 64, 64],
        heads=["Default"],
        magmom_scale=magmom_scale,
        conditioning_mode=getattr(args, "conditioning_mode", "film"),
        film_scale=getattr(args, "film_scale", 1.0),
        film_use_tanh=getattr(args, "film_use_tanh", False),
        magmom_input_clip=getattr(args, "magmom_input_clip", None),
        film_element_embedding_dim=getattr(args, "film_element_embedding_dim", 0),
    )


def _run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {
        "loss": 0.0,
        "energy": 0.0,
        "force": 0.0,
        "magmom": 0.0,
        "total_magmom": 0.0,
        "n": 0,
    }
    for batch in loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = model(batch_dict, training=training, compute_force=True)
        total, e_loss, f_loss, m_loss, t_loss = loss_fn.components(pred, batch)
        if training:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        n = int(batch.num_graphs)
        sums["loss"] += float(total.detach().cpu()) * n
        sums["energy"] += float(e_loss.detach().cpu()) * n
        sums["force"] += float(f_loss.detach().cpu()) * n
        sums["magmom"] += float(m_loss.detach().cpu()) * n
        sums["total_magmom"] += float(t_loss.detach().cpu()) * n
        sums["n"] += n
    return {
        k: sums[k] / max(sums["n"], 1)
        for k in ("loss", "energy", "force", "magmom", "total_magmom")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extxyz", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--energy-w", type=float, default=1.0)
    parser.add_argument("--force-w", type=float, default=100.0)
    parser.add_argument("--magmom-w", type=float, default=100.0)
    parser.add_argument("--total-magmom-w", type=float, default=1.0)
    parser.add_argument(
        "--loss-type",
        choices=["l1", "smoothl1"],
        default="l1",
    )
    parser.add_argument(
        "--atomic-energy-linref",
        default=None,
        help=(
            "JSON file with atomic_numbers and coefficients. These values are "
            "used as MACE E0 terms, so the network learns the residual energy."
        ),
    )
    parser.add_argument("--use-zbl", action="store_true")
    parser.add_argument(
        "--conditioning-mode",
        choices=["film", "residual_film", "concat"],
        default="film",
        help=(
            "Magmom conditioning mode. 'film' keeps the existing behavior; "
            "'residual_film' weakens it; 'concat' replaces modulation with "
            "a concat+projection style condition."
        ),
    )
    parser.add_argument(
        "--film-scale",
        type=float,
        default=1.0,
        help=(
            "Residual FiLM amplitude. Old behavior is 1.0. Values such as 0.1 "
            "make magnetic conditioning more conservative during rollout."
        ),
    )
    parser.add_argument(
        "--film-use-tanh",
        action="store_true",
        help="Bound FiLM gamma/beta residuals with tanh before applying film-scale.",
    )
    parser.add_argument(
        "--magmom-input-clip",
        type=float,
        default=None,
        help=(
            "Optional symmetric clamp for normalized initial magmom before FiLM. "
            "For example, 2.0 limits rollout extrapolation beyond twice the "
            "training normalization scale."
        ),
    )
    parser.add_argument(
        "--film-element-embedding-dim",
        type=int,
        default=0,
        help=(
            "Optional element embedding dimension concatenated with normalized "
            "magmom before FiLM. Keep 0 for the shared FiLM baseline."
        ),
    )
    parser.add_argument(
        "--max-initial-equals-target-fraction",
        type=float,
        default=0.05,
        help=(
            "Maximum allowed fraction of frames where initial_magmom exactly "
            "matches target_magmom. Keep this low to catch target leakage; "
            "scan/opt mixed datasets may need a slightly higher value because "
            "each short trajectory can contribute an unavoidable first frame."
        ),
    )
    parser.add_argument("--hidden-irreps", default="64x0e + 64x1o")
    parser.add_argument("--mlp-irreps", default="16x0e")
    parser.add_argument("--num-interactions", type=int, default=3)
    parser.add_argument("--max-ell", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--num-radial-basis", type=int, default=8)
    parser.add_argument("--num-cutoff-basis", type=int, default=5)
    parser.add_argument("--avg-num-neighbors", type=float, default=30.0)
    parser.add_argument("--interaction", default="RealAgnosticResidualInteractionBlock")
    parser.add_argument("--interaction-first", default="RealAgnosticResidualInteractionBlock")
    parser.add_argument("--gate", default="silu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Load the best checkpoint from --save-dir and only run the test split.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional checkpoint to initialize model weights before training.",
    )
    args = parser.parse_args()

    frac_sum = args.train_fraction + args.val_fraction + args.test_fraction
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test fractions must sum to 1.0, got {frac_sum:.6f}"
        )

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_configs, val_configs, test_configs, split_by_state = _load_configs(
        args.extxyz,
        args.train_fraction,
        args.val_fraction,
        args.test_fraction,
        args.seed,
        args.max_initial_equals_target_fraction,
    )
    z_table = _infer_atomic_number_table([train_configs, val_configs, test_configs])
    args.magmom_scale = _compute_magmom_scale(
        train_configs + val_configs, z_table
    )
    train_ds = _to_atomic_dataset(train_configs, z_table, args.cutoff)
    val_ds = _to_atomic_dataset(val_configs, z_table, args.cutoff)
    test_ds = _to_atomic_dataset(test_configs, z_table, args.cutoff)

    train_loader = torch_geometric.dataloader.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch_geometric.dataloader.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = torch_geometric.dataloader.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    split_summary = _write_split_summary(
        save_dir / "split_summary.json", train_ds, val_ds, test_ds, split_by_state
    )
    _print_split_summary(split_summary)

    model = _build_model(args, z_table).to(device)
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {resume_path}")
        payload = torch.load(resume_path, map_location=device)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict)
        print(f"Loaded resume checkpoint: {resume_path}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    loss_fn = EnergyForceMagmomLoss(
        args.energy_w,
        args.force_w,
        args.magmom_w,
        args.total_magmom_w,
        args.loss_type,
    )

    if not args.test_only:
        metrics_path = save_dir / "metrics.csv"
        best = float("inf")
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "lr",
                    "train_loss",
                    "train_energy_loss",
                    "train_force_loss",
                    "train_magmom_loss",
                    "train_total_magmom_loss",
                    "val_loss",
                    "val_energy_loss",
                    "val_force_loss",
                    "val_magmom_loss",
                    "val_total_magmom_loss",
                ],
            )
            writer.writeheader()
            for epoch in range(args.max_epochs):
                train_m = _run_epoch(model, train_loader, loss_fn, device, optimizer)
                with torch.enable_grad():
                    val_m = _run_epoch(model, val_loader, loss_fn, device, optimizer=None)
                scheduler.step(val_m["loss"])
                lr = optimizer.param_groups[0]["lr"]
                row = {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_m["loss"],
                    "train_energy_loss": train_m["energy"],
                    "train_force_loss": train_m["force"],
                    "train_magmom_loss": train_m["magmom"],
                    "train_total_magmom_loss": train_m["total_magmom"],
                    "val_loss": val_m["loss"],
                    "val_energy_loss": val_m["energy"],
                    "val_force_loss": val_m["force"],
                    "val_magmom_loss": val_m["magmom"],
                    "val_total_magmom_loss": val_m["total_magmom"],
                }
                writer.writerow(row)
                handle.flush()
                print(row, flush=True)
                if val_m["loss"] < best:
                    best = val_m["loss"]
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "args": vars(args),
                            "epoch": epoch,
                            "val_loss": best,
                        },
                        save_dir / "best.pt",
                    )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                    },
                    save_dir / "last.pt",
                )

    best_path = save_dir / "best.pt"
    if best_path.exists():
        payload = torch.load(best_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])

    with torch.enable_grad():
        test_m = _run_epoch(model, test_loader, loss_fn, device, optimizer=None)
    test_path = save_dir / "test_metrics.csv"
    with test_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "loss",
                "energy_loss",
                "force_loss",
                "magmom_loss",
                "total_magmom_loss",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "test",
                "loss": test_m["loss"],
                "energy_loss": test_m["energy"],
                "force_loss": test_m["force"],
                "magmom_loss": test_m["magmom"],
                "total_magmom_loss": test_m["total_magmom"],
            }
        )
    print(
        {
            "test_loss": test_m["loss"],
            "test_energy_loss": test_m["energy"],
            "test_force_loss": test_m["force"],
            "test_magmom_loss": test_m["magmom"],
            "test_total_magmom_loss": test_m["total_magmom"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
