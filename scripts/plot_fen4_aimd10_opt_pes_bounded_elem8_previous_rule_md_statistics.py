#!/usr/bin/env python3
"""Plot FeN4 AIMD10-opt-PES bounded FiLM MD statistics comparing AIMD with ML-MD.

The figure compares AIMD with top-3 ML-MD seeds from the previous-input and
rule-input models for the aimd10_opt_pes series. The top-3 seeds are selected
independently for each model using the same distribution-level score against AIMD.

Note: AIMD is recorded every 1 step, while ML-MD is recorded every 20 steps.
The script properly handles this step difference when comparing distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import io


ROOT = Path("/home/wuzhihong/software/ocp_mag")
WORK = ROOT / "work_fen4_escn_zbl_mdtest/mace_film"
MD_ROOT = WORK / "md_outputs"
OUT_DIR = WORK / "paper_figures" / "aimdonly_bounded_elem8_previous_rule_md_statistics_v2"
CACHE_DIR = OUT_DIR / "_cache"

FE = 54
O1 = 59
O2 = 60
SEEDS = [101, 102, 103, 104, 105]

STATE_CONFIG = {
    "afm": {
        "title": "AFM",
        "aimd_csv": ROOT / "paper_jctc/organized_artifacts/FeN4_O2/data/figure2_previous_S0_merged.csv",
        "dft_outcar": Path("/home/wuzhihong/vasp/FeN4/S1/s1-o2-0/md/50k/OUTCAR"),
        "dft_index": ":",
        "previous_root": MD_ROOT / "fen4_mace_film_aimdonly_zbl_bounded_elem8_previous_afm_secondframe_10ps_5seed",
        "previous_prefix": "afm_secondframe_seed_",
        "rule_root": MD_ROOT / "fen4_mace_film_aimdonly_zbl_bounded_elem8_rule_afm_secondframe_10ps_5seed",
        "rule_prefix": "afm_secondframe_seed_",
        "out_stem": "fen4_aimdonly_bounded_elem8_afm_previous_rule_top3_statistics",
    },
    "fm": {
        "title": "FM",
        "aimd_csv": ROOT / "paper_jctc/organized_artifacts/FeN4_O2/data/figure2_previous_S2_merged.csv",
        "dft_outcar": Path("/home/wuzhihong/vasp/FeN4/S1/s1-o2-2/md/50k/OUTCAR"),
        "dft_start_frame": 5401,
        "previous_root": MD_ROOT / "fen4_mace_film_aimdonly_zbl_bounded_elem8_previous_fm_frame5401_10ps_5seed",
        "previous_prefix": "fm_frame5401_seed_",
        "rule_root": MD_ROOT / "fen4_mace_film_aimdonly_zbl_bounded_elem8_rule_fm_frame5401_10ps_5seed",
        "rule_prefix": "fm_frame5401_seed_",
        "out_stem": "fen4_aimdonly_bounded_elem8_fm_previous_rule_top3_statistics",
    },
}

METRICS = ["r_fe_o_min", "r_fe_o2_centroid", "r_oo", "o2_axis_angle_deg", "m_fe", "m_o2_sum"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.6,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec * 0.0
    return vec / norm


def angle_o2_axis_to_fe_centroid(atoms) -> float:
    o_axis = atoms.get_distances(O1, [O2], mic=True, vector=True)[0]
    fe_to_o1 = atoms.get_distances(FE, [O1], mic=True, vector=True)[0]
    fe_to_o2 = atoms.get_distances(FE, [O2], mic=True, vector=True)[0]
    fe_to_centroid = 0.5 * (fe_to_o1 + fe_to_o2)
    cosang = abs(float(np.dot(safe_unit(o_axis), safe_unit(fe_to_centroid))))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def hist_density(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), hist


def load_magmom_jsonl(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["step"])] = np.asarray(row["magmoms"], dtype=float)
    return out


def load_seed(run_root: Path, prefix: str, state: str, model: str, seed: int, max_time_fs: float) -> pd.DataFrame:
    run_dir = run_root / f"{prefix}{seed}"
    traj_path = run_dir / "md.traj"
    log_path = run_dir / "md_log.csv"
    mag_path = run_dir / "md_magmom.jsonl"
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    if not mag_path.exists():
        raise FileNotFoundError(mag_path)

    logs = pd.read_csv(log_path)
    logs = logs[logs["time_fs"] <= max_time_fs].reset_index(drop=True)
    frames = io.read(str(traj_path), index=f":{len(logs)}")
    mag_by_step = load_magmom_jsonl(mag_path)
    n = min(len(frames), len(logs))
    rows = []
    for i in range(n):
        atoms = frames[i]
        log = logs.iloc[i]
        step = int(log["step"])
        mags = mag_by_step.get(step)
        if mags is None:
            mags = np.full(len(atoms), np.nan)
        pos = atoms.get_positions()
        centroid = 0.5 * (pos[O1] + pos[O2])
        rows.append(
            {
                "state": state.upper(),
                "model": model,
                "seed": seed,
                "step": step,
                "time_fs": float(log["time_fs"]),
                "temperature_K": float(log["temperature_K"]),
                "potential_energy_eV": float(log["potential_energy_eV"]),
                "max_abs_magmom": float(log["max_abs_magmom"]),
                "r_fe_o_min": float(min(atoms.get_distance(FE, O1, mic=True), atoms.get_distance(FE, O2, mic=True))),
                "r_fe_o2_centroid": float(np.linalg.norm(pos[FE] - centroid)),
                "r_oo": float(atoms.get_distance(O1, O2, mic=True)),
                "o2_axis_angle_deg": float(angle_o2_axis_to_fe_centroid(atoms)),
                "m_fe": float(mags[FE]),
                "m_o2_sum": float(mags[O1] + mags[O2]),
                "m_total": float(np.nansum(mags)),
            }
        )
    return pd.DataFrame(rows)


def load_aimd(state: str, max_time_fs: float) -> pd.DataFrame:
    cfg = STATE_CONFIG[state]
    raw = pd.read_csv(cfg["aimd_csv"])
    raw = raw[raw["time_fs"] <= max_time_fs].copy()
    raw = raw[raw["step"] % 20 == 0].copy()
    if state == "fm":
        start = int(cfg["dft_start_frame"])
        steps = raw["step"].to_numpy(dtype=int)
        frames = io.read(str(cfg["dft_outcar"]), index=f"{start + int(steps.min())}:{start + int(steps.max()) + 1}:20")
    else:
        steps = raw["step"].to_numpy(dtype=int)
        frames = io.read(str(cfg["dft_outcar"]), index=f"{int(steps.min())}:{int(steps.max()) + 1}:20")
    n = min(len(raw), len(frames))
    raw = raw.iloc[:n].reset_index(drop=True)
    frames = frames[:n]
    angle_vals = [float(angle_o2_axis_to_fe_centroid(atoms)) for atoms in frames]
    return pd.DataFrame(
        {
            "state": state.upper(),
            "model": "AIMD",
            "seed": "all",
            "time_fs": raw["time_fs"],
            "r_fe_o_min": np.minimum(raw["r_Fe_O1_dft"], raw["r_Fe_O2_dft"]),
            "r_fe_o2_centroid": raw["r_Fe_O2_centroid_dft"],
            "r_oo": raw["r_OO_dft"],
            "m_fe": raw["atom_55_magmom_dft"],
            "m_o2_sum": raw["o_sum_magmom_dft"],
            "o2_axis_angle_deg": angle_vals,
        }
    )


def select_top3(aimd: pd.DataFrame, runs: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    aimd_stats = {}
    for metric in METRICS:
        vals = aimd[metric].to_numpy(dtype=float)
        scale = float(np.nanstd(vals, ddof=1))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = max(abs(float(np.nanmean(vals))), 1.0)
        aimd_stats[metric] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)),
            "scale": scale,
        }

    rows = []
    for df in runs:
        seed = int(df["seed"].iloc[0])
        model = str(df["model"].iloc[0])
        row = {"model": model, "seed": seed}
        scores = []
        for metric in METRICS:
            mean_bias = float(np.nanmean(df[metric]) - aimd_stats[metric]["mean"])
            std_bias = float(np.nanstd(df[metric], ddof=1) - aimd_stats[metric]["std"])
            score = abs(mean_bias) / aimd_stats[metric]["scale"] + 0.25 * abs(std_bias) / aimd_stats[metric]["scale"]
            row[f"{metric}_mean_bias"] = mean_bias
            row[f"{metric}_std_bias"] = std_bias
            row[f"{metric}_score"] = score
            scores.append(score)
        row["score"] = float(np.mean(scores))
        rows.append(row)

    rank = pd.DataFrame(rows).sort_values(["score", "seed"]).reset_index(drop=True)
    selected_seeds = rank.head(3)["seed"].astype(int).tolist()
    selected = [df for df in runs if int(df["seed"].iloc[0]) in selected_seeds]
    return selected, rank


def summarize_series(aimd: pd.DataFrame, selected_by_model: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        vals = aimd[metric].to_numpy(dtype=float)
        rows.append(
            {
                "series": "AIMD",
                "seed": "all",
                "metric": metric,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }
        )
        for model, selected in selected_by_model.items():
            for df in selected:
                vals = df[metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "series": model,
                        "seed": int(df["seed"].iloc[0]),
                        "metric": metric,
                        "mean": float(np.nanmean(vals)),
                        "std": float(np.nanstd(vals, ddof=1)),
                        "min": float(np.nanmin(vals)),
                        "max": float(np.nanmax(vals)),
                    }
                )
    return pd.DataFrame(rows)


def plot_state(state: str, aimd: pd.DataFrame, selected_by_model: dict[str, list[pd.DataFrame]]) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.05))
    panels = [
        ("r_fe_o_min", np.linspace(1.5, 7.5, 70), r"Fe-O$_{\min}$ ($\AA$)"),
        ("r_fe_o2_centroid", np.linspace(1.5, 8.0, 70), r"Fe-O$_2$ centroid ($\AA$)"),
        ("r_oo", np.linspace(1.1, 2.1, 50), r"O-O ($\AA$)"),
        ("o2_axis_angle_deg", np.linspace(0, 90, 50), r"O$_2$ axis angle (deg)"),
        ("m_fe", np.linspace(-4.0, 4.0, 60), r"Fe moment ($\mu_B$)"),
        ("m_o2_sum", np.linspace(-3.0, 3.0, 60), r"O$_2$ total moment ($\mu_B$)"),
    ]
    colors = {"previous": "#276FBF", "rule": "#C45A1D"}
    labels = {"previous": "previous ML-MD top-3", "rule": "rule ML-MD top-3"}

    for ax, (metric, bins, xlabel) in zip(axes.ravel(), panels):
        x_aimd, y_aimd = hist_density(aimd[metric].to_numpy(dtype=float), bins)
        ax.plot(x_aimd, y_aimd, color="#202020", lw=2.15, label="AIMD")

        for model in ["previous", "rule"]:
            densities = []
            x_seed = None
            for df in selected_by_model[model]:
                x_seed, y_seed = hist_density(df[metric].to_numpy(dtype=float), bins)
                densities.append(y_seed)
            density_arr = np.vstack(densities)
            mean = density_arr.mean(axis=0)
            std = density_arr.std(axis=0)
            color = colors[model]
            ax.plot(x_seed, mean, color=color, lw=1.75, label=labels[model])
            ax.fill_between(x_seed, np.maximum(mean - std, 0.0), mean + std, color=color, alpha=0.16, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18, lw=0.6)

    for idx, ax in enumerate(axes.ravel()):
        ax.text(0.02, 0.96, chr(ord("a") + idx), transform=ax.transAxes, va="top", ha="left", fontweight="bold")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, legend_labels):
        dedup.setdefault(label, handle)
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.028),
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.1,
    )
    title = f"FeN4-O2 {STATE_CONFIG[state]['title']}: AIMD vs AIMD-only bounded MACE-FiLM ML-MD"
    fig.suptitle(title, y=0.955, fontsize=10.0)
    fig.tight_layout(rect=(0.010, 0.125, 0.995, 0.920), pad=0.35, w_pad=0.35, h_pad=0.65)
    fig.subplots_adjust(left=0.070, right=0.990, bottom=0.175, top=0.890, wspace=0.23, hspace=0.38)

    out = OUT_DIR / f"{STATE_CONFIG[state]['out_stem']}.png"
    fig.savefig(out, dpi=500, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def process_state(state: str, max_time_fs: float, rebuild_cache: bool) -> None:
    cfg = STATE_CONFIG[state]
    cache_path = CACHE_DIR / f"{cfg['out_stem']}_t{int(max_time_fs)}.pkl"
    signature = {
        "state": state,
        "max_time_fs": max_time_fs,
        "previous_root": str(cfg["previous_root"]),
        "rule_root": str(cfg["rule_root"]),
        "aimd_csv": str(cfg["aimd_csv"]),
        "dft_outcar": str(cfg["dft_outcar"]),
    }
    if cache_path.exists() and not rebuild_cache:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("signature") == signature:
            print(f"Using cached processed data: {cache_path}")
            plot_state(state, payload["aimd"], payload["selected_by_model"])
            payload["summary"].to_csv(OUT_DIR / f"{cfg['out_stem']}_summary.csv", index=False)
            payload["rank"].to_csv(OUT_DIR / f"{cfg['out_stem']}_seed_selection_scores.csv", index=False)
            return

    aimd = load_aimd(state, max_time_fs)
    selected_by_model: dict[str, list[pd.DataFrame]] = {}
    rank_frames = []
    for model in ["previous", "rule"]:
        runs = [
            load_seed(cfg[f"{model}_root"], str(cfg[f"{model}_prefix"]), state, model, seed, max_time_fs)
            for seed in SEEDS
        ]
        selected, rank = select_top3(aimd, runs)
        selected_by_model[model] = selected
        rank["state"] = state.upper()
        rank_frames.append(rank)

    rank_all = pd.concat(rank_frames, ignore_index=True)
    summary = summarize_series(aimd, selected_by_model)
    plot_state(state, aimd, selected_by_model)
    summary.to_csv(OUT_DIR / f"{cfg['out_stem']}_summary.csv", index=False)
    rank_all.to_csv(OUT_DIR / f"{cfg['out_stem']}_seed_selection_scores.csv", index=False)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "signature": signature,
                "aimd": aimd,
                "selected_by_model": selected_by_model,
                "summary": summary,
                "rank": rank_all,
            },
            handle,
        )
    print(f"Wrote processed-data cache: {cache_path}")
    print(f"Wrote figure: {OUT_DIR / (cfg['out_stem'] + '.png')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=["afm", "fm", "both"], default="both")
    parser.add_argument("--max-time-fs", type=float, default=10020.0)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    states = ["afm", "fm"] if args.state == "both" else [args.state]
    for state in states:
        process_state(state, args.max_time_fs, args.rebuild_cache)


if __name__ == "__main__":
    main()