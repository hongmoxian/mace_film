#!/usr/bin/env python3
"""Compact FeN4 AFM/FM MD comparison with summary error panel and representative densities."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ocp-mag")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from ase import io


ROOT = Path("/home/wuzhihong/software/ocp_mag")
WORK = ROOT / "work_fen4_escn_zbl_mdtest" / "mace_film"
OUT_DIR = WORK / "paper_figures" / "fen4_afm_fm_previous_rule_selected_statistics"
CACHE_DIR = OUT_DIR / "_cache"
SEEDS = [101, 102, 103, 104, 105]

FE = 54
O1 = 59
O2 = 60

STATE_CONFIG = {
    "AFM": {
        "aimd_csv": ROOT / "paper_jctc/organized_artifacts/FeN4_O2/data/figure2_previous_S0_merged.csv",
        "dft_outcar": Path("/home/wuzhihong/vasp/FeN4/S1/s1-o2-0/md/50k/OUTCAR"),
        "dft_start_frame": 0,
        "previous_root": WORK / "md_outputs" / "fen4_mace_film_aimdonly_zbl_bounded_elem8_previous_afm_secondframe_10ps_5seed",
        "previous_prefix": "afm_secondframe_seed_",
        "rule_root": WORK / "md_outputs" / "fen4_mace_film_aimdonly_zbl_bounded_elem8_rule_afm_secondframe_10ps_5seed",
        "rule_prefix": "afm_secondframe_seed_",
    },
    "FM": {
        "aimd_csv": ROOT / "paper_jctc/organized_artifacts/FeN4_O2/data/figure2_previous_S2_merged.csv",
        "dft_outcar": Path("/home/wuzhihong/vasp/FeN4/S1/s1-o2-2/md/50k/OUTCAR"),
        "dft_start_frame": 5401,
        "previous_root": WORK / "md_outputs" / "fen4_mace_film_aimdonly_zbl_bounded_elem8_previous_fm_frame5401_10ps_5seed",
        "previous_prefix": "fm_frame5401_seed_",
        "rule_root": WORK / "md_outputs" / "fen4_mace_film_aimdonly_zbl_bounded_elem8_rule_fm_frame5401_10ps_5seed",
        "rule_prefix": "fm_frame5401_seed_",
    },
}

SUMMARY_METRICS = [
    "r_fe_o_min",
    "r_fe_o2_centroid",
    "r_oo",
    "o2_axis_angle_deg",
    "m_fe",
    "m_o2_sum",
]

SUMMARY_LABELS = {
    "r_fe_o_min": r"Fe-O$_\mathrm{min}$",
    "r_fe_o2_centroid": r"Fe-O$_2$",
    "r_oo": "O-O",
    "o2_axis_angle_deg": r"O$_2$ angle",
    "m_fe": "Fe moment",
    "m_o2_sum": r"O$_2$ moment",
}

MAIN_DENSITY_PANELS = [
    ("r_fe_o_min", np.linspace(1.5, 5.8, 58), r"Fe-O$_\mathrm{min}$ ($\AA$)"),
    ("r_oo", np.linspace(1.1, 2.1, 50), r"O-O ($\AA$)"),
    ("m_fe", np.linspace(-4.0, 4.0, 60), r"Fe moment ($\mu_B$)"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
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


def load_seed(run_root: Path, prefix: str, model: str, seed: int, max_time_fs: float) -> pd.DataFrame:
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
                "model": model,
                "seed": seed,
                "step": step,
                "time_fs": float(log["time_fs"]),
                "r_fe_o_min": float(min(atoms.get_distance(FE, O1, mic=True), atoms.get_distance(FE, O2, mic=True))),
                "r_fe_o2_centroid": float(np.linalg.norm(pos[FE] - centroid)),
                "r_oo": float(atoms.get_distance(O1, O2, mic=True)),
                "o2_axis_angle_deg": float(angle_o2_axis_to_fe_centroid(atoms)),
                "m_fe": float(mags[FE]),
                "m_o2_sum": float(mags[O1] + mags[O2]),
            }
        )
    return pd.DataFrame(rows)


def load_aimd(state: str, max_time_fs: float) -> pd.DataFrame:
    cfg = STATE_CONFIG[state]
    raw = pd.read_csv(cfg["aimd_csv"])
    raw = raw[raw["time_fs"] <= max_time_fs].copy()
    raw = raw[raw["step"] % 20 == 0].copy()
    start = int(cfg["dft_start_frame"])
    steps = raw["step"].to_numpy(dtype=int)
    frames = io.read(
        str(cfg["dft_outcar"]),
        index=f"{start + int(steps.min())}:{start + int(steps.max()) + 1}:20",
    )
    n = min(len(raw), len(frames))
    raw = raw.iloc[:n].reset_index(drop=True)
    frames = frames[:n]
    angle_vals = [float(angle_o2_axis_to_fe_centroid(atoms)) for atoms in frames]
    return pd.DataFrame(
        {
            "time_fs": raw["time_fs"],
            "r_fe_o_min": np.minimum(raw["r_Fe_O1_dft"], raw["r_Fe_O2_dft"]),
            "r_fe_o2_centroid": raw["r_Fe_O2_centroid_dft"],
            "r_oo": raw["r_OO_dft"],
            "m_fe": raw["atom_55_magmom_dft"],
            "m_o2_sum": raw["o_sum_magmom_dft"],
            "o2_axis_angle_deg": angle_vals,
        }
    )


def rank_runs(aimd: pd.DataFrame, runs: list[pd.DataFrame], selection: str, top_k: int = 3):
    aimd_stats = {}
    for metric in SUMMARY_METRICS:
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
        row = {"seed": seed}
        scores = []
        for metric in SUMMARY_METRICS:
            mean_bias = float(np.nanmean(df[metric]) - aimd_stats[metric]["mean"])
            std_bias = float(np.nanstd(df[metric], ddof=1) - aimd_stats[metric]["std"])
            score = abs(mean_bias) / aimd_stats[metric]["scale"] + 0.25 * abs(std_bias) / aimd_stats[metric]["scale"]
            row[f"{metric}_mean_bias"] = mean_bias
            row[f"{metric}_std_bias"] = std_bias
            row[f"{metric}_score"] = score
            scores.append(score)
        row["score"] = float(np.mean(scores))
        rows.append(row)

    ranks = pd.DataFrame(rows).sort_values(["score", "seed"]).reset_index(drop=True)
    if selection == "best":
        selected_seeds = set(ranks.head(top_k)["seed"].astype(int).tolist())
    elif selection == "worst":
        selected_seeds = set(ranks.tail(top_k)["seed"].astype(int).tolist())
    else:
        raise ValueError(selection)
    selected = [df for df in runs if int(df["seed"].iloc[0]) in selected_seeds]
    selected.sort(key=lambda df: int(df["seed"].iloc[0]))
    ranks["selected"] = ranks["seed"].isin(selected_seeds)
    return selected, ranks


def normalized_wasserstein_from_hist(reference: pd.Series, sample: pd.Series, bins: np.ndarray) -> float:
    ref_values = np.asarray(reference, dtype=float)
    sample_values = np.asarray(sample, dtype=float)
    ref_values = ref_values[np.isfinite(ref_values)]
    sample_values = sample_values[np.isfinite(sample_values)]
    ref_hist, edges = np.histogram(ref_values, bins=bins, density=False)
    sample_hist, _ = np.histogram(sample_values, bins=bins, density=False)
    if ref_hist.sum() == 0 or sample_hist.sum() == 0:
        return np.nan
    ref_cdf = np.cumsum(ref_hist / ref_hist.sum())
    sample_cdf = np.cumsum(sample_hist / sample_hist.sum())
    widths = np.diff(edges)
    w1 = float(np.sum(np.abs(ref_cdf - sample_cdf) * widths))
    scale = float(np.std(ref_values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = max(abs(float(np.mean(ref_values))), 1.0)
    return w1 / scale


def summarize_error(case_payloads: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for state, (_, previous_payload, rule_payload) in case_payloads.items():
        for metric, bins, _ in [
            ("r_fe_o_min", np.linspace(1.5, 5.8, 58), r"Fe-O$_\mathrm{min}$"),
            ("r_fe_o2_centroid", np.linspace(1.5, 6.5, 60), r"Fe-O$_2$"),
            ("r_oo", np.linspace(1.1, 2.1, 50), "O-O"),
            ("o2_axis_angle_deg", np.linspace(0, 90, 50), r"O$_2$ angle"),
            ("m_fe", np.linspace(-4.0, 4.0, 60), "Fe moment"),
            ("m_o2_sum", np.linspace(-3.0, 3.0, 60), r"O$_2$ moment"),
        ]:
            for model_label, runs in (
                ("Previous-init. ML-MD", previous_payload["selected_runs"]),
                ("Rule-init. ML-MD", rule_payload["selected_runs"]),
            ):
                distances = [
                    normalized_wasserstein_from_hist(previous_payload["aimd"][metric], run[metric], bins)
                    for run in runs
                ]
                rows.append(
                    {
                        "state": state,
                        "model": model_label,
                        "metric": metric,
                        "metric_label": SUMMARY_LABELS[metric],
                        "normalized_wasserstein_mean": float(np.nanmean(distances)),
                        "normalized_wasserstein_std": float(np.nanstd(distances)),
                        "n_runs": int(np.sum(np.isfinite(distances))),
                    }
                )
    return pd.DataFrame(rows)


def plot_density_panel(
    ax,
    aimd: pd.DataFrame,
    previous_runs: list[pd.DataFrame],
    rule_runs: list[pd.DataFrame],
    metric: str,
    bins: np.ndarray,
    xlabel: str,
    show_xlabel: bool = True,
) -> None:
    colors = {"aimd": "#202020", "previous": "#2F5D8C", "rule": "#C45A1D"}
    x_aimd, y_aimd = hist_density(aimd[metric].to_numpy(dtype=float), bins)
    ax.plot(x_aimd, y_aimd, color=colors["aimd"], lw=2.2, label="Reference AIMD")

    for label, runs in (("previous", previous_runs), ("rule", rule_runs)):
        densities = []
        x_seed = None
        for df in runs:
            x_seed, y_seed = hist_density(df[metric].to_numpy(dtype=float), bins)
            densities.append(y_seed)
        arr = np.vstack(densities)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(x_seed, mean, color=colors[label], lw=1.9, label=f"{label.capitalize()}-init. ML-MD mean")
        ax.fill_between(
            x_seed,
            np.maximum(mean - std, 0.0),
            mean + std,
            color=colors[label],
            alpha=0.16,
            linewidth=0,
            label=f"{label.capitalize()}-init. ML-MD std",
        )
    if show_xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.18, lw=0.6)


def process_state(state: str, max_time_fs: float = 10020.0) -> dict[str, object]:
    cfg = STATE_CONFIG[state]
    aimd = load_aimd(state, max_time_fs)
    previous_runs = [
        load_seed(cfg["previous_root"], cfg["previous_prefix"], "previous", seed, max_time_fs)
        for seed in SEEDS
    ]
    rule_runs = [
        load_seed(cfg["rule_root"], cfg["rule_prefix"], "rule", seed, max_time_fs)
        for seed in SEEDS
    ]
    previous_selected, previous_ranks = rank_runs(aimd, previous_runs, "best")
    rule_selected, rule_ranks = rank_runs(aimd, rule_runs, "worst")
    return {
        "aimd": aimd,
        "previous_selected": previous_selected,
        "rule_selected": rule_selected,
        "previous_ranks": previous_ranks,
        "rule_ranks": rule_ranks,
    }


def plot_main_figure(case_payloads: dict[str, dict[str, object]]) -> Path:
    setup_style()
    fig = plt.figure(figsize=(7.35, 5.85))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.080,
        right=0.990,
        bottom=0.150,
        top=0.965,
        height_ratios=[1.0, 0.85, 0.85],
        hspace=0.50,
        wspace=0.34,
    )
    summary_ax = fig.add_subplot(grid[0, :])
    error_df = summarize_error({
        state: (payload["aimd"], {"aimd": payload["aimd"], "selected_runs": payload["previous_selected"]}, {"aimd": payload["aimd"], "selected_runs": payload["rule_selected"]})
        for state, payload in case_payloads.items()
    })

    colors = {"Previous-init. ML-MD": "#2F5D8C", "Rule-init. ML-MD": "#C45A1D"}
    markers = {"AFM": "o", "FM": "s"}
    offsets = {
        ("AFM", "Previous-init. ML-MD"): -0.18,
        ("AFM", "Rule-init. ML-MD"): -0.06,
        ("FM", "Previous-init. ML-MD"): 0.06,
        ("FM", "Rule-init. ML-MD"): 0.18,
    }
    x = np.arange(len(SUMMARY_METRICS), dtype=float)
    for state in ("AFM", "FM"):
        for model in ("Previous-init. ML-MD", "Rule-init. ML-MD"):
            sub = error_df[(error_df["state"] == state) & (error_df["model"] == model)]
            sub = sub.set_index("metric").loc[SUMMARY_METRICS]
            summary_ax.errorbar(
                x + offsets[(state, model)],
                sub["normalized_wasserstein_mean"].to_numpy(),
                yerr=sub["normalized_wasserstein_std"].to_numpy(),
                fmt=markers[state],
                ms=4.0,
                lw=1.1,
                capsize=2.0,
                color=colors[model],
                markerfacecolor=colors[model] if state == "AFM" else "white",
                markeredgewidth=1.1,
                label=f"{state}, {model}",
            )
    summary_ax.set_ylabel("Distribution error vs AIMD\n(normalized W1)")
    summary_ax.set_xticks(x)
    summary_ax.set_xticklabels([SUMMARY_LABELS[m] for m in SUMMARY_METRICS], rotation=20, ha="right")
    summary_ax.grid(axis="y", alpha=0.22, lw=0.6)
    summary_ax.text(0.01, 0.95, "a", transform=summary_ax.transAxes, va="top", ha="left", fontweight="bold")

    prev_afm_proxy = Line2D(
        [],
        [],
        color="#2F5D8C",
        marker="o",
        linestyle="None",
        markersize=4.5,
        label="Previous-init. AFM",
    )
    prev_fm_proxy = Line2D(
        [],
        [],
        color="#2F5D8C",
        marker="s",
        linestyle="None",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="Previous-init. FM",
    )
    rule_afm_proxy = Line2D(
        [],
        [],
        color="#C45A1D",
        marker="o",
        linestyle="None",
        markersize=4.5,
        label="Rule-init. AFM",
    )
    rule_fm_proxy = Line2D(
        [],
        [],
        color="#C45A1D",
        marker="s",
        linestyle="None",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="Rule-init. FM",
    )
    summary_ax.legend(
        handles=[prev_afm_proxy, prev_fm_proxy, rule_afm_proxy, rule_fm_proxy],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    density_panels = MAIN_DENSITY_PANELS
    letters = iter("bcdefg")
    for col, (metric, bins, xlabel) in enumerate(density_panels):
        ax_afm = fig.add_subplot(grid[1, col])
        ax_fm = fig.add_subplot(grid[2, col], sharex=ax_afm)
        for ax, state, show_xlabel in ((ax_afm, "AFM", False), (ax_fm, "FM", True)):
            payload = case_payloads[state]
            plot_density_panel(
                ax,
                payload["aimd"],
                payload["previous_selected"],
                payload["rule_selected"],
                metric,
                bins,
                xlabel,
                show_xlabel=show_xlabel,
            )
            ax.text(0.03, 0.92, next(letters), transform=ax.transAxes, va="top", ha="left", fontweight="bold")
            ax.text(0.97, 0.90, state, transform=ax.transAxes, va="top", ha="right", fontsize=7.5, fontweight="bold")
        ax_afm.set_title(xlabel, pad=3)

    density_handles, density_labels = fig.axes[1].get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(density_handles, density_labels):
        dedup.setdefault(label, handle)
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.0,
    )
    out = OUT_DIR / "fen4_afm_fm_previous_rule_selected_statistics.png"
    fig.savefig(out, dpi=450, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out, error_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    case_payloads = {state: process_state(state) for state in ("AFM", "FM")}

    for state, payload in case_payloads.items():
        prev_seeds = [int(df["seed"].iloc[0]) for df in payload["previous_selected"]]
        rule_seeds = [int(df["seed"].iloc[0]) for df in payload["rule_selected"]]
        print(f"{state}: previous-initialized seeds={prev_seeds}; rule-initialized seeds={rule_seeds}")

    out, error_df = plot_main_figure(case_payloads)
    error_df.to_csv(OUT_DIR / "fen4_afm_fm_previous_rule_selected_normalized_wasserstein.csv", index=False)
    print(f"Wrote figure: {out}")
    print(f"Wrote figure: {out.with_suffix('.pdf')}")
    print(f"Wrote CSV: {OUT_DIR / 'fen4_afm_fm_previous_rule_selected_normalized_wasserstein.csv'}")


if __name__ == "__main__":
    main()
