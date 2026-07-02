#!/usr/bin/env python3
"""Plot latest FeN4-O2 bounded-FiLM PES and moments against DFT."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ocp-mag")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[1]
FEN4_EXAMPLE = RELEASE_ROOT / "examples" / "fen4"
DEFAULT_DFT = FEN4_EXAMPLE / "figure_source_data" / "dft_pes_reference.csv"
DEFAULT_ML = (
    FEN4_EXAMPLE
    / "figure_source_data"
    / "fe_o2_scan_mace_aimd10_opt_pes_zbl_bounded_elem8.csv"
)
DEFAULT_FIG_DIR = FEN4_EXAMPLE / "figures"
DEFAULT_CACHE_DIR = DEFAULT_FIG_DIR / "_cache"

STATE_MAP = {"S0": "AFM", "S2": "FM", "AFM": "AFM", "FM": "FM"}
COLORS = {"AFM": "#2F5D8C", "FM": "#B85C38"}
PANEL_LABELS = tuple("abcdefghijklmnopqrstuvwxyz")


def source_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.labelsize": 9.2,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.85,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def load_dft(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"state", "distance", "energy_aligned_d4", "fe_mag", "o_sum_mag"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing DFT columns in {path}: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "source": "DFT",
            "state": df["state"].map(STATE_MAP).fillna(df["state"]),
            "distance": df["distance"].astype(float),
            "rel_energy": df["energy_aligned_d4"].astype(float),
            "fe_magmom": df["fe_mag"].astype(float),
            "o2_sum_magmom": df["o_sum_mag"].astype(float),
        }
    )
    return out.sort_values(["state", "distance"]).reset_index(drop=True)


def load_ml(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "model_label",
        "scan_distance",
        "total_energy_eV",
        "fe_magmom",
        "o1_magmom",
        "o2_magmom",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing ML columns in {path}: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "source": "ML",
            "state": df["model_label"].map(STATE_MAP).fillna(df["model_label"]),
            "distance": df["scan_distance"].astype(float),
            "energy": df["total_energy_eV"].astype(float),
            "fe_magmom": df["fe_magmom"].astype(float),
            "o2_sum_magmom": df["o1_magmom"].astype(float) + df["o2_magmom"].astype(float),
        }
    )
    refs: dict[str, float] = {}
    for state, group in out.groupby("state"):
        idx = (group["distance"] - 4.0).abs().idxmin()
        refs[state] = float(group.loc[idx, "energy"])
    out["rel_energy"] = [row.energy - refs[row.state] for row in out.itertuples()]
    return out.drop(columns=["energy"]).sort_values(["state", "distance"]).reset_index(drop=True)


def load_or_build_cache(cache_path: Path, dft_csv: Path, ml_csv: Path, rebuild: bool) -> pd.DataFrame:
    sources = {"dft_csv": source_signature(dft_csv), "ml_csv": source_signature(ml_csv)}
    if cache_path.exists() and not rebuild:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("sources") == sources:
            print(f"Using cached plotting data: {cache_path}")
            return payload["data"]

    dft = load_dft(dft_csv)
    ml = load_ml(ml_csv)
    data = pd.concat([dft, ml], ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump({"sources": sources, "data": data}, handle)
    print(f"Wrote plotting cache: {cache_path}")
    return data


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.085,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )


def plot_overlay(data: pd.DataFrame, output: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(4.45, 6.35),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    panels = [
        ("rel_energy", r"$E - E_{4.0\ \mathrm{\AA}}$ (eV)", "PES scan"),
        ("fe_magmom", r"Fe moment ($\mu_{\mathrm{B}}$)", "Fe magnetic moment"),
        ("o2_sum_magmom", r"O$_2$ total moment ($\mu_{\mathrm{B}}$)", "O$_2$ magnetic moment"),
    ]
    source_style = {
        "DFT": {"linestyle": "-", "marker": "o", "mfc": None, "lw": 1.55, "ms": 3.9},
        "ML": {"linestyle": "--", "marker": "s", "mfc": "white", "lw": 1.35, "ms": 3.5},
    }

    for ax, (metric, ylabel, title), panel_label in zip(axes, panels, PANEL_LABELS):
        for state in ["AFM", "FM"]:
            for source in ["DFT", "ML"]:
                group = data[(data["state"] == state) & (data["source"] == source)].sort_values("distance")
                if group.empty:
                    continue
                style = source_style[source]
                ax.plot(
                    group["distance"],
                    group[metric],
                    color=COLORS[state],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=style["ms"],
                    markerfacecolor=COLORS[state] if style["mfc"] is None else style["mfc"],
                    markeredgecolor=COLORS[state],
                    markeredgewidth=0.75,
                    linewidth=style["lw"],
                    label=f"{state} {source}",
                    zorder=3 if source == "DFT" else 2,
                )
        ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.8, zorder=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=2.0)
        ax.grid(True, color="0.90", linewidth=0.55)
        ax.set_xlim(1.75, 4.05)
        ax.tick_params(direction="in", top=False, right=False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        add_panel_label(ax, panel_label)

    axes[0].set_ylim(-0.40, 0.32)
    axes[0].set_yticks(np.arange(-0.4, 0.26, 0.2))
    axes[1].set_ylim(-2.4, 2.8)
    axes[1].set_yticks(np.arange(-2.0, 2.1, 2.0))
    axes[2].set_ylim(-0.35, 1.55)
    axes[2].set_yticks(np.arange(0.0, 1.6, 0.5))
    axes[-1].set_xlabel(r"Fe-O distance ($\mathrm{\AA}$)")
    axes[-1].set_xticks([1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.0])

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.006),
        columnspacing=1.0,
        handlelength=1.7,
        handletextpad=0.45,
    )
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.965, bottom=0.118)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def write_merged_csv(data: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data.sort_values(["state", "source", "distance"]).to_csv(output, index=False)
    print(f"Wrote merged plotting data: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dft-csv", type=Path, default=DEFAULT_DFT)
    parser.add_argument("--ml-csv", type=Path, default=DEFAULT_ML)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.cache_dir / "fen4_bounded_film_dft_ml_pes_magmom_overlay.pkl"
    data = load_or_build_cache(cache_path, args.dft_csv, args.ml_csv, args.rebuild_cache)
    out = args.figure_dir / "fen4_bounded_film_dft_ml_pes_magmom_overlay.png"
    plot_overlay(data, out)
    write_merged_csv(data, args.cache_dir / "fen4_bounded_film_dft_ml_pes_magmom_overlay_data.csv")
    print(f"Wrote figure: {out}")
    print(f"Wrote figure: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
