#!/usr/bin/env python3
"""Plot FeN4 DFT-vs-ML PES and magnetic moments for opt-vaspinit fine-tuned scan."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ocp-mag")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/wuzhihong/software/ocp_mag")
WORK = ROOT / "work_fen4_escn_zbl_mdtest/mace_film"
ML_CSV = (
    WORK
    / "pes_outputs/fen4_mace_aimd10_opt_pes_zbl_bounded_elem8_opt_vaspinit_ft_v1_aimdtemplate_optstep2magmom"
    / "fe_o2_scan_fen4_mace_bounded_elem8_opt_vaspinit_ft_aimdtemplate_optstep2magmom.csv"
)
DFT_CSV = ROOT / "paper_jctc/organized_artifacts/FeN4_O2/data/dft_pes_reference.csv"
OUT_DIR = ROOT / "paper_jctc/figures/si/fen4_opt_vaspinit_ft"
OUT_STEM = "fen4_opt_vaspinit_ft_dft_ml_pes_magmom_overlay"


def load_overlay_data() -> pd.DataFrame:
    dft = pd.read_csv(DFT_CSV)
    dft = dft.rename(
        columns={
            "energy_aligned_d4": "rel_energy",
            "fe_mag": "fe_magmom",
            "o_sum_mag": "o2_sum_magmom",
        }
    )
    dft["state"] = dft["state"].map({"S0": "AFM", "S2": "FM"})
    dft["source"] = "DFT"
    dft = dft[
        ["source", "state", "distance", "rel_energy", "fe_magmom", "o2_sum_magmom", "branch_bad", "is_anomalous"]
    ]

    ml = pd.read_csv(ML_CSV)
    ml = ml.rename(
        columns={
            "model_label": "state",
            "scan_distance": "distance",
            "adsorption_energy_relative_eV": "rel_energy",
        }
    )
    ml["source"] = "ML"
    ml["o2_sum_magmom"] = ml["o1_magmom"] + ml["o2_magmom"]
    ml["branch_bad"] = False
    ml["is_anomalous"] = False
    ml = ml[
        ["source", "state", "distance", "rel_energy", "fe_magmom", "o2_sum_magmom", "branch_bad", "is_anomalous"]
    ]

    out = pd.concat([dft, ml], ignore_index=True)
    out["distance"] = out["distance"].round(2)
    return out.sort_values(["state", "source", "distance"])


def write_error_summary(data: pd.DataFrame) -> None:
    dft = data[data["source"] == "DFT"].copy()
    ml = data[data["source"] == "ML"].copy()
    merged = pd.merge(
        dft,
        ml,
        on=["state", "distance"],
        suffixes=("_dft", "_ml"),
        how="inner",
    )
    for metric in ("rel_energy", "fe_magmom", "o2_sum_magmom"):
        merged[f"{metric}_err"] = merged[f"{metric}_ml"] - merged[f"{metric}_dft"]
    summary = []
    for state, group in merged.groupby("state"):
        row = {"state": state, "count": len(group)}
        for metric in ("rel_energy", "fe_magmom", "o2_sum_magmom"):
            err = group[f"{metric}_err"]
            row[f"{metric}_mae"] = err.abs().mean()
            row[f"{metric}_max_abs"] = err.abs().max()
            row[f"{metric}_mean"] = err.mean()
        summary.append(row)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / f"{OUT_STEM}_error_points.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT_DIR / f"{OUT_STEM}_error_summary.csv", index=False)


def plot(data: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 18,
            "axes.labelsize": 23,
            "axes.titlesize": 23,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {"AFM": "#2F5F93", "FM": "#C45D38"}
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 10.8),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"hspace": 0.18},
    )

    panel_specs = [
        ("rel_energy", r"$E - E_{4.0\ \AA}$ (eV)", "PES", (-0.45, 0.30)),
        ("fe_magmom", r"Fe moment ($\mu_B$)", "Fe magnetic moment", None),
        ("o2_sum_magmom", r"O$_2$ total moment ($\mu_B$)", "O$_2$ magnetic moment", None),
    ]

    legend_handles = []
    legend_labels = []
    for ax, (metric, ylabel, title, ylim) in zip(axes, panel_specs):
        for state in ("AFM", "FM"):
            color = colors[state]
            dft = data[(data["state"] == state) & (data["source"] == "DFT")].sort_values("distance")
            ml = data[(data["state"] == state) & (data["source"] == "ML")].sort_values("distance")
            line_dft = ax.plot(dft["distance"], dft[metric], "-o", color=color, lw=2.8, ms=7.0, label=f"{state} DFT")[0]
            line_ml = ax.plot(
                ml["distance"],
                ml[metric],
                "--s",
                color=color,
                lw=2.8,
                ms=6.4,
                markerfacecolor="white",
                markeredgewidth=1.8,
                label=f"{state} ML",
            )[0]
            if ax is axes[0]:
                legend_handles.extend([line_dft, line_ml])
                legend_labels.extend([f"{state} DFT", f"{state} ML"])

        ax.set_title(title, pad=6)
        ax.set_ylabel(ylabel)
        ax.set_xlim(1.75, 4.05)
        ax.set_xticks([1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.0])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, color="#D5D5D5", alpha=0.65, linewidth=1.0)
        ax.tick_params(direction="in", length=6.0, width=1.6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel(r"Fe-O distance ($\AA$)")
    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(-0.085, 1.04, label, transform=ax.transAxes, fontsize=28, fontweight="bold", va="bottom", ha="left")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=2.2,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.965, bottom=0.12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{OUT_STEM}.png"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    csv = OUT_DIR / f"{OUT_STEM}_data.csv"
    data.to_csv(csv, index=False)
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv}")


def main() -> None:
    data = load_overlay_data()
    plot(data)
    write_error_summary(data)
    summary = pd.read_csv(OUT_DIR / f"{OUT_STEM}_error_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
