#!/usr/bin/env python3
"""Plot FeN4-O2 DFT and MACE-FiLM ML PES on the same axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/wuzhihong/software/ocp_mag")
DFT_CSV = ROOT / "paper_jctc/organized_artifacts/FeN4_O2/data/dft_pes_reference.csv"
ML_CSV = (
    ROOT
    / "work_fen4_escn_zbl_mdtest/mace_film/pes_outputs/"
    / "fen4_mace_aimd10_opt_pes_zbl_previous_gemoc_params/"
    / "fe_o2_scan_mace_aimd10_opt_pes_zbl_previous.csv"
)
OUT_DIR = (
    ROOT
    / "work_fen4_escn_zbl_mdtest/mace_film/pes_outputs/"
    / "fen4_mace_aimd10_opt_pes_zbl_previous_gemoc_params"
)


def load_data() -> pd.DataFrame:
    dft = pd.read_csv(DFT_CSV)
    ml = pd.read_csv(ML_CSV)

    dft = dft.rename(
        columns={
            "state": "state_raw",
            "distance": "distance",
            "energy_aligned_d4": "dft_e_rel",
        }
    )
    dft["state"] = dft["state_raw"].map({"S0": "AFM", "S2": "FM"})
    dft = dft[["state", "distance", "dft_e_rel", "branch_bad", "is_anomalous"]]

    ml = ml.rename(
        columns={
            "model_label": "state",
            "scan_distance": "distance",
            "adsorption_energy_relative_eV": "ml_e_rel",
        }
    )
    ml = ml[["state", "distance", "ml_e_rel"]]

    merged = pd.merge(dft, ml, on=["state", "distance"], how="outer").sort_values(
        ["state", "distance"]
    )
    merged["ml_minus_dft"] = merged["ml_e_rel"] - merged["dft_e_rel"]
    return merged


def plot(merged: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {"AFM": "#2F5D8C", "FM": "#B85C38"}
    labels = {"AFM": "AFM", "FM": "FM"}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.9),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28},
        constrained_layout=False,
    )

    ax = axes[0]
    err_ax = axes[1]
    for state in ["AFM", "FM"]:
        g = merged[merged["state"] == state].sort_values("distance")
        color = colors[state]
        ax.plot(
            g["distance"],
            g["dft_e_rel"],
            "-o",
            color=color,
            lw=1.8,
            ms=4.0,
            label=f"DFT {labels[state]}",
        )
        ax.plot(
            g["distance"],
            g["ml_e_rel"],
            "--s",
            color=color,
            lw=1.5,
            ms=3.5,
            markerfacecolor="white",
            label=f"ML {labels[state]}",
        )

        ge = g.dropna(subset=["ml_minus_dft"])
        err_ax.plot(
            ge["distance"],
            ge["ml_minus_dft"],
            "-o",
            color=color,
            lw=1.6,
            ms=3.8,
            label=labels[state],
        )

    ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
    err_ax.axhline(0.0, color="0.45", lw=0.9, ls=":")

    ax.set_xlabel("Fe-O distance (A)")
    ax.set_ylabel(r"$E - E_{4.0\ \mathrm{A}}$ (eV)")
    ax.set_title("PES")
    ax.set_xlim(1.75, 4.05)
    ax.set_ylim(-0.42, 0.30)
    ax.set_xticks([1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.0])
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.52, 1.02))

    err_ax.set_xlabel("Fe-O distance (A)")
    err_ax.set_ylabel("ML - DFT (eV)")
    err_ax.set_title("PES error")
    err_ax.set_xlim(1.75, 4.05)
    err_ax.set_xticks([1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.0])
    err_ax.set_ylim(-0.08, 0.12)
    err_ax.legend(frameon=False, loc="upper right")

    for label, axis in zip(["a", "b"], axes):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        axis.tick_params(direction="out", length=3.5, width=0.8)
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.35)

    out_png = OUT_DIR / "fen4_mace_dft_ml_pes_compare.png"
    out_pdf = OUT_DIR / "fen4_mace_dft_ml_pes_compare.pdf"
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_data()
    merged.to_csv(OUT_DIR / "fen4_mace_dft_ml_pes_compare_data.csv", index=False)
    plot(merged)

    valid = merged.dropna(subset=["ml_minus_dft"])
    summary = (
        valid.groupby("state")["ml_minus_dft"]
        .agg(
            mae=lambda s: s.abs().mean(),
            max_abs=lambda s: s.abs().max(),
            mean="mean",
            count="count",
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "fen4_mace_dft_ml_pes_error_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
