"""Generate publication-quality figures for the manuscript."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig_gcs_comparison():
    """Figure 1: GCS across modes under each shift regime (the headline result)."""
    df = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")

    # Parse mode and regime from condition column
    df["mode"] = df["condition"].apply(lambda c: c.rsplit("_", 2)[0] if "tau" in c or "instB" in c or "schema" in c else c.rsplit("_", 2)[0])
    df["regime"] = df["condition"].apply(lambda c: "baseline" if "baseline" in c else "temporal" if "temporal" in c else "instB" if "instB" in c else "schema")

    # Filter to non-schema regimes for the main figure
    regimes = ["baseline", "temporal", "instB"]
    regime_labels = ["Baseline", "Temporal Drift", "Institutional Swap"]
    modes = ["single", "naive_rag", "linear", "graph"]
    mode_labels = ["Single", "Naive RAG", "Linear", "Graph"]
    mode_colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    x = np.arange(len(regimes))
    width = 0.19
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for i, (mode, label, color) in enumerate(zip(modes, mode_labels, mode_colors)):
        gcs_values = []
        for regime in regimes:
            row = df[df["condition"].str.contains(mode) & df["condition"].str.contains(
                {"baseline": "baseline", "temporal": "temporal", "instB": "instB"}[regime]
            )]
            if len(row) > 0:
                gcs_values.append(row.iloc[0]["GCS"])
            else:
                # naive_rag baseline = linear baseline by construction
                fallback = df[df["condition"].str.contains("linear") & df["condition"].str.contains("baseline")]
                gcs_values.append(fallback.iloc[0]["GCS"] if len(fallback) > 0 else 0.0)

        offset = (i - 1.5) * width
        ax.bar(x + offset, gcs_values, width, label=label, color=color, alpha=0.85)

    ax.set_ylabel("Guideline Compliance Score (GCS)")
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper right", ncol=2, framealpha=0.95, fontsize=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

    # Annotate the key drop (Linear under instB)
    linear_instb_gcs = df[df["condition"] == "linear_institution_swap_instB"]["GCS"].values[0]
    bar_x = 2 + 0.5 * width  # Linear is 3rd mode (index 2), instB is 3rd regime (index 2)
    ax.text(bar_x, linear_instb_gcs + 0.03, f"{linear_instb_gcs:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_gcs_comparison.pdf")
    plt.savefig(FIGURES_DIR / "fig_gcs_comparison.png")
    plt.close()
    print(f"Saved: figures/fig_gcs_comparison.pdf")


def fig_sfi_latency_tradeoff():
    """Figure 2: SFI vs Latency showing the safety-latency tradeoff."""
    df = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")

    # Filter to baseline + instB conditions (excluding naive_rag and schema)
    conditions = [
        "single_baseline_tau_old",
        "single_institution_swap_instB",
        "linear_baseline_tau_old",
        "linear_institution_swap_instB",
        "graph_baseline_tau_old",
        "graph_institution_swap_instB",
    ]
    plot_df = df[df["condition"].isin(conditions)].copy()

    colors = {"single": "#d62728", "linear": "#1f77b4", "graph": "#2ca02c"}
    markers = {"baseline": "o", "instB": "s"}

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for _, row in plot_df.iterrows():
        cond = row["condition"]
        mode = cond.split("_")[0]
        regime = "instB" if "instB" in cond else "baseline"
        gcs = row["GCS"]
        size = 80 + gcs * 150

        ax.scatter(
            row["latency_mean"], row["SFI"],
            s=size, c=colors[mode], marker=markers[regime],
            alpha=0.8, edgecolors="black", linewidth=0.5, zorder=5,
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="Single"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="Linear"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=8, label="Graph"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="Baseline (circle)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=6, label="Inst. shift (square)"),
    ]
    ax.legend(handles=legend_elements, loc="center right", framealpha=0.95, fontsize=7.5)

    # Annotate key points
    linear_instb = plot_df[plot_df["condition"] == "linear_institution_swap_instB"].iloc[0]
    graph_instb = plot_df[plot_df["condition"] == "graph_institution_swap_instB"].iloc[0]

    ax.annotate(
        f"Linear + shift\nGCS = {linear_instb['GCS']:.2f}",
        xy=(linear_instb["latency_mean"], linear_instb["SFI"]),
        xytext=(22, 0.615),
        fontsize=7.5, ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
    )
    ax.annotate(
        f"Graph + shift\nGCS = {graph_instb['GCS']:.2f}",
        xy=(graph_instb["latency_mean"], graph_instb["SFI"]),
        xytext=(36, 0.655),
        fontsize=7.5, ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
    )

    ax.set_xlabel("Latency (seconds per patient)")
    ax.set_ylabel("Semantic Fidelity Index (SFI)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(10, 38)
    ax.set_ylim(0.60, 0.80)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_sfi_latency.pdf")
    plt.savefig(FIGURES_DIR / "fig_sfi_latency.png")
    plt.close()
    print(f"Saved: figures/fig_sfi_latency.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate manuscript figures.")
    parser.parse_args()

    fig_gcs_comparison()
    fig_sfi_latency_tradeoff()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
