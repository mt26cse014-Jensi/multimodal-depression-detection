"""Generate exploratory analysis plots for EATD-Corpus dataset.

Generates:
1. results/eatd_class_counts.png: Class distribution (Control vs Depressed)
2. results/eatd_clip_durations.png: Audio clip duration histograms by valence and combined
3. results/eatd_text_lengths.png: Transcript character length distributions
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)

# Use clean aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]


def plot_class_counts(df: pd.DataFrame, output_path: Path | str) -> None:
    """Plot distribution of control vs depressed participants."""
    counts = df["label"].value_counts().sort_index()
    labels = ["Control (0)", "Depressed (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    total = len(df)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bars = ax.bar(labels, values, color=["#4C72B0", "#C44E52"], width=0.5, edgecolor="black", linewidth=1.2)

    for bar, val in zip(bars, values):
        pct = (val / total) * 100
        height = bar.get_height()
        ax.annotate(
            f"N = {val}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("EATD-Corpus: Participant Class Distribution (SDS Score >= 53)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Participants", fontsize=11)
    ax.set_ylim(0, max(values) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    logger.info("Saved class counts plot to %s", out_file)


def plot_clip_durations(df: pd.DataFrame, output_path: Path | str) -> None:
    """Plot audio clip duration distributions across positive, neutral, negative answers."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
    axes = axes.flatten()

    valences = ["positive", "neutral", "negative"]
    colors = ["#55A868", "#4C72B0", "#C44E52"]

    all_durs = []
    for idx, (val, color) in enumerate(zip(valences, colors)):
        ax = axes[idx]
        col = f"audio_{val}_dur"
        # Filter for valid non-dropped clips
        durs = df[df[f"audio_{val}_dropped"] == False][col].values
        all_durs.extend(durs)

        mean_dur = np.mean(durs) if len(durs) > 0 else 0
        median_dur = np.median(durs) if len(durs) > 0 else 0

        ax.hist(durs, bins=25, color=color, edgecolor="black", alpha=0.75)
        ax.axvline(mean_dur, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_dur:.1f}s")
        ax.axvline(median_dur, color="darkred", linestyle=":", linewidth=1.5, label=f"Median: {median_dur:.1f}s")

        ax.set_title(f"{val.capitalize()} Clips (N={len(durs)})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Duration (seconds)", fontsize=10)
        ax.set_ylabel("Clip Count", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Combined all clips
    ax_all = axes[3]
    mean_all = np.mean(all_durs) if len(all_durs) > 0 else 0
    median_all = np.median(all_durs) if len(all_durs) > 0 else 0

    ax_all.hist(all_durs, bins=30, color="#8172B3", edgecolor="black", alpha=0.75)
    ax_all.axvline(mean_all, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_all:.1f}s")
    ax_all.axvline(median_all, color="darkred", linestyle=":", linewidth=1.5, label=f"Median: {median_all:.1f}s")

    ax_all.set_title(f"All Valid Audio Clips Combined (N={len(all_durs)})", fontsize=12, fontweight="bold")
    ax_all.set_xlabel("Duration (seconds)", fontsize=10)
    ax_all.set_ylabel("Clip Count", fontsize=10)
    ax_all.legend(loc="upper right", fontsize=9)
    ax_all.spines["top"].set_visible(False)
    ax_all.spines["right"].set_visible(False)

    plt.suptitle("EATD-Corpus: Preprocessed Audio Clip Durations (16 kHz, Trimmed)", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    logger.info("Saved clip duration plot to %s", out_file)


def plot_text_lengths(df: pd.DataFrame, output_path: Path | str) -> None:
    """Plot character length distributions for Chinese transcripts."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
    axes = axes.flatten()

    valences = ["positive", "neutral", "negative"]
    colors = ["#55A868", "#4C72B0", "#C44E52"]

    for idx, (val, color) in enumerate(zip(valences, colors)):
        ax = axes[idx]
        col = f"text_{val}_clean"
        lengths = df[col].fillna("").apply(len).values

        mean_len = np.mean(lengths)
        median_len = np.median(lengths)

        ax.hist(lengths, bins=25, color=color, edgecolor="black", alpha=0.75)
        ax.axvline(mean_len, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_len:.1f}")
        ax.axvline(median_len, color="darkred", linestyle=":", linewidth=1.5, label=f"Median: {median_len:.1f}")

        ax.set_title(f"{val.capitalize()} Transcripts", fontsize=12, fontweight="bold")
        ax.set_xlabel("Character Count", fontsize=10)
        ax.set_ylabel("Transcript Count", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Combined text length
    ax_all = axes[3]
    combined_lengths = df["text_combined_clean"].fillna("").apply(len).values
    mean_comb = np.mean(combined_lengths)
    median_comb = np.median(combined_lengths)

    ax_all.hist(combined_lengths, bins=25, color="#8172B3", edgecolor="black", alpha=0.75)
    ax_all.axvline(mean_comb, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_comb:.1f}")
    ax_all.axvline(median_comb, color="darkred", linestyle=":", linewidth=1.5, label=f"Median: {median_comb:.1f}")

    ax_all.set_title("Combined Multimodal Transcripts", fontsize=12, fontweight="bold")
    ax_all.set_xlabel("Total Character Count", fontsize=10)
    ax_all.set_ylabel("Transcript Count", fontsize=10)
    ax_all.legend(loc="upper right", fontsize=9)
    ax_all.spines["top"].set_visible(False)
    ax_all.spines["right"].set_visible(False)

    plt.suptitle("EATD-Corpus: Cleaned Transcript Character Length Distributions", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    logger.info("Saved text lengths plot to %s", out_file)


def generate_all_plots(
    manifest_path: Path | str = "data/processed/eatd/manifest.csv",
    config_path: Path | str = "configs/eatd_preprocess.yaml",
) -> None:
    """Generate all Phase 1 visualization figures."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(manifest_path)

    plot_class_counts(df, config["plots"]["class_counts"])
    plot_clip_durations(df, config["plots"]["clip_durations"])
    plot_text_lengths(df, config["plots"]["text_lengths"])


def main():
    parser = argparse.ArgumentParser(description="Generate plots for EATD dataset.")
    parser.add_argument("--manifest", type=str, default="data/processed/eatd/manifest.csv")
    parser.add_argument("--config", type=str, default="configs/eatd_preprocess.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_all_plots(manifest_path=args.manifest, config_path=args.config)


if __name__ == "__main__":
    main()
