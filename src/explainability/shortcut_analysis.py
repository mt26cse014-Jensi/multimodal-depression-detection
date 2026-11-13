"""
Shortcut Pattern Analysis: Interview Temporal Dynamics.

Evaluates whether model predictions rely on conversational shortcut patterns,
such as turns concentrated disproportionately in the last half or wrap-up
phase of the clinical interview.

Statistical Tests:
1. Spearman rank correlation between relative position and turn importance.
2. Mann-Whitney U test and mean ratio comparing first half (r < 0.5) vs second half (r >= 0.5).
3. Quartile distribution of all turns and top-5 influential turns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def analyze_interview_temporal_shortcuts(
    occlusion_df: pd.DataFrame,
    manifest_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Perform statistical analysis on turn importance vs. interview position.

    Args:
        occlusion_df: DataFrame containing relative_position, abs_delta_p, session_id.
        manifest_df: Optional manifest DataFrame to merge ground-truth labels.

    Returns:
        Dict of statistical metrics: spearman_r, spearman_p,
        first_half_mean, second_half_mean, half_ratio, mann_whitney_u,
        mann_whitney_p, quartile_stats.
    """
    df = occlusion_df.copy()
    if manifest_df is not None and "y_true" not in df.columns:
        sub_m = manifest_df[["session_id", "phq8_binary"]].rename(columns={"phq8_binary": "y_true"})
        df = df.merge(sub_m, on="session_id", how="left")

    rel_pos = df["relative_position"].to_numpy(dtype=float)
    imp = df["abs_delta_p"].to_numpy(dtype=float)

    # 1. Spearman Rank Correlation
    spearman_res = stats.spearmanr(rel_pos, imp)
    spearman_r = float(spearman_res.statistic) if hasattr(spearman_res, "statistic") else float(spearman_res[0])
    spearman_p = float(spearman_res.pvalue) if hasattr(spearman_res, "pvalue") else float(spearman_res[1])

    # 2. First Half vs. Second Half
    first_half_mask = rel_pos < 0.5
    second_half_mask = rel_pos >= 0.5

    imp_first = imp[first_half_mask]
    imp_second = imp[second_half_mask]

    first_mean = float(imp_first.mean()) if len(imp_first) > 0 else 0.0
    second_mean = float(imp_second.mean()) if len(imp_second) > 0 else 0.0
    half_ratio = (second_mean / first_mean) if first_mean > 0 else float("nan")

    if len(imp_first) > 0 and len(imp_second) > 0:
        mwu_res = stats.mannwhitneyu(imp_second, imp_first, alternative="two-sided")
        mwu_stat = float(mwu_res.statistic)
        mwu_p = float(mwu_res.pvalue)
    else:
        mwu_stat = float("nan")
        mwu_p = float("nan")

    # 3. Quartile Breakdown
    df["quartile"] = pd.cut(
        df["relative_position"],
        bins=[-1e-4, 0.25, 0.50, 0.75, 1.0001],
        labels=["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"],
    )

    quartile_means = df.groupby("quartile", observed=False)["abs_delta_p"].mean().to_dict()
    quartile_counts = df.groupby("quartile", observed=False)["abs_delta_p"].count().to_dict()

    # Top-5 turns breakdown
    top5_df = df[df["rank_in_session"] <= 5] if "rank_in_session" in df.columns else df.nlargest(len(df) // 10, "abs_delta_p")
    top5_quartile_pct = (top5_df["quartile"].value_counts(normalize=True) * 100.0).to_dict()

    results = {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "first_half_mean_imp": first_mean,
        "second_half_mean_imp": second_mean,
        "half_ratio_second_to_first": half_ratio,
        "mann_whitney_u": mwu_stat,
        "mann_whitney_p": mwu_p,
        "quartile_means": {str(k): float(v) for k, v in quartile_means.items()},
        "quartile_counts": {str(k): int(v) for k, v in quartile_counts.items()},
        "top5_quartile_pct": {str(k): float(v) for k, v in top5_quartile_pct.items()},
        "total_turns_analyzed": len(df),
    }

    logger.info(
        "Shortcut Analysis: Spearman r=%.3f (p=%.3e), 1st half mean=%.4f, 2nd half mean=%.4f (ratio=%.2fx, MWU p=%.3e)",
        spearman_r,
        spearman_p,
        first_mean,
        second_mean,
        half_ratio,
        mwu_p,
    )
    return results


def plot_shortcut_analysis(
    occlusion_df: pd.DataFrame,
    stats_dict: Dict[str, Any],
    output_path: Path | str = "results/figures/turn_importance_by_position.png",
) -> None:
    """Generate comprehensive 3-panel visualization of turn importance vs interview position."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    df = occlusion_df.copy()
    rel_pos = df["relative_position"].to_numpy()
    imp = df["abs_delta_p"].to_numpy()

    # Panel 1: Scatter + Running Binned Mean Trend
    ax1 = axes[0]
    ax1.scatter(rel_pos, imp, color="#1f77b4", alpha=0.35, s=25, edgecolors="none", label="Turns")

    # Binned average curve (10 bins across [0, 1])
    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    binned_means = []
    binned_errs = []
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        sub = imp[(rel_pos >= b_low) & (rel_pos < b_high)]
        if len(sub) > 0:
            binned_means.append(float(sub.mean()))
            binned_errs.append(float(sub.std() / np.sqrt(len(sub))))
        else:
            binned_means.append(np.nan)
            binned_errs.append(np.nan)

    ax1.errorbar(bin_centers, binned_means, yerr=binned_errs, fmt="-o", color="#d62728", lw=2.5, capsize=4, label="Binned Mean +- SEM")
    ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Interview Midpoint (r=0.5)")
    ax1.set_title(
        f"Turn Importance vs Relative Position\nSpearman r = {stats_dict['spearman_r']:.3f} (p = {stats_dict['spearman_p']:.3e})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Relative Position in Interview (0.0=Start, 1.0=End)", fontsize=10)
    ax1.set_ylabel("Turn Importance (|Delta P|)", fontsize=10)
    ax1.set_xlim([-0.02, 1.02])
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Quartile Comparison Boxplot
    ax2 = axes[1]
    if "quartile" not in df.columns:
        df["quartile"] = pd.cut(
            df["relative_position"],
            bins=[-1e-4, 0.25, 0.50, 0.75, 1.0001],
            labels=["Q1\n(0-25%)", "Q2\n(25-50%)", "Q3\n(50-75%)", "Q4\n(75-100%)"],
        )
    q_data = [df[df["quartile"] == q]["abs_delta_p"].to_numpy() for q in df["quartile"].cat.categories]
    bp = ax2.boxplot(q_data, patch_artist=True, labels=df["quartile"].cat.categories, showfliers=False)
    q_colors = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd"]
    for patch, color in zip(bp["boxes"], q_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set(color="red", linewidth=2)

    ax2.set_title(
        f"Importance across Interview Quartiles\n2nd-to-1st Half Ratio = {stats_dict['half_ratio_second_to_first']:.2f}x (MWU p={stats_dict['mann_whitney_p']:.3e})",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_ylabel("Turn Importance (|Delta P|)", fontsize=10)
    ax2.set_xlabel("Interview Phase", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Distribution of Top-5 Influential Turns
    ax3 = axes[2]
    top_q_dict = stats_dict.get("top5_quartile_pct", {})
    categories = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]
    short_cats = ["Q1\n(0-25%)", "Q2\n(25-50%)", "Q3\n(50-75%)", "Q4\n(75-100%)"]
    top_pcts = [top_q_dict.get(c, 0.0) for c in categories]

    bars = ax3.bar(short_cats, top_pcts, color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax3.axhline(25.0, color="gray", linestyle="--", alpha=0.7, label="Uniform Chance (25%)")
    ax3.set_title("Distribution of Top-5 Turns by Interview Phase", fontsize=11, fontweight="bold")
    ax3.set_ylabel("% of Top-5 Most Important Turns", fontsize=10)
    ax3.set_xlabel("Interview Phase", fontsize=10)
    ax3.set_ylim([0, max(max(top_pcts, default=25) + 15, 40)])
    ax3.legend()
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h + 1), ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved shortcut analysis plot to %s", output_path)
