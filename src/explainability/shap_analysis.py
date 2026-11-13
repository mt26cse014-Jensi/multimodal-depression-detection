"""
Grouped KernelSHAP / Exact 2-Group Shapley Analysis (Text vs. Audio).

Computes the relative contribution of Text embeddings (768-d) versus
Audio embeddings (1536-d) on the multimodal late-fusion architecture.

Formulation:
With M=2 groups, the coalition lattice has 4 states:
  emptyset, {text}, {audio}, {text, audio}
We evaluate the exact expectation over background training windows,
satisfying the efficiency, symmetry, and dummy player axioms identically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_exact_2group_shap(
    model: nn.Module,
    X_query: np.ndarray,
    X_background: np.ndarray,
    text_dim: int = 768,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute exact 2-group Shapley values (Text vs Audio) for query samples.

    Args:
        model: Trained late fusion MLP (takes normalized 2304-d vector, outputs logits).
        X_query: Query window features of shape (N, 2304).
        X_background: Background reference features of shape (B, 2304).
        text_dim: Dimension of text modality (first text_dim columns).
        batch_size: Batch size for evaluating query windows.
        device: Torch device.

    Returns:
        phi_text: (N,) Shapley values for Text group.
        phi_audio: (N,) Shapley values for Audio group.
        base_value: Expected baseline prediction over background distribution.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    bg_t = torch.tensor(X_background, dtype=torch.float32, device=device)
    B = bg_t.shape[0]

    with torch.no_grad():
        # Baseline: v(emptyset)
        bg_logits, _ = model(bg_t)
        base_value = float(torch.sigmoid(bg_logits).mean().item())

    N = X_query.shape[0]
    phi_text = np.zeros(N, dtype=np.float32)
    phi_audio = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_x = X_query[start:end]
            b_len = end - start

            # Evaluate v(both)
            x_t = torch.tensor(batch_x, dtype=torch.float32, device=device)
            both_logits, _ = model(x_t)
            v_both = torch.sigmoid(both_logits).squeeze(-1).cpu().numpy()  # (b_len,)

            for i in range(b_len):
                idx = start + i
                xi = x_t[i : i + 1]  # (1, 2304)

                # v({text}): text from xi, audio from background
                x_text_bg_audio = bg_t.clone()
                x_text_bg_audio[:, :text_dim] = xi[:, :text_dim]
                l_text, _ = model(x_text_bg_audio)
                v_text = float(torch.sigmoid(l_text).mean().item())

                # v({audio}): audio from xi, text from background
                bg_text_x_audio = bg_t.clone()
                bg_text_x_audio[:, text_dim:] = xi[:, text_dim:]
                l_audio, _ = model(bg_text_x_audio)
                v_audio = float(torch.sigmoid(l_audio).mean().item())

                vb = float(v_both[i])
                p_text = 0.5 * (v_text - base_value) + 0.5 * (vb - v_audio)
                p_audio = 0.5 * (v_audio - base_value) + 0.5 * (vb - v_text)

                phi_text[idx] = p_text
                phi_audio[idx] = p_audio

    return phi_text, phi_audio, base_value


def run_shap_modality_suite(
    model: nn.Module,
    scaler: Any,
    dev_windows_df: pd.DataFrame,
    X_dev_raw: np.ndarray,
    X_train_raw: np.ndarray,
    output_csv: Path | str = "results/shap_modality_contributions.csv",
    output_plot: Path | str = "results/figures/shap_modality_contributions.png",
    n_background: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute grouped SHAP modality analysis across evaluation participants.

    Saves:
        1. Modality contribution CSV (per participant and summary rows).
        2. Tri-panel visualization plot in results/figures/.
    """
    output_csv = Path(output_csv)
    output_plot = Path(output_plot)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    # Subsample background from train
    rng = np.random.default_rng(seed)
    n_bg = min(n_background, len(X_train_raw))
    bg_idx = rng.choice(len(X_train_raw), size=n_bg, replace=False)
    X_bg_raw = X_train_raw[bg_idx]

    # Normalize strictly using train-fitted scaler
    X_bg_norm = scaler.transform(X_bg_raw).astype(np.float32)
    X_dev_norm = scaler.transform(X_dev_raw).astype(np.float32)

    logger.info("Computing exact 2-group SHAP for %d dev windows with %d background samples...", len(X_dev_norm), n_bg)
    phi_t, phi_a, base_val = compute_exact_2group_shap(model, X_dev_norm, X_bg_norm)

    # Attach SHAP values to windows DataFrame
    df = dev_windows_df.copy()
    df["shap_text"] = phi_t
    df["shap_audio"] = phi_a
    df["session_id"] = df["session_id"].astype(int)

    # Participant-level aggregation
    part_records: List[Dict[str, Any]] = []
    for sid, grp in df.groupby("session_id", sort=True):
        mean_phi_t = float(grp["shap_text"].mean())
        mean_phi_a = float(grp["shap_audio"].mean())
        abs_t = abs(mean_phi_t)
        abs_a = abs(mean_phi_a)
        total_abs = abs_t + abs_a
        pct_t = (abs_t / total_abs * 100.0) if total_abs > 0 else 50.0
        pct_a = (abs_a / total_abs * 100.0) if total_abs > 0 else 50.0

        y_true = float(grp["label"].iloc[0]) if "label" in grp.columns and pd.notna(grp["label"].iloc[0]) else float("nan")
        phq_true = float(grp["phq8_score"].iloc[0]) if "phq8_score" in grp.columns and pd.notna(grp["phq8_score"].iloc[0]) else float("nan")

        part_records.append({
            "session_id": sid,
            "y_true": y_true,
            "phq8_true": phq_true,
            "mean_shap_text": mean_phi_t,
            "mean_shap_audio": mean_phi_a,
            "abs_shap_text": abs_t,
            "abs_shap_audio": abs_a,
            "pct_shap_text": pct_t,
            "pct_shap_audio": pct_a,
            "dominant_modality": "text" if abs_t >= abs_a else "audio",
        })

    part_df = pd.DataFrame(part_records)

    # Summary statistics
    overall_mean_abs_t = float(part_df["abs_shap_text"].mean())
    overall_mean_abs_a = float(part_df["abs_shap_audio"].mean())

    dep_df = part_df[part_df["y_true"] == 1.0]
    ctrl_df = part_df[part_df["y_true"] == 0.0]

    dep_mean_t = float(dep_df["abs_shap_text"].mean()) if not dep_df.empty else float("nan")
    dep_mean_a = float(dep_df["abs_shap_audio"].mean()) if not dep_df.empty else float("nan")

    ctrl_mean_t = float(ctrl_df["abs_shap_text"].mean()) if not ctrl_df.empty else float("nan")
    ctrl_mean_a = float(ctrl_df["abs_shap_audio"].mean()) if not ctrl_df.empty else float("nan")

    summary_rows = [
        {"session_id": -1, "y_true": np.nan, "phq8_true": np.nan, "mean_shap_text": np.nan, "mean_shap_audio": np.nan, "abs_shap_text": overall_mean_abs_t, "abs_shap_audio": overall_mean_abs_a, "pct_shap_text": (overall_mean_abs_t / (overall_mean_abs_t + overall_mean_abs_a) * 100), "pct_shap_audio": (overall_mean_abs_a / (overall_mean_abs_t + overall_mean_abs_a) * 100), "dominant_modality": "OVERALL_MEAN"},
        {"session_id": -2, "y_true": 1.0, "phq8_true": np.nan, "mean_shap_text": np.nan, "mean_shap_audio": np.nan, "abs_shap_text": dep_mean_t, "abs_shap_audio": dep_mean_a, "pct_shap_text": (dep_mean_t / (dep_mean_t + dep_mean_a) * 100) if (dep_mean_t + dep_mean_a) > 0 else np.nan, "pct_shap_audio": (dep_mean_a / (dep_mean_t + dep_mean_a) * 100) if (dep_mean_t + dep_mean_a) > 0 else np.nan, "dominant_modality": "DEPRESSED_MEAN"},
        {"session_id": -3, "y_true": 0.0, "phq8_true": np.nan, "mean_shap_text": np.nan, "mean_shap_audio": np.nan, "abs_shap_text": ctrl_mean_t, "abs_shap_audio": ctrl_mean_a, "pct_shap_text": (ctrl_mean_t / (ctrl_mean_t + ctrl_mean_a) * 100) if (ctrl_mean_t + ctrl_mean_a) > 0 else np.nan, "pct_shap_audio": (ctrl_mean_a / (ctrl_mean_t + ctrl_mean_a) * 100) if (ctrl_mean_t + ctrl_mean_a) > 0 else np.nan, "dominant_modality": "CONTROL_MEAN"},
    ]
    summary_df = pd.DataFrame(summary_rows)
    combined_export = pd.concat([part_df, summary_df], ignore_index=True)
    combined_export.to_csv(output_csv, index=False)
    logger.info("Saved SHAP modality results to %s", output_csv)

    # Plot
    _plot_shap_modality(part_df, overall_mean_abs_t, overall_mean_abs_a, dep_mean_t, dep_mean_a, ctrl_mean_t, ctrl_mean_a, output_plot)
    logger.info("Saved SHAP modality plot to %s", output_plot)

    return part_df, summary_df


def _plot_shap_modality(
    part_df: pd.DataFrame,
    overall_t: float,
    overall_a: float,
    dep_t: float,
    dep_a: float,
    ctrl_t: float,
    ctrl_a: float,
    output_path: Path,
) -> None:
    """Generate 3-panel visualization of modality contribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Overall Mean Absolute SHAP
    ax1 = axes[0]
    modalities = ["Text (RoBERTa)", "Audio (WavLM)"]
    vals = [overall_t, overall_a]
    colors = ["#1f77b4", "#ff7f0e"]
    bars = ax1.bar(modalities, vals, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Overall Mean Absolute SHAP Value", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean |SHAP| (Probability Scale)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h / 2), ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    # Panel 2: Modality Importance by Diagnostic Cohort
    ax2 = axes[1]
    x = np.arange(2)
    w = 0.35
    ax2.bar(x - w / 2, [dep_t, ctrl_t], width=w, label="Text", color="#1f77b4", alpha=0.85, edgecolor="black")
    ax2.bar(x + w / 2, [dep_a, ctrl_a], width=w, label="Audio", color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Depressed (PHQ-8>=10)", "Control (PHQ-8<10)"], fontsize=10, fontweight="bold")
    ax2.set_title("Modality Contribution by Cohort", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean |SHAP|", fontsize=11)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Distribution of Relative Modality Share per Participant
    ax3 = axes[2]
    # Scatter of pct_shap_text vs pct_shap_audio
    dep = part_df[part_df["y_true"] == 1.0]
    ctrl = part_df[part_df["y_true"] == 0.0]

    ax3.scatter(dep["session_id"].astype(str), dep["pct_shap_text"], color="#d62728", label="Depressed", s=60, edgecolors="black", zorder=3)
    ax3.scatter(ctrl["session_id"].astype(str), ctrl["pct_shap_text"], color="#2ca02c", label="Control", s=60, edgecolors="black", zorder=3)
    ax3.axhline(50.0, color="gray", linestyle="--", alpha=0.7, label="Equal Share (50%)")
    ax3.set_title("Relative Text Contribution (%) per Participant", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Text Contribution Share (%)", fontsize=11)
    ax3.set_xlabel("Participant (Session ID)", fontsize=11)
    ax3.set_ylim([0, 100])
    ax3.tick_params(axis="x", rotation=90, labelsize=8)
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

