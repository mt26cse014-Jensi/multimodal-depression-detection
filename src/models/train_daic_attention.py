"""
Phase 6: Training and Evaluation Harness for Cross-Modal Attention on DAIC-WOZ.

Trains and evaluates 1-layer and 2-layer CrossModalAttention models using:
- Same 5 random seeds: [42, 123, 456, 789, 2024]
- Identical dev early-stopping protocol (composite BCE + 0.25 SmoothL1)
- Identical hierarchical majority voting aggregation (30s, 75s, 120s -> cross-scale)
- Identical 2,000-draw stratified participant bootstrap 95% CIs
- Produces unified comparison table against Phase 5 baselines:
    results/daic_phase6_comparison.csv
  and comparison plot:
    results/daic/daic_phase6_comparison_barplot.png
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.features.extract_daic_audio import match_turns_to_window
from src.features.extract_daic_text import (
    _get_smoke_test_sessions,
    load_config,
    resolve_device,
)
from src.models.cross_modal_attention import SmallCrossModalModel
from src.models.daic_eval import (
    METRIC_NAMES,
    aggregate_participant_predictions,
    bootstrap_participant_ci,
    compute_participant_metrics,
    paired_bootstrap_gap_ci,
)

logger = logging.getLogger(__name__)


# ── Data structures & session packing ──────────────────────────────────────────

def build_session_data(
    session_id: int,
    npz_path: Path,
    session_windows: pd.DataFrame,
    device: torch.device,
    phq8_max_score: float = 24.0,
    tol: float = 1e-3,
) -> Optional[Dict[str, Any]]:
    """Load cached turn sequences and build padded tensors + window aggregation matrix."""
    if not npz_path.exists():
        logger.warning("Cache file %s does not exist for session %d", npz_path, session_id)
        return None

    try:
        with np.load(npz_path) as d:
            turn_ids = d["turn_ids"]
            turn_durs = d["turn_durations"]
            start_times = d["start_times"]
            stop_times = d["stop_times"]
            text_embs_flat = d["text_embs"].astype(np.float32)
            text_offsets = d["text_offsets"]
            audio_embs_flat = d["audio_embs"].astype(np.float32)
            audio_offsets = d["audio_offsets"]
    except Exception as exc:
        logger.warning("Error loading %s for session %d: %s", npz_path, session_id, exc)
        return None

    n_turns = len(turn_ids)
    if n_turns == 0:
        logger.warning("Session %d has 0 turns", session_id)
        return None

    # Unpack turns
    turn_texts: List[np.ndarray] = []
    turn_audios: List[np.ndarray] = []
    for i in range(n_turns):
        t_arr = text_embs_flat[text_offsets[i] : text_offsets[i + 1]]
        a_arr = audio_embs_flat[audio_offsets[i] : audio_offsets[i + 1]]
        if len(t_arr) == 0:
            t_arr = np.zeros((1, 768), dtype=np.float32)
        if len(a_arr) == 0:
            a_arr = np.zeros((1, 768), dtype=np.float32)
        turn_texts.append(t_arr)
        turn_audios.append(a_arr)

    # Pad within session
    max_l_t = max(arr.shape[0] for arr in turn_texts)
    max_l_a = max(arr.shape[0] for arr in turn_audios)

    t_pad = np.zeros((n_turns, max_l_t, 768), dtype=np.float32)
    t_mask = np.ones((n_turns, max_l_t), dtype=bool)
    for i, arr in enumerate(turn_texts):
        L = arr.shape[0]
        t_pad[i, :L] = arr
        t_mask[i, :L] = False

    a_pad = np.zeros((n_turns, max_l_a, 768), dtype=np.float32)
    a_mask = np.ones((n_turns, max_l_a), dtype=bool)
    for i, arr in enumerate(turn_audios):
        T = arr.shape[0]
        a_pad[i, :T] = arr
        a_mask[i, :T] = False

    # Compute duration-weighted window matrix
    session_windows = session_windows.sort_values(["scale", "start_time"]).reset_index(drop=True)
    m_windows = len(session_windows)
    w_mat = np.zeros((m_windows, n_turns), dtype=np.float32)

    turns_pseudo_df = pd.DataFrame({
        "turn_id": turn_ids,
        "start_time": start_times,
        "stop_time": stop_times,
        "duration": turn_durs,
    })

    for w_idx, w_row in enumerate(session_windows.itertuples()):
        matched_idx = match_turns_to_window(turns_pseudo_df, w_row.start_time, w_row.end_time, tol=tol)
        if not matched_idx:
            # Fallback overlap
            ov_mask = (start_times < float(w_row.end_time)) & (stop_times > float(w_row.start_time))
            matched_idx = np.where(ov_mask)[0].tolist()

        if matched_idx:
            durs = turn_durs[matched_idx]
            sum_d = durs.sum()
            if sum_d > 0:
                w_mat[w_idx, matched_idx] = durs / sum_d
            else:
                w_mat[w_idx, matched_idx] = 1.0 / len(matched_idx)

    # Labels
    y_cls = session_windows["label"].to_numpy(dtype=np.float32) if "label" in session_windows.columns else np.full(m_windows, np.nan, dtype=np.float32)
    y_phq = session_windows["phq8_score"].to_numpy(dtype=np.float32) if "phq8_score" in session_windows.columns else np.full(m_windows, np.nan, dtype=np.float32)
    y_phq_norm = y_phq / float(phq8_max_score)

    return {
        "session_id": session_id,
        "t_pad": torch.tensor(t_pad, dtype=torch.float32, device=device),
        "a_pad": torch.tensor(a_pad, dtype=torch.float32, device=device),
        "t_mask": torch.tensor(t_mask, dtype=torch.bool, device=device),
        "a_mask": torch.tensor(a_mask, dtype=torch.bool, device=device),
        "w_mat": torch.tensor(w_mat, dtype=torch.float32, device=device),
        "y_cls": torch.tensor(y_cls, dtype=torch.float32, device=device),
        "y_phq_norm": torch.tensor(y_phq_norm, dtype=torch.float32, device=device),
        "windows_df": session_windows,
        "turn_ids": turn_ids,
        "start_times": start_times,
        "stop_times": stop_times,
        "turn_durations": turn_durs,
    }


def load_split_dataset(
    split: str,
    manifest_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    turns_cache_dir: Path,
    device: torch.device,
    scales: Sequence[int] = (30, 75, 120),
    session_subset: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Build list of packed session dictionaries for a given split."""
    sub_manifest = manifest_df[manifest_df["split"] == split].copy()
    if session_subset is not None:
        sub_manifest = sub_manifest[sub_manifest["session_id"].astype(int).isin(set(session_subset))]

    sessions = sub_manifest["session_id"].astype(int).tolist()
    sub_windows = windows_df[windows_df["split"] == split].copy()
    sub_windows = sub_windows[sub_windows["scale"].astype(int).isin(set(scales))]

    # Join phq8_score if not present in windows_df
    if "phq8_score" not in sub_windows.columns and "phq8_score" in sub_manifest.columns:
        sub_windows = sub_windows.merge(
            sub_manifest[["session_id", "phq8_score"]],
            on="session_id",
            how="left",
        )

    dataset: List[Dict[str, Any]] = []
    for sid in sessions:
        s_win = sub_windows[sub_windows["session_id"].astype(int) == sid].copy()
        if s_win.empty:
            continue
        npz_path = turns_cache_dir / f"{sid}.npz"
        data = build_session_data(sid, npz_path, s_win, device=device)
        if data is not None:
            dataset.append(data)

    logger.info("Loaded split '%s': %d valid sessions", split, len(dataset))
    return dataset


# ── Training & Evaluation ──────────────────────────────────────────────────────

def evaluate_sessions(
    model: nn.Module,
    sessions_data: Sequence[Dict[str, Any]],
    phq8_max_score: float = 24.0,
) -> Tuple[float, float, float, pd.DataFrame]:
    """Evaluate model across sessions.

    Returns:
        (bce_loss, reg_loss, composite_loss, window_preds_df)
    """
    model.eval()
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    n_cls_windows = 0
    n_reg_windows = 0

    bce_unweighted = nn.BCEWithLogitsLoss(reduction="sum")
    reg_criterion = nn.SmoothL1Loss(beta=0.1, reduction="sum")

    pred_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for s_data in sessions_data:
            logits, phq_norm = model(
                s_data["t_pad"],
                s_data["a_pad"],
                s_data["w_mat"],
                t_mask=s_data["t_mask"],
                a_mask=s_data["a_mask"],
            )

            y_cls = s_data["y_cls"]
            y_phq_norm = s_data["y_phq_norm"]
            w_df = s_data["windows_df"]

            logits_1d = logits.squeeze(-1)
            phq_norm_1d = phq_norm.squeeze(-1)

            # Losses
            valid_cls = ~torch.isnan(y_cls)
            if valid_cls.any():
                l_cls = bce_unweighted(logits_1d[valid_cls], y_cls[valid_cls])
                total_cls_loss += float(l_cls.item())
                n_cls_windows += int(valid_cls.sum().item())

            valid_reg = ~torch.isnan(y_phq_norm)
            if valid_reg.any():
                l_reg = reg_criterion(phq_norm_1d[valid_reg], y_phq_norm[valid_reg])
                total_reg_loss += float(l_reg.item())
                n_reg_windows += int(valid_reg.sum().item())

            probs = torch.sigmoid(logits_1d).cpu().numpy()
            phq_preds_scaled = (phq_norm_1d * float(phq8_max_score)).cpu().numpy()

            for i, row in enumerate(w_df.itertuples()):
                pred_records.append({
                    "session_id": int(row.session_id),
                    "window_id": getattr(row, "window_id", f"{row.session_id}_{row.scale}s_{i:04d}"),
                    "scale": int(row.scale),
                    "y_true": float(row.label) if hasattr(row, "label") and pd.notna(row.label) else float("nan"),
                    "phq8_true": float(row.phq8_score) if hasattr(row, "phq8_score") and pd.notna(row.phq8_score) else float("nan"),
                    "y_prob": float(probs[i]),
                    "phq8_pred": float(phq_preds_scaled[i]),
                })

    avg_cls_loss = total_cls_loss / max(n_cls_windows, 1)
    avg_reg_loss = total_reg_loss / max(n_reg_windows, 1)
    composite_loss = avg_cls_loss + 0.25 * avg_reg_loss

    preds_df = pd.DataFrame(pred_records)
    return avg_cls_loss, avg_reg_loss, composite_loss, preds_df


def train_single_seed(
    train_data: Sequence[Dict[str, Any]],
    dev_data: Sequence[Dict[str, Any]],
    n_layers: int,
    cfg: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Train one cross-modal attention seed with early stopping on dev composite loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Class weighting
    all_train_y = torch.cat([s["y_cls"][~torch.isnan(s["y_cls"])] for s in train_data])
    n_pos = float((all_train_y == 1.0).sum().item())
    n_neg = float((all_train_y == 0.0).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    reg_criterion = nn.SmoothL1Loss(beta=0.1)
    reg_weight = float(cfg["training"].get("reg_loss_weight", 0.25))

    model_params = cfg.get("model", {})
    model = SmallCrossModalModel(
        n_layers=n_layers,
        d_in_text=int(model_params.get("d_in_text", 768)),
        d_in_audio=int(model_params.get("d_in_audio", 768)),
        d_model=int(model_params.get("d_model", 128)),
        n_heads=int(model_params.get("n_heads", 4)),
        d_ff=int(model_params.get("d_ff", 256)),
        hidden_dim=int(model_params.get("hidden_dim", 128)),
        dropout=float(model_params.get("dropout", 0.3)),
        attn_dropout=float(model_params.get("attn_dropout", 0.1)),
    ).to(device)

    lr = float(cfg["training"].get("lr", 5e-4))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = int(cfg["training"].get("batch_size", 16))
    max_epochs = int(cfg["training"].get("epochs", 60))
    patience = int(cfg["training"].get("patience", 10))
    phq8_max = float(cfg["training"].get("phq8_max_score", 24.0))

    best_dev_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    rng = np.random.default_rng(seed)
    n_train_sessions = len(train_data)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(n_train_sessions)
        optimizer.zero_grad()

        for b_start in range(0, n_train_sessions, batch_size):
            b_indices = perm[b_start : b_start + batch_size]
            b_len = len(b_indices)
            accum_loss = 0.0

            for s_idx in b_indices:
                s_data = train_data[s_idx]
                logits, phq_norm = model(
                    s_data["t_pad"],
                    s_data["a_pad"],
                    s_data["w_mat"],
                    t_mask=s_data["t_mask"],
                    a_mask=s_data["a_mask"],
                )

                y_cls = s_data["y_cls"]
                y_phq_norm = s_data["y_phq_norm"]

                loss_cls = bce_criterion(logits.squeeze(-1), y_cls)
                valid_reg = ~torch.isnan(y_phq_norm)
                if valid_reg.any():
                    loss_reg = reg_criterion(phq_norm.squeeze(-1)[valid_reg], y_phq_norm[valid_reg])
                    loss = loss_cls + reg_weight * loss_reg
                else:
                    loss = loss_cls

                loss_scaled = loss / b_len
                loss_scaled.backward()
                accum_loss += float(loss.item())

            optimizer.step()
            optimizer.zero_grad()

        # Dev check
        _, _, dev_loss, _ = evaluate_sessions(model, dev_data, phq8_max_score=phq8_max)
        if dev_loss < best_dev_loss - 1e-4:
            best_dev_loss = dev_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best checkpoint and evaluate
    model.load_state_dict(best_state)
    _, _, final_dev_loss, dev_win_preds = evaluate_sessions(model, dev_data, phq8_max_score=phq8_max)

    # Participant aggregation
    part_df, scale_dfs = aggregate_participant_predictions(
        dev_win_preds,
        scales=tuple(cfg["training"].get("scales", [30, 75, 120])),
        phq8_max_score=phq8_max,
    )

    metrics = compute_participant_metrics(
        y_true=part_df["y_true"].to_numpy(),
        y_pred=part_df["y_pred"].to_numpy(),
        y_prob=part_df["y_prob"].to_numpy(),
        phq8_true=part_df["phq8_true"].to_numpy(),
        phq8_pred=part_df["phq8_pred"].to_numpy(),
    )

    return {
        "seed": seed,
        "n_layers": n_layers,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "metrics": metrics,
        "participant_df": part_df,
        "scale_dfs": scale_dfs,
        "window_preds_df": dev_win_preds,
        "model_state": best_state,
        "param_count": model.count_trainable_parameters(),
    }


# ── Full Experiment Suite & Table Generation ───────────────────────────────────

def run_attention_experiments(
    config_path: str | Path = "configs/daic_attention.yaml",
    smoke_test: bool = False,
    n_layers_list: Sequence[int] = (1, 2),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train 1L and 2L models, compute bootstrap CIs, and build the Phase 6 comparison table."""
    cfg = load_config(config_path)

    manifest_path = Path(cfg["dataset"]["manifest_path"])
    windows_path = Path(cfg["dataset"]["windows_path"])
    cache_root = Path(cfg["features"]["cache_dir"])
    turns_cache_dir = cache_root / "turns"
    checkpoints_dir = Path(cfg["training"].get("checkpoints_dir", "data/processed/daic/checkpoints_phase6"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    windows_df = pd.read_parquet(windows_path)

    device = resolve_device(cfg["training"].get("device", "cpu"))
    scales = tuple(cfg["training"].get("scales", [30, 75, 120]))
    phq8_max = float(cfg["training"].get("phq8_max_score", 24.0))

    if smoke_test:
        smoke_sessions = _get_smoke_test_sessions(manifest_path)
        seeds = [42]
        logger.info("SMOKE TEST MODE: 5 sessions %s, 1 seed", smoke_sessions)
        train_data = load_split_dataset("train", manifest_df, windows_df, turns_cache_dir, device, scales, smoke_sessions)
        dev_data = load_split_dataset("dev", manifest_df, windows_df, turns_cache_dir, device, scales, smoke_sessions)
        test_data = load_split_dataset("test", manifest_df, windows_df, turns_cache_dir, device, scales, smoke_sessions)
    else:
        seeds = cfg["training"].get("seeds", [42, 123, 456, 789, 2024])
        logger.info("FULL RUN MODE: %d seeds %s", len(seeds), seeds)
        train_data = load_split_dataset("train", manifest_df, windows_df, turns_cache_dir, device, scales)
        dev_data = load_split_dataset("dev", manifest_df, windows_df, turns_cache_dir, device, scales)
        test_data = load_split_dataset("test", manifest_df, windows_df, turns_cache_dir, device, scales)

    all_seed_results: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []

    # 1. Run 1-layer and 2-layer models
    for n_layers in n_layers_list:
        arch_name = f"cross_modal_attention_{n_layers}layer{'s' if n_layers > 1 else ''}"
        logger.info("=== Running %s (%d layers) across %d seeds ===", arch_name, n_layers, len(seeds))

        seed_part_dfs: List[pd.DataFrame] = []
        param_count = 0

        for seed in seeds:
            t0 = time.time()
            res = train_single_seed(train_data, dev_data, n_layers, cfg, seed, device)
            dt = time.time() - t0
            param_count = res["param_count"]
            all_seed_results.append(res)
            seed_part_dfs.append(res["participant_df"])

            logger.info(
                "[%s seed %d] Best epoch %d (dev loss %.4f) in %.1fs | Macro F1: %.3f, Dep F1: %.3f, Acc: %.3f, AUC: %.3f, RMSE: %.3f",
                arch_name,
                seed,
                res["best_epoch"],
                res["best_dev_loss"],
                dt,
                res["metrics"]["macro_f1"],
                res["metrics"]["dep_f1"],
                res["metrics"]["accuracy"],
                res["metrics"]["auc_roc"],
                res["metrics"]["phq8_rmse"],
            )

            # Save checkpoint
            ckpt_path = checkpoints_dir / f"{arch_name}_seed_{seed}.pt"
            torch.save(res["model_state"], ckpt_path)

        # Bootstrap CIs across seeds
        boot_ci = bootstrap_participant_ci(
            seed_part_dfs,
            n_bootstrap=2000 if not smoke_test else 50,
            seed=int(cfg["training"].get("bootstrap_seed", 42)),
        )

        row: Dict[str, Any] = {
            "split": "dev",
            "modality": arch_name,
            "variant": "participant",
            "scale_scope": "combined_majority",
            "n_participants": len(seed_part_dfs[0]),
            "n_seeds": len(seeds),
            "trainable_parameters": param_count,
        }
        for m in METRIC_NAMES:
            stats = boot_ci[m]
            row[f"{m}_mean"] = stats["mean"]
            row[f"{m}_seed_std"] = stats["seed_std"]
            row[f"{m}_ci_low"] = stats["ci_low"]
            row[f"{m}_ci_high"] = stats["ci_high"]

        comparison_rows.append(row)

    cma_results_df = pd.DataFrame(comparison_rows)

    # 2. Merge with Phase 5 results
    phase5_csv = Path(cfg["training"].get("phase5_results_csv", "results/daic_results.csv"))
    if phase5_csv.exists():
        phase5_df = pd.read_csv(phase5_csv)
        # Filter to dev combined_majority participant models
        phase5_sub = phase5_df[
            (phase5_df["split"] == "dev")
            & (phase5_df["scale_scope"] == "combined_majority")
            & (phase5_df["variant"] == "participant")
        ].copy()

        # Add parameter counts for Phase 5 models
        param_map = {
            "majority_baseline": 0,
            "text_only": 98_946,
            "audio_only": 197_378,
            "late_fusion": 295_810,
        }
        phase5_sub["trainable_parameters"] = phase5_sub["modality"].map(param_map).fillna(0).astype(int)

        unified_df = pd.concat([phase5_sub, cma_results_df], ignore_index=True)
    else:
        unified_df = cma_results_df

    # Calculate paired gap vs late_fusion for macro_f1 and dep_f1 if late_fusion is present
    late_fusion_row = unified_df[unified_df["modality"] == "late_fusion"]
    if not late_fusion_row.empty:
        lf_macro_f1 = float(late_fusion_row["macro_f1_mean"].iloc[0])
        lf_dep_f1 = float(late_fusion_row["dep_f1_mean"].iloc[0])
        unified_df["macro_f1_gap_vs_late_fusion"] = unified_df["macro_f1_mean"] - lf_macro_f1
        unified_df["dep_f1_gap_vs_late_fusion"] = unified_df["dep_f1_mean"] - lf_dep_f1
    else:
        unified_df["macro_f1_gap_vs_late_fusion"] = np.nan
        unified_df["dep_f1_gap_vs_late_fusion"] = np.nan

    out_csv = Path(cfg["training"].get("results_csv", "results/daic_phase6_comparison.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    unified_df.to_csv(out_csv, index=False)
    logger.info("Saved Phase 6 unified comparison to %s", out_csv)

    # 3. Generate Comparison Bar Plot
    plot_path = Path(cfg["training"].get("results_plot", "results/daic/daic_phase6_comparison_barplot.png"))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_phase6_comparison(unified_df, plot_path)
    logger.info("Saved comparison plot to %s", plot_path)

    # Save dev predictions
    pred_path = Path(cfg["training"].get("predictions_csv", "results/daic_attention_dev_predictions.csv"))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    if all_seed_results:
        all_seed_preds = pd.concat([r["participant_df"].assign(seed=r["seed"], n_layers=r["n_layers"]) for r in all_seed_results], ignore_index=True)
        all_seed_preds.to_csv(pred_path, index=False)

    return unified_df, cma_results_df


def _plot_phase6_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Generate side-by-side bar chart of Macro F1, Depressed F1, Accuracy, AUC-ROC, and PHQ-8 RMSE."""
    plot_df = df[df["split"] == "dev"].copy()
    if plot_df.empty:
        return

    modalities = plot_df["modality"].tolist()
    labels = [m.replace("_", " ").title() for m in modalities]

    metrics = [
        ("macro_f1", "Macro F1", [0.0, 1.0]),
        ("dep_f1", "Depressed F1", [0.0, 1.0]),
        ("accuracy", "Accuracy", [0.0, 1.0]),
        ("auc_roc", "AUC-ROC", [0.0, 1.0]),
        ("phq8_rmse", "PHQ-8 RMSE (lower is better)", [0.0, 10.0]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(modalities)]

    for idx, (m_key, title, ylim) in enumerate(metrics):
        ax = axes[idx]
        means = plot_df[f"{m_key}_mean"].to_numpy()
        err_low = means - plot_df[f"{m_key}_ci_low"].to_numpy()
        err_high = plot_df[f"{m_key}_ci_high"].to_numpy() - means
        yerr = [np.clip(err_low, 0, None), np.clip(err_high, 0, None)]

        bars = ax.bar(labels, means, yerr=yerr, capsize=4, color=colors, alpha=0.85, edgecolor="black")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=30)

        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Cross-Modal Attention on DAIC-WOZ.")
    parser.add_argument("--config", default="configs/daic_attention.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 seed on 5 smoke sessions")
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2], help="Number of CMA layers to evaluate")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_attention_experiments(
        config_path=args.config,
        smoke_test=args.smoke_test,
        n_layers_list=args.layers,
    )


if __name__ == "__main__":
    main()

