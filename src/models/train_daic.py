"""Train and evaluate DAIC-WOZ single-modality, late-fusion, and ablation models (Phase 5).

Protocol:
  - Frozen pretrained text (MentalBERT or RoBERTa/BERT fallback) & audio (WavLM) embeddings
  - Shared MultiTaskDepMLP trunk (hidden_dim=128, dropout=0.3) -> binary logit + PHQ-8 score
  - StandardScaler fit strictly on train windows
  - Class-weighted BCE + weighted SmoothL1 PHQ-8 regression loss
  - Dev split used ONLY for early stopping and hyperparameter selection
  - Participant aggregation: within-scale majority vote (30s, 75s, 120s), then
    majority vote across the 3 scales (plus single-scale breakdown)
  - 5 random seeds -> mean, seed std, and 2,000-draw stratified participant bootstrap 95% CIs
  - Interviewer ablation: participant-only vs participant+Ellie text
  - Test split evaluated ONCE after freezing dev configs (outputs predictions; computes
    metrics if a test-label file is supplied)

Usage:
    python -m src.models.train_daic --smoke-test
    python -m src.models.train_daic --tune --run-dev
    python -m src.models.train_daic --eval-test-once
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler

from src.features.extract_daic_audio import load_daic_audio_embeddings
from src.features.extract_daic_text import _get_smoke_test_sessions, load_config, load_daic_text_embeddings
from src.models.daic_eval import (
    METRIC_NAMES,
    aggregate_participant_predictions,
    bootstrap_participant_ci,
    compute_daic_majority_baseline,
    compute_participant_metrics,
    paired_bootstrap_gap_ci,
)

logger = logging.getLogger(__name__)


# ── Model definition ───────────────────────────────────────────────────────────

class MultiTaskDepMLP(nn.Module):
    """Single hidden-layer MLP with parallel binary classification and PHQ-8 regression heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden_dim, 1)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cls_logit, reg_norm_pred)`` of shape ``(B,)`` each."""
        h = self.trunk(x)
        return self.cls_head(h).squeeze(-1), self.reg_head(h).squeeze(-1)


# ── Split loading & leakage checks ─────────────────────────────────────────────

def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32, copy=False)


def verify_no_split_overlap(train_sessions: Sequence[int], eval_sessions: Sequence[int]) -> None:
    """Assert zero participant/session overlap between train and evaluation splits."""
    overlap = set(int(s) for s in train_sessions).intersection(int(s) for s in eval_sessions)
    if overlap:
        raise ValueError(f"Participant leakage detected across splits: {sorted(overlap)}")


def build_daic_split_data(
    config_path: str = "configs/daic_models.yaml",
    modality: str = "text_only",
    split: str = "train",
    variant: str = "participant",
    scales: Optional[Sequence[int]] = None,
    session_subset: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings + labels + PHQ-8 scores for one DAIC split and modality.

    Args:
        modality: ``"text_only"``, ``"audio_only"``, or ``"late_fusion"``.
        split: ``"train"``, ``"dev"``, or ``"test"``.
        variant: ``"participant"`` or ``"with_ellie"`` (for text/fusion).
    """
    if modality not in ("text_only", "audio_only", "late_fusion"):
        raise ValueError(f"Unsupported modality: {modality!r}")

    cfg = load_config(config_path)
    l2_norm = bool(cfg["training"].get("l2_norm_modalities", False))
    manifest = pd.read_csv(cfg["dataset"]["manifest_path"])
    sid_to_phq8 = {
        int(r.session_id): (math.nan if pd.isna(r.phq8_score) else float(r.phq8_score))
        for r in manifest.itertuples()
    }

    if modality == "text_only":
        X, win_df = load_daic_text_embeddings(
            config_path=config_path, split=split, variant=variant,
            scales=scales, session_subset=session_subset,
        )
    elif modality == "audio_only":
        X, win_df = load_daic_audio_embeddings(
            config_path=config_path, split=split,
            scales=scales, session_subset=session_subset,
        )
    else:
        Xt, win_t = load_daic_text_embeddings(
            config_path=config_path, split=split, variant=variant,
            scales=scales, session_subset=session_subset,
        )
        Xa, win_a = load_daic_audio_embeddings(
            config_path=config_path, split=split,
            scales=scales, session_subset=session_subset,
        )
        if not win_t["window_id"].equals(win_a["window_id"]):
            raise ValueError("Window alignment mismatch between text and audio caches")
        if l2_norm:
            Xt = _l2_normalize_rows(Xt)
            Xa = _l2_normalize_rows(Xa)
        X = np.concatenate([Xt, Xa], axis=1).astype(np.float32, copy=False)
        win_df = win_t

    win_df = win_df.copy()
    win_df["y_true"] = win_df["label"].astype(float)
    win_df["phq8_true"] = [sid_to_phq8.get(int(s), math.nan) for s in win_df["session_id"]]
    return X, win_df


# ── Single-seed training with dev early stopping ───────────────────────────────

def _predict_windows(
    model: MultiTaskDepMLP,
    X_scaled: np.ndarray,
    phq8_max_score: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward pass on scaled features -> ``(probs, phq8_preds)``."""
    model.eval()
    device = torch.device("cpu")
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, reg_norm = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        phq8_preds = np.clip(reg_norm.cpu().numpy() * phq8_max_score, 0.0, phq8_max_score)
    return probs.astype(np.float64), phq8_preds.astype(np.float64)


def train_single_seed(
    X_train: np.ndarray,
    win_train: pd.DataFrame,
    X_dev: np.ndarray,
    win_dev: pd.DataFrame,
    mlp_cfg: dict[str, Any],
    seed: int,
    scales: Sequence[int] = (30, 75, 120),
    phq8_max_score: float = 24.0,
) -> dict[str, Any]:
    """Train one MultiTaskDepMLP seed on ``train`` with early stopping on ``dev``.

    Fits ``StandardScaler`` strictly on ``X_train``.
    """
    verify_no_split_overlap(win_train["session_id"].tolist(), win_dev["session_id"].tolist())

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cpu")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train).astype(np.float32)
    X_dv_s = scaler.transform(X_dev).astype(np.float32)

    y_tr_cls = win_train["y_true"].to_numpy(dtype=np.float32)
    y_tr_reg = (win_train["phq8_true"].to_numpy(dtype=np.float32) / float(phq8_max_score))
    y_dv_cls = win_dev["y_true"].to_numpy(dtype=np.float32)
    y_dv_reg = (win_dev["phq8_true"].to_numpy(dtype=np.float32) / float(phq8_max_score))

    n_pos = float((y_tr_cls == 1.0).sum())
    n_neg = float((y_tr_cls == 0.0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    reg_criterion = nn.SmoothL1Loss(beta=0.1)
    reg_weight = float(mlp_cfg.get("reg_loss_weight", 0.25))

    model = MultiTaskDepMLP(
        input_dim=int(X_tr_s.shape[1]),
        hidden_dim=int(mlp_cfg.get("hidden_dim", 128)),
        dropout=float(mlp_cfg.get("dropout", 0.3)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(mlp_cfg.get("lr", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-2)),
    )

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
    y_tr_cls_t = torch.tensor(y_tr_cls, dtype=torch.float32, device=device)
    y_tr_reg_t = torch.tensor(np.nan_to_num(y_tr_reg, nan=0.0), dtype=torch.float32, device=device)
    tr_reg_valid_t = torch.tensor(~np.isnan(y_tr_reg), dtype=torch.bool, device=device)

    X_dv_t = torch.tensor(X_dv_s, dtype=torch.float32, device=device)
    y_dv_cls_t = torch.tensor(y_dv_cls, dtype=torch.float32, device=device)
    y_dv_reg_t = torch.tensor(np.nan_to_num(y_dv_reg, nan=0.0), dtype=torch.float32, device=device)
    dv_reg_valid_t = torch.tensor(~np.isnan(y_dv_reg), dtype=torch.bool, device=device)

    batch_size = int(mlp_cfg.get("batch_size", 64))
    max_epochs = int(mlp_cfg.get("epochs", 60))
    patience = int(mlp_cfg.get("patience", 10))
    n_samples = len(X_tr_t)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    best_dev_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_samples, generator=gen)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            logits, reg_pred = model(X_tr_t[idx])
            loss_cls = bce_criterion(logits, y_tr_cls_t[idx])
            valid_m = tr_reg_valid_t[idx]
            if valid_m.any():
                loss_reg = reg_criterion(reg_pred[valid_m], y_tr_reg_t[idx][valid_m])
                loss = loss_cls + reg_weight * loss_reg
            else:
                loss = loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            dv_logits, dv_reg = model(X_dv_t)
            dv_cls_loss = bce_criterion(dv_logits, y_dv_cls_t)
            if dv_reg_valid_t.any():
                dv_reg_loss = reg_criterion(dv_reg[dv_reg_valid_t], y_dv_reg_t[dv_reg_valid_t])
                dev_loss = float((dv_cls_loss + reg_weight * dv_reg_loss).item())
            else:
                dev_loss = float(dv_cls_loss.item())

        if dev_loss < best_dev_loss - 1e-5:
            best_dev_loss = dev_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    dev_probs, dev_phq8 = _predict_windows(model, X_dv_s, phq8_max_score=phq8_max_score)
    dev_win_preds = pd.DataFrame(
        {
            "session_id": win_dev["session_id"].to_numpy(dtype=int),
            "window_id": win_dev["window_id"].tolist(),
            "scale": win_dev["scale"].to_numpy(dtype=int),
            "y_true": win_dev["y_true"].to_numpy(dtype=float),
            "phq8_true": win_dev["phq8_true"].to_numpy(dtype=float),
            "y_prob": dev_probs,
            "phq8_pred": dev_phq8,
        }
    )
    part_df, scale_dfs = aggregate_participant_predictions(
        dev_win_preds, scales=scales, phq8_max_score=phq8_max_score,
    )

    return {
        "model_state": best_state,
        "input_dim": int(X_tr_s.shape[1]),
        "scaler": scaler,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "dev_window_preds": dev_win_preds,
        "dev_participant_df": part_df,
        "dev_scale_dfs": scale_dfs,
    }


# ── Dev tuning & checkpoint helpers ────────────────────────────────────────────

def _selection_objective(metrics: dict[str, float], phq8_max_score: float = 24.0) -> float:
    auc = metrics["auc_roc"] if not math.isnan(metrics["auc_roc"]) else 0.5
    mac = metrics["macro_f1"]
    rmse = metrics["phq8_rmse"] if not math.isnan(metrics["phq8_rmse"]) else phq8_max_score
    return float(0.5 * (mac + auc) - 0.1 * (rmse / phq8_max_score))


def tune_hyperparameters_on_dev(
    config_path: str = "configs/daic_models.yaml",
    modalities: Sequence[str] = ("text_only", "audio_only", "late_fusion"),
    save_to_config: bool = True,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Tune ``lr``, ``weight_decay``, and ``reg_loss_weight`` strictly on dev (seed 42)."""
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    base_mlp = dict(train_cfg["mlp"])
    grid = train_cfg["tuning_grid"]
    scales = tuple(int(s) for s in train_cfg.get("scales", [30, 75, 120]))
    phq8_max = float(train_cfg.get("phq8_max_score", 24.0))
    tune_seed = int(train_cfg["seeds"][0])

    tuning_rows: list[dict[str, Any]] = []
    best_by_mod: dict[str, dict[str, Any]] = {}

    for modality in modalities:
        X_tr, w_tr = build_daic_split_data(config_path, modality=modality, split="train")
        X_dv, w_dv = build_daic_split_data(config_path, modality=modality, split="dev")

        best_score = -float("inf")
        best_params = dict(base_mlp)

        for lr in grid["lr"]:
            for wd in grid["weight_decay"]:
                for lam in grid["reg_loss_weight"]:
                    trial_cfg = dict(base_mlp)
                    trial_cfg["lr"] = float(lr)
                    trial_cfg["weight_decay"] = float(wd)
                    trial_cfg["reg_loss_weight"] = float(lam)

                    out = train_single_seed(
                        X_tr, w_tr, X_dv, w_dv,
                        mlp_cfg=trial_cfg, seed=tune_seed,
                        scales=scales, phq8_max_score=phq8_max,
                    )
                    p_df = out["dev_participant_df"]
                    m = compute_participant_metrics(
                        p_df["y_true"].to_numpy(),
                        p_df["y_pred"].to_numpy(),
                        p_df["y_prob"].to_numpy(),
                        p_df["phq8_true"].to_numpy(),
                        p_df["phq8_pred"].to_numpy(),
                    )
                    obj = _selection_objective(m, phq8_max_score=phq8_max)
                    tuning_rows.append(
                        {
                            "modality": modality,
                            "lr": lr,
                            "weight_decay": wd,
                            "reg_loss_weight": lam,
                            "best_epoch": out["best_epoch"],
                            "best_dev_loss": out["best_dev_loss"],
                            "objective": obj,
                            **m,
                        }
                    )
                    if obj > best_score:
                        best_score = obj
                        best_params = dict(trial_cfg)

        best_by_mod[modality] = best_params
        logger.info(
            "Dev-tuned %s: lr=%g wd=%g reg_w=%g (obj=%.4f)",
            modality, best_params["lr"], best_params["weight_decay"],
            best_params["reg_loss_weight"], best_score,
        )

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_csv = Path(train_cfg["tuning_csv"])
    tuning_csv.parent.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(tuning_csv, index=False)

    if save_to_config:
        cfg["training"]["frozen_hparams"] = {
            mod: {
                "hidden_dim": int(p["hidden_dim"]),
                "dropout": float(p["dropout"]),
                "epochs": int(p["epochs"]),
                "patience": int(p["patience"]),
                "batch_size": int(p["batch_size"]),
                "lr": float(p["lr"]),
                "weight_decay": float(p["weight_decay"]),
                "reg_loss_weight": float(p["reg_loss_weight"]),
            }
            for mod, p in best_by_mod.items()
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        logger.info("Frozen dev hyperparameters written to %s", config_path)

    return best_by_mod, tuning_df


def _save_checkpoint(
    ckpt_path: Path,
    run_out: dict[str, Any],
    mlp_cfg: dict[str, Any],
    modality: str,
    variant: str,
    seed: int,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    scaler: StandardScaler = run_out["scaler"]
    payload = {
        "modality": modality,
        "variant": variant,
        "seed": int(seed),
        "input_dim": int(run_out["input_dim"]),
        "mlp_cfg": dict(mlp_cfg),
        "best_epoch": int(run_out["best_epoch"]),
        "best_dev_loss": float(run_out["best_dev_loss"]),
        "model_state": run_out["model_state"],
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "scaler_var": scaler.var_,
        "scaler_n_samples": int(scaler.n_samples_seen_),
    }
    torch.save(payload, ckpt_path)


def _load_checkpoint_and_predict(
    ckpt_path: Path,
    X_eval: np.ndarray,
    phq8_max_score: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = ckpt["scaler_var"]
    scaler.n_samples_seen_ = ckpt["scaler_n_samples"]

    X_s = scaler.transform(X_eval).astype(np.float32)
    mlp_cfg = ckpt["mlp_cfg"]
    model = MultiTaskDepMLP(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(mlp_cfg.get("hidden_dim", 128)),
        dropout=float(mlp_cfg.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model_state"])
    return _predict_windows(model, X_s, phq8_max_score=phq8_max_score)


# ── Formatting & plotting ──────────────────────────────────────────────────────

def _summary_dict_to_row(
    meta: dict[str, Any],
    summary: dict[str, dict[str, float]],
) -> dict[str, Any]:
    row = dict(meta)
    for m in METRIC_NAMES:
        row[f"{m}_mean"] = summary[m]["mean"]
        row[f"{m}_seed_std"] = summary[m]["seed_std"]
        row[f"{m}_ci_low"] = summary[m]["ci_low"]
        row[f"{m}_ci_high"] = summary[m]["ci_high"]
    return row


def _plot_daic_comparison(results_df: pd.DataFrame, output_path: str | Path) -> None:
    """Create grouped bar chart with 95% CI error bars for combined multi-scale dev metrics."""
    sub = results_df[results_df["scale_scope"] == "combined_majority"].copy()
    if sub.empty:
        return

    metrics = ["macro_f1", "dep_f1", "accuracy", "auc_roc"]
    labels = ["Macro F1", "Dep F1", "Accuracy", "AUC-ROC"]
    names = sub["modality"].tolist()
    x = np.arange(len(names))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 2.4), 5.5))
    colors = plt.cm.Set2.colors

    for j, (m, lbl) in enumerate(zip(metrics, labels)):
        means = sub[f"{m}_mean"].to_numpy(dtype=float)
        lo_err = np.maximum(0.0, means - sub[f"{m}_ci_low"].to_numpy(dtype=float))
        hi_err = np.maximum(0.0, sub[f"{m}_ci_high"].to_numpy(dtype=float) - means)
        yerr = np.vstack([np.nan_to_num(lo_err, nan=0.0), np.nan_to_num(hi_err, nan=0.0)])
        offset = (j - len(metrics) / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width, yerr=yerr,
            label=lbl, color=colors[j % len(colors)],
            capsize=3, edgecolor="black", linewidth=0.5,
        )

    ax.set_ylabel("Score")
    ax.set_title("DAIC-WOZ Dev Results (30/75/120s Majority Vote, 5 Seeds & 95% Bootstrap CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main evaluation runners ────────────────────────────────────────────────────

def _evaluate_modality_seeds(
    config_path: str,
    modality: str,
    variant: str,
    mlp_cfg: dict[str, Any],
    seeds: Sequence[int],
    scales: Sequence[int],
    phq8_max: float,
    n_bootstrap: int,
    boot_seed: int,
    session_subset: Optional[Sequence[int]] = None,
    save_checkpoints: bool = True,
    ckpt_dir: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    """Train all seeds for one (modality, variant) and return summary rows + predictions."""
    X_tr, w_tr = build_daic_split_data(
        config_path=config_path, modality=modality, split="train",
        variant=variant, session_subset=session_subset,
    )
    X_dv, w_dv = build_daic_split_data(
        config_path=config_path, modality=modality, split="dev",
        variant=variant, session_subset=session_subset,
    )

    seed_part_dfs: list[pd.DataFrame] = []
    seed_scale_dfs: dict[int, list[pd.DataFrame]] = {int(s): [] for s in scales}
    pred_dfs: list[pd.DataFrame] = []

    for seed in seeds:
        out = train_single_seed(
            X_tr, w_tr, X_dv, w_dv,
            mlp_cfg=mlp_cfg, seed=int(seed),
            scales=scales, phq8_max_score=phq8_max,
        )
        if save_checkpoints and ckpt_dir is not None:
            tag = modality if variant == "participant" else f"{modality}_{variant}"
            _save_checkpoint(
                ckpt_dir / f"{tag}_seed{int(seed)}.pt",
                run_out=out, mlp_cfg=mlp_cfg,
                modality=modality, variant=variant, seed=int(seed),
            )

        p_df = out["dev_participant_df"].copy()
        p_df.insert(0, "seed", int(seed))
        p_df.insert(0, "variant", variant)
        p_df.insert(0, "modality", modality)
        pred_dfs.append(p_df)
        seed_part_dfs.append(out["dev_participant_df"])

        for s in scales:
            s_int = int(s)
            if s_int in out["dev_scale_dfs"]:
                seed_scale_dfs[s_int].append(out["dev_scale_dfs"][s_int])

    rows: list[dict[str, Any]] = []
    combined_summary = bootstrap_participant_ci(seed_part_dfs, n_bootstrap=n_bootstrap, seed=boot_seed)
    rows.append(
        _summary_dict_to_row(
            {
                "split": "dev",
                "modality": modality,
                "variant": variant,
                "scale_scope": "combined_majority",
                "n_participants": int(len(seed_part_dfs[0])),
                "n_seeds": int(len(seeds)),
            },
            combined_summary,
        )
    )

    for s in scales:
        s_int = int(s)
        if not seed_scale_dfs[s_int]:
            continue
        s_summary = bootstrap_participant_ci(seed_scale_dfs[s_int], n_bootstrap=n_bootstrap, seed=boot_seed)
        rows.append(
            _summary_dict_to_row(
                {
                    "split": "dev",
                    "modality": modality,
                    "variant": variant,
                    "scale_scope": f"scale_{s_int}s",
                    "n_participants": int(len(seed_scale_dfs[s_int][0])),
                    "n_seeds": int(len(seeds)),
                },
                s_summary,
            )
        )

    return rows, pred_dfs, seed_part_dfs


def run_dev_experiments(
    config_path: str = "configs/daic_models.yaml",
    smoke_test: bool = False,
    smoke_sessions: Optional[Sequence[int]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run main DAIC models (text, audio, fusion) + interviewer ablation on dev.

    Returns:
        (results_df, dev_preds_df, ablation_df, ablation_preds_df)
    """
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    scales = [int(s) for s in train_cfg.get("scales", [30, 75, 120])]
    phq8_max = float(train_cfg.get("phq8_max_score", 24.0))
    frozen = train_cfg.get("frozen_hparams", {})
    default_mlp = dict(train_cfg["mlp"])
    ckpt_dir = Path(train_cfg["checkpoints_dir"])

    if smoke_test:
        seeds = [int(train_cfg["seeds"][0])]
        n_boot = 200
        if smoke_sessions is None:
            smoke_sessions = _get_smoke_test_sessions(cfg["dataset"]["manifest_path"])
        save_ckpts = False
    else:
        seeds = [int(s) for s in train_cfg["seeds"]]
        n_boot = int(train_cfg.get("bootstrap_samples", 2000))
        save_ckpts = True

    boot_seed = int(train_cfg.get("bootstrap_seed", 42))
    manifest_df = pd.read_csv(cfg["dataset"]["manifest_path"])

    all_rows: list[dict[str, Any]] = []
    all_dev_preds: list[pd.DataFrame] = []
    text_part_seed_dfs: list[pd.DataFrame] = []
    text_part_rows: list[dict[str, Any]] = []
    text_part_preds: list[pd.DataFrame] = []

    dev_sids_eval: Optional[list[int]] = None

    for modality in ("text_only", "audio_only", "late_fusion"):
        mlp_cfg = dict(frozen.get(modality, default_mlp))
        try:
            rows, pred_dfs, seed_part_dfs = _evaluate_modality_seeds(
                config_path=config_path,
                modality=modality,
                variant="participant",
                mlp_cfg=mlp_cfg,
                seeds=seeds,
                scales=scales,
                phq8_max=phq8_max,
                n_bootstrap=n_boot,
                boot_seed=boot_seed,
                session_subset=smoke_sessions,
                save_checkpoints=save_ckpts,
                ckpt_dir=ckpt_dir,
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", modality, e)
            continue

        all_rows.extend(rows)
        all_dev_preds.extend(pred_dfs)
        if dev_sids_eval is None and seed_part_dfs:
            dev_sids_eval = seed_part_dfs[0]["session_id"].astype(int).tolist()

        if modality == "text_only":
            text_part_seed_dfs = seed_part_dfs
            text_part_rows = rows
            text_part_preds = pred_dfs

    # Majority-class baseline row on identical dev participants
    if dev_sids_eval:
        maj_summary = compute_daic_majority_baseline(
            manifest_df=manifest_df,
            eval_session_ids=dev_sids_eval,
            n_bootstrap=n_boot,
            seed=boot_seed,
        )
        all_rows.insert(
            0,
            _summary_dict_to_row(
                {
                    "split": "dev",
                    "modality": "majority_baseline",
                    "variant": "participant",
                    "scale_scope": "combined_majority",
                    "n_participants": len(dev_sids_eval),
                    "n_seeds": 1,
                },
                maj_summary,
            ),
        )

    results_df = pd.DataFrame(all_rows)
    dev_preds_df = pd.concat(all_dev_preds, ignore_index=True) if all_dev_preds else pd.DataFrame()

    # ── Interviewer ablation (participant-only vs participant + Ellie) ─────────
    ablation_rows: list[dict[str, Any]] = []
    ablation_preds: list[pd.DataFrame] = []

    if text_part_seed_dfs:
        text_mlp_cfg = dict(frozen.get("text_only", default_mlp))
        try:
            ellie_rows, ellie_preds, ellie_seed_dfs = _evaluate_modality_seeds(
                config_path=config_path,
                modality="text_only",
                variant="with_ellie",
                mlp_cfg=text_mlp_cfg,
                seeds=seeds,
                scales=scales,
                phq8_max=phq8_max,
                n_bootstrap=n_boot,
                boot_seed=boot_seed,
                session_subset=smoke_sessions,
                save_checkpoints=save_ckpts,
                ckpt_dir=ckpt_dir,
            )
            ablation_rows.extend(text_part_rows)
            ablation_rows.extend(ellie_rows)
            ablation_preds.extend(text_part_preds)
            ablation_preds.extend(ellie_preds)

            gap_summary = paired_bootstrap_gap_ci(
                base_seed_dfs=text_part_seed_dfs,
                alt_seed_dfs=ellie_seed_dfs,
                n_bootstrap=n_boot,
                seed=boot_seed,
            )
            ablation_rows.append(
                _summary_dict_to_row(
                    {
                        "split": "dev",
                        "modality": "text_only",
                        "variant": "gap_with_ellie_minus_participant",
                        "scale_scope": "combined_majority",
                        "n_participants": len(text_part_seed_dfs[0]),
                        "n_seeds": len(seeds),
                    },
                    gap_summary,
                )
            )
        except FileNotFoundError as e:
            logger.warning("Skipping with_ellie ablation: %s", e)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_preds_df = pd.concat(ablation_preds, ignore_index=True) if ablation_preds else pd.DataFrame()

    if not smoke_test:
        for df_out, key in (
            (results_df, "results_csv"),
            (dev_preds_df, "dev_predictions_csv"),
            (ablation_df, "ablation_csv"),
            (ablation_preds_df, "ablation_predictions_csv"),
        ):
            if not df_out.empty:
                p = Path(train_cfg[key])
                p.parent.mkdir(parents=True, exist_ok=True)
                df_out.to_csv(p, index=False)
                logger.info("Saved %s (%d rows)", p, len(df_out))

        if not results_df.empty:
            _plot_daic_comparison(results_df, train_cfg["results_plot"])

    return results_df, dev_preds_df, ablation_df, ablation_preds_df


# ── One-time frozen test inference ─────────────────────────────────────────────

def evaluate_test_once(
    config_path: str = "configs/daic_models.yaml",
    test_labels_csv: Optional[str] = None,
    force: bool = False,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Evaluate frozen dev checkpoints ONCE on the DAIC test split.

    Always produces per-participant predictions at ``results/daic_test_predictions.csv``.
    If ``test_labels_csv`` (columns: session_id/Participant_ID, PHQ8_Binary, PHQ8_Score)
    is provided, computes test metrics + bootstrap 95% CIs cleanly without touching any config.
    """
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    if train_cfg.get("test_evaluated", False) and not force:
        raise RuntimeError(
            "Test split has already been evaluated (test_evaluated=true in config). "
            "Pass --force only if reproducing identically without tuning."
        )

    seeds = [int(s) for s in train_cfg["seeds"]]
    scales = [int(s) for s in train_cfg.get("scales", [30, 75, 120])]
    phq8_max = float(train_cfg.get("phq8_max_score", 24.0))
    ckpt_dir = Path(train_cfg["checkpoints_dir"])
    n_boot = int(train_cfg.get("bootstrap_samples", 2000))
    boot_seed = int(train_cfg.get("bootstrap_seed", 42))

    # Optional external labels if released to the user
    ext_labels: dict[int, tuple[float, float]] = {}
    if test_labels_csv:
        ldf = pd.read_csv(test_labels_csv)
        id_col = next(c for c in ldf.columns if c.lower() in ("session_id", "participant_id"))
        bin_col = next(c for c in ldf.columns if c.lower() in ("phq8_binary", "label"))
        scr_col = next((c for c in ldf.columns if c.lower() == "phq8_score"), None)
        for _, r in ldf.iterrows():
            ext_labels[int(r[id_col])] = (
                float(r[bin_col]),
                float(r[scr_col]) if scr_col else math.nan,
            )

    test_pred_frames: list[pd.DataFrame] = []
    test_metric_rows: list[dict[str, Any]] = []

    for modality, variant in (
        ("text_only", "participant"),
        ("audio_only", "participant"),
        ("late_fusion", "participant"),
        ("text_only", "with_ellie"),
    ):
        tag = modality if variant == "participant" else f"{modality}_{variant}"
        try:
            X_te, w_te = build_daic_split_data(
                config_path=config_path, modality=modality, split="test", variant=variant,
            )
        except FileNotFoundError:
            logger.warning("Skipping test evaluation for (%s, %s): embeddings missing", modality, variant)
            continue

        if ext_labels:
            w_te["y_true"] = [ext_labels.get(int(s), (math.nan, math.nan))[0] for s in w_te["session_id"]]
            w_te["phq8_true"] = [ext_labels.get(int(s), (math.nan, math.nan))[1] for s in w_te["session_id"]]

        seed_part_dfs: list[pd.DataFrame] = []
        for seed in seeds:
            ckpt_path = ckpt_dir / f"{tag}_seed{seed}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Frozen checkpoint missing: {ckpt_path}")

            probs, phq8 = _load_checkpoint_and_predict(ckpt_path, X_te, phq8_max_score=phq8_max)
            win_preds = pd.DataFrame(
                {
                    "session_id": w_te["session_id"].to_numpy(dtype=int),
                    "window_id": w_te["window_id"].tolist(),
                    "scale": w_te["scale"].to_numpy(dtype=int),
                    "y_true": w_te["y_true"].to_numpy(dtype=float),
                    "phq8_true": w_te["phq8_true"].to_numpy(dtype=float),
                    "y_prob": probs,
                    "phq8_pred": phq8,
                }
            )
            part_df, _ = aggregate_participant_predictions(win_preds, scales=scales, phq8_max_score=phq8_max)
            part_df.insert(0, "seed", seed)
            part_df.insert(0, "variant", variant)
            part_df.insert(0, "modality", modality)
            test_pred_frames.append(part_df)
            seed_part_dfs.append(part_df)

        if ext_labels and seed_part_dfs:
            summary = bootstrap_participant_ci(seed_part_dfs, n_bootstrap=n_boot, seed=boot_seed)
            test_metric_rows.append(
                _summary_dict_to_row(
                    {
                        "split": "test",
                        "modality": modality,
                        "variant": variant,
                        "scale_scope": "combined_majority",
                        "n_participants": len(seed_part_dfs[0]),
                        "n_seeds": len(seeds),
                    },
                    summary,
                )
            )

    test_preds_df = pd.concat(test_pred_frames, ignore_index=True) if test_pred_frames else pd.DataFrame()
    if not test_preds_df.empty:
        out_path = Path(train_cfg["test_predictions_csv"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        test_preds_df.to_csv(out_path, index=False)
        logger.info("Saved test predictions to %s (%d rows)", out_path, len(test_preds_df))

    # Mark test evaluation completed cleanly unless dry-running without outputs
    if not test_preds_df.empty:
        cfg["training"]["test_evaluated"] = True
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    test_metrics_df = pd.DataFrame(test_metric_rows) if test_metric_rows else None
    return test_preds_df, test_metrics_df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate DAIC-WOZ models (Phase 5).")
    parser.add_argument("--config", default="configs/daic_models.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast check on 5 sessions, seed 42")
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters strictly on dev and freeze config")
    parser.add_argument("--run-dev", action="store_true", help="Run full 5-seed dev training & ablation")
    parser.add_argument("--eval-test-once", action="store_true", help="Evaluate frozen checkpoints once on test split")
    parser.add_argument("--test-labels-csv", default=None, help="Optional test labels CSV if available")
    parser.add_argument("--force", action="store_true", help="Override test_evaluated lock")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.smoke_test:
        res_df, _, abl_df, _ = run_dev_experiments(config_path=args.config, smoke_test=True)
        print("\n=== DAIC PHASE 5 SMOKE TEST (DEV SUBSET) ===")
        if not res_df.empty:
            print(res_df.to_string(index=False))
        if not abl_df.empty:
            print("\n=== INTERVIEWER ABLATION (SMOKE TEST) ===")
            print(abl_df.to_string(index=False))
        return

    if args.tune:
        tune_hyperparameters_on_dev(config_path=args.config, save_to_config=True)

    if args.run_dev:
        res_df, _, abl_df, _ = run_dev_experiments(config_path=args.config, smoke_test=False)
        print("\n=== DAIC-WOZ DEV RESULTS ===")
        print(res_df.to_string(index=False))
        if not abl_df.empty:
            print("\n=== DAIC-WOZ INTERVIEWER ABLATION ===")
            print(abl_df.to_string(index=False))

    if args.eval_test_once:
        test_preds_df, test_metrics_df = evaluate_test_once(
            config_path=args.config, test_labels_csv=args.test_labels_csv, force=args.force,
        )
        print(f"\nTest predictions generated: {len(test_preds_df)} rows across {test_preds_df['session_id'].nunique() if not test_preds_df.empty else 0} sessions")
        if test_metrics_df is not None:
            print(test_metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

