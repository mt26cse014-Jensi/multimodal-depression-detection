"""Train and evaluate EATD-Corpus late-fusion models (Phase 3).

Concatenates cached text and audio embeddings per participant and trains:
  - MLP head (same architecture & hyperparameters as single-modality baselines)
  - Logistic Regression (with stratified inner-CV selection of C)

Reuses loaders, aggregation, inner-CV C selection, MLP training/prediction,
metrics, majority baseline, and plotting directly from existing src/ modules.

Usage:
    python -m src.models.train_eatd_fusion --config configs/eatd_fusion.yaml --smoke-test
    python -m src.models.train_eatd_fusion --config configs/eatd_fusion.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.extract_audio import load_audio_embeddings
from src.features.extract_text import load_text_embeddings
from src.models.train_eatd_baselines import (
    _compute_metrics,
    _get_smoke_test_pids,
    _plot_barplot,
    _predict_mlp,
    _train_mlp,
    compute_majority_baseline,
    load_config,
    select_best_logreg_c,
)

logger = logging.getLogger(__name__)

METRIC_COLS = ["macro_f1", "dep_f1", "accuracy", "auc_roc"]

AUDIO_ENCODER_LABELS = {
    "model_name": "xlsr",
    "alt_model_name": "wavlm",
}


# ── Feature fusion ───────────────────────────────────────────────────

def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2-D matrix safely."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32, copy=False)


def build_fused_features(
    config_path: str = "configs/eatd_fusion.yaml",
    encoder_key: str = "model_name",
    aggregation: str = "mean",
    l2_norm_modalities: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load cached text and audio embeddings and concatenate them per participant.

    Verifies exact participant alignment and label consistency before fusing.

    Args:
        config_path: Path to YAML config file.
        encoder_key: Audio encoder key in config (``"model_name"`` or ``"alt_model_name"``).
        aggregation: Answer aggregation mode (``"mean"`` or ``"concat"``).
        l2_norm_modalities: If True, L2-normalize each modality block before concatenation.

    Returns:
        X_fused: (N, D_text + D_audio) float32 array.
        y: (N,) int64 binary labels.
        pids: List of N participant IDs.
    """
    X_text, y_text, pids_text = load_text_embeddings(
        config_path=config_path, aggregation=aggregation,
    )
    X_audio, y_audio, pids_audio = load_audio_embeddings(
        config_path=config_path, encoder_key=encoder_key, aggregation=aggregation,
    )

    if set(pids_text) != set(pids_audio):
        raise ValueError("Participant mismatch between text and audio embeddings")

    if pids_text != pids_audio:
        audio_idx = {pid: i for i, pid in enumerate(pids_audio)}
        order = [audio_idx[pid] for pid in pids_text]
        X_audio = X_audio[order]
        y_audio = y_audio[order]

    if not np.array_equal(y_text, y_audio):
        raise ValueError("Label mismatch between text and audio embeddings")

    if l2_norm_modalities:
        X_text = _l2_normalize_rows(X_text)
        X_audio = _l2_normalize_rows(X_audio)

    X_fused = np.concatenate([X_text, X_audio], axis=1).astype(np.float32, copy=False)
    logger.info(
        "Built fused features (%s, %s): text=%s + audio=%s -> fused=%s",
        aggregation, encoder_key, X_text.shape, X_audio.shape, X_fused.shape,
    )
    return X_fused, y_text, pids_text


# ── Fusion training loop ─────────────────────────────────────────────

def run_fusion(
    config_path: str = "configs/eatd_fusion.yaml",
    smoke_test: bool = False,
    smoke_pids: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run late-fusion experiments across folds, seeds, aggregations, and audio encoders.

    Args:
        config_path: Path to Phase 3 YAML config.
        smoke_test: If True, run only fold 0, seed 42 on ``smoke_pids``.
        smoke_pids: Participant subset for smoke testing.

    Returns:
        runs_df: Per-(aggregation, modality, model, fold, seed) metric rows.
        preds_df: Out-of-fold per-participant predictions for explainability.
    """
    cfg = load_config(config_path)
    feat_cfg = cfg["features"]
    train_cfg = cfg["training"]
    mlp_cfg = train_cfg["mlp"]
    logreg_cfg = train_cfg["logreg"]

    aggregations = feat_cfg.get("aggregations", ["mean"])
    l2_norm = bool(feat_cfg.get("l2_norm_modalities", False))
    seeds = list(train_cfg["seeds"])
    class_weight = train_cfg.get("class_weight", "balanced")

    c_grid = logreg_cfg["c_grid"]
    inner_cv_folds = int(logreg_cfg.get("inner_cv_folds", 5))
    inner_cv_scoring = logreg_cfg.get("inner_cv_scoring", "roc_auc")
    lr_max_iter = int(logreg_cfg.get("max_iter", 2000))
    lr_solver = logreg_cfg.get("solver", "lbfgs")

    manifest = pd.read_csv(cfg["dataset"]["manifest_path"])
    pid_to_sds = dict(zip(manifest["participant_id"], manifest["sds_score"].astype(float)))

    with open(cfg["dataset"]["folds_path"], "r", encoding="utf-8") as f:
        folds = json.load(f)["folds"]

    if smoke_test:
        folds = [folds[0]]
        seeds = [seeds[0]]

    audio_cfg = feat_cfg["audio"]
    encoder_keys = [k for k in ("model_name", "alt_model_name") if k in audio_cfg]

    run_rows = []
    pred_rows = []

    for aggregation in aggregations:
        for encoder_key in encoder_keys:
            audio_tag = AUDIO_ENCODER_LABELS.get(encoder_key, encoder_key)
            mod_name = f"fusion_text+{audio_tag}"

            X_all, y_all, pids_all = build_fused_features(
                config_path=config_path,
                encoder_key=encoder_key,
                aggregation=aggregation,
                l2_norm_modalities=l2_norm,
            )
            pid_to_idx = {pid: i for i, pid in enumerate(pids_all)}

            for fold_info in folds:
                fold_idx = int(fold_info["fold"])
                train_pids = fold_info["train"]
                test_pids = fold_info["test"]

                if smoke_pids is not None:
                    smoke_set = set(smoke_pids)
                    train_pids = [p for p in train_pids if p in smoke_set and p in pid_to_idx]
                    test_pids = [p for p in test_pids if p in smoke_set and p in pid_to_idx]

                train_idx = [pid_to_idx[p] for p in train_pids if p in pid_to_idx]
                test_idx = [pid_to_idx[p] for p in test_pids if p in pid_to_idx]

                if len(train_idx) < 2 or len(test_idx) < 1:
                    logger.warning(
                        "Fold %d: insufficient participants (train=%d, test=%d), skipping",
                        fold_idx, len(train_idx), len(test_idx),
                    )
                    continue

                X_train, y_train = X_all[train_idx], y_all[train_idx]
                X_test, y_test = X_all[test_idx], y_all[test_idx]
                eval_pids = [pids_all[idx] for idx in test_idx]

                # StandardScaler fit strictly on outer training fold
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                for seed in seeds:
                    # ── Logistic Regression (inner-CV C) ──
                    best_c, _ = select_best_logreg_c(
                        X_train=X_train,
                        y_train=y_train,
                        c_grid=c_grid,
                        n_inner_folds=inner_cv_folds,
                        scoring=inner_cv_scoring,
                        seed=seed,
                        max_iter=lr_max_iter,
                        solver=lr_solver,
                        class_weight=class_weight,
                    )
                    lr_model = LogisticRegression(
                        class_weight=class_weight,
                        max_iter=lr_max_iter,
                        C=best_c,
                        random_state=seed,
                        solver=lr_solver,
                    )
                    lr_model.fit(X_train_s, y_train)
                    lr_preds = lr_model.predict(X_test_s)
                    lr_probs = lr_model.predict_proba(X_test_s)[:, 1]
                    lr_metrics = _compute_metrics(y_test, lr_preds, lr_probs)

                    run_rows.append({
                        "aggregation": aggregation,
                        "modality": mod_name,
                        "model": "logreg",
                        "fold": fold_idx,
                        "seed": seed,
                        **lr_metrics,
                    })
                    for pid, yt, yp, ypr in zip(eval_pids, y_test, lr_preds, lr_probs):
                        pred_rows.append({
                            "participant_id": pid,
                            "fold": fold_idx,
                            "seed": seed,
                            "aggregation": aggregation,
                            "modality": mod_name,
                            "model": "logreg",
                            "y_true": int(yt),
                            "sds_score": pid_to_sds.get(pid, float("nan")),
                            "y_prob": float(ypr),
                            "y_pred": int(yp),
                            "correct": int(yp == yt),
                        })

                    # ── MLP (fair comparison head) ──
                    mlp_model = _train_mlp(X_train_s, y_train, mlp_cfg, seed)
                    mlp_preds, mlp_probs = _predict_mlp(mlp_model, X_test_s)
                    mlp_metrics = _compute_metrics(y_test, mlp_preds, mlp_probs)

                    run_rows.append({
                        "aggregation": aggregation,
                        "modality": mod_name,
                        "model": "mlp",
                        "fold": fold_idx,
                        "seed": seed,
                        **mlp_metrics,
                    })
                    for pid, yt, yp, ypr in zip(eval_pids, y_test, mlp_preds, mlp_probs):
                        pred_rows.append({
                            "participant_id": pid,
                            "fold": fold_idx,
                            "seed": seed,
                            "aggregation": aggregation,
                            "modality": mod_name,
                            "model": "mlp",
                            "y_true": int(yt),
                            "sds_score": pid_to_sds.get(pid, float("nan")),
                            "y_prob": float(ypr),
                            "y_pred": int(yp),
                            "correct": int(yp == yt),
                        })

                    logger.info(
                        "  [%s/%s] Fold %d / seed %d — LR(C=%g): macF1=%.3f depF1=%.3f "
                        "acc=%.3f auc=%.3f | MLP: macF1=%.3f depF1=%.3f acc=%.3f auc=%.3f",
                        aggregation, mod_name, fold_idx, seed, best_c,
                        lr_metrics["macro_f1"], lr_metrics["dep_f1"],
                        lr_metrics["accuracy"], lr_metrics["auc_roc"],
                        mlp_metrics["macro_f1"], mlp_metrics["dep_f1"],
                        mlp_metrics["accuracy"], mlp_metrics["auc_roc"],
                    )

    return pd.DataFrame(run_rows), pd.DataFrame(pred_rows)


# ── Comparison table & plotting ──────────────────────────────────────

def _load_numeric_baseline_runs(csv_path: str) -> pd.DataFrame:
    """Load baseline CSV and keep only numeric per-(fold, seed) rows."""
    df = pd.read_csv(csv_path)
    df = df[df["fold"].astype(str).str.isdigit()].copy()
    df["fold"] = df["fold"].astype(int)
    df["seed"] = df["seed"].astype(int)
    for col in METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _summarize_group(
    df: pd.DataFrame,
    role: str,
    source: str,
    aggregation: str,
    modality: str,
    model: str,
) -> dict:
    """Compute mean, total std (fold x seed), and mean within-fold seed std."""
    row = {
        "role": role,
        "source": source,
        "aggregation": aggregation,
        "modality": modality,
        "model": model,
        "n_runs": len(df),
    }
    has_seed = "seed" in df.columns and df["seed"].nunique() > 1
    for col in METRIC_COLS:
        vals = df[col].dropna()
        row[f"{col}_mean"] = float(vals.mean())
        row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if has_seed:
            per_fold_seed_std = df.groupby("fold")[col].std(ddof=1).dropna()
            row[f"{col}_seed_std"] = float(per_fold_seed_std.mean()) if not per_fold_seed_std.empty else 0.0
        else:
            row[f"{col}_seed_std"] = 0.0
    return row


def build_comparison_table(
    fusion_runs_df: pd.DataFrame,
    config_path: str = "configs/eatd_fusion.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine majority baseline, v2 single-modality baselines, and Phase 3 fusion.

    Uses the primary aggregation (``"mean"``) consistently across text-only,
    audio-only, and primary fused rows, with ``"concat"`` fusion reported as
    secondary rows.

    Returns:
        comparison_df: Aggregated comparison table with mean, total std, and seed std.
        primary_plot_df: Long-format numeric runs (primary aggregation only) for ``_plot_barplot``.
    """
    cfg = load_config(config_path)
    primary_agg = cfg.get("features", {}).get("primary_aggregation", "mean")
    baselines_cfg = cfg.get("baselines", {})

    summary_rows = []
    plot_frames = []

    # 1. Majority-class baseline
    maj_df = compute_majority_baseline(config_path)
    summary_rows.append(
        _summarize_group(maj_df, "primary_baseline", "majority_baseline", "-", "majority", "always_0")
    )

    # 2. Single-modality baselines using the SAME primary aggregation ("mean" = v2)
    v2_path = baselines_cfg.get("v2_csv")
    if v2_path and Path(v2_path).exists():
        bdf = _load_numeric_baseline_runs(v2_path)
        for (mod, mdl), grp in bdf.groupby(["modality", "model"], sort=False):
            summary_rows.append(_summarize_group(grp, "primary", "v2_baseline", primary_agg, mod, mdl))
        plot_frames.append(bdf[["modality", "model", "fold", *METRIC_COLS]])
    else:
        logger.warning("Primary baseline CSV not found (v2_csv=%s)", v2_path)

    # 3. Late-fusion results: primary ("mean") first, then secondary ("concat")
    if not fusion_runs_df.empty:
        for agg in [primary_agg] + [
            a for a in fusion_runs_df["aggregation"].unique() if a != primary_agg
        ]:
            sub = fusion_runs_df[fusion_runs_df["aggregation"] == agg]
            role = "primary" if agg == primary_agg else "secondary"
            for (mod, mdl), grp in sub.groupby(["modality", "model"], sort=False):
                summary_rows.append(_summarize_group(grp, role, "v3_fusion", agg, mod, mdl))

            if agg == primary_agg:
                plot_frames.append(sub[["modality", "model", "fold", *METRIC_COLS]])

    comparison_df = pd.DataFrame(summary_rows)
    primary_plot_df = pd.concat(plot_frames, ignore_index=True) if plot_frames else pd.DataFrame()
    return comparison_df, primary_plot_df


def save_fusion_outputs(
    runs_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    config_path: str = "configs/eatd_fusion.yaml",
) -> pd.DataFrame:
    """Save fusion runs, out-of-fold predictions, comparison CSV, and comparison plot."""
    cfg = load_config(config_path)
    feat_cfg = cfg.get("features", {})
    train_cfg = cfg["training"]
    primary_agg = feat_cfg.get("primary_aggregation", "mean")

    runs_csv = train_cfg["results_runs_csv"]
    preds_csv = train_cfg["results_predictions_csv"]
    comp_csv = train_cfg["results_comparison_csv"]
    plot_path = train_cfg["results_plot"]

    for p in (runs_csv, preds_csv, comp_csv, plot_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    runs_df.to_csv(runs_csv, index=False)
    logger.info("Saved raw fusion runs to %s (%d rows)", runs_csv, len(runs_df))

    preds_df.to_csv(preds_csv, index=False)
    logger.info("Saved out-of-fold predictions to %s (%d rows)", preds_csv, len(preds_df))

    comparison_df, plot_df = build_comparison_table(runs_df, config_path=config_path)
    comparison_df.to_csv(comp_csv, index=False)
    logger.info("Saved unified comparison table to %s (%d rows)", comp_csv, len(comparison_df))

    if not plot_df.empty:
        _plot_barplot(
            plot_df,
            plot_path,
            title=f"EATD-Corpus Comparison ({primary_agg} embeddings): Single-Modality vs. Late Fusion (5-fold CV × 3 seeds)",
        )

    return comparison_df


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train EATD late-fusion models.")
    parser.add_argument("--config", default="configs/eatd_fusion.yaml")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick test on 10 participants, fold 0, seed 42 (does not write output CSVs)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    smoke_pids = None
    if args.smoke_test:
        smoke_pids = _get_smoke_test_pids(args.config)
        logger.info("Smoke test mode with participants: %s", smoke_pids)

    runs_df, preds_df = run_fusion(
        config_path=args.config,
        smoke_test=args.smoke_test,
        smoke_pids=smoke_pids,
    )

    if runs_df.empty:
        logger.error("No fusion results produced — check cached embeddings.")
        return

    if args.smoke_test:
        print("\n" + "=" * 90)
        print("PHASE 3 SMOKE TEST RESULTS (10 participants, fold 0, seed 42 — CSVs unchanged)")
        print("=" * 90)
        print(runs_df.to_string(index=False))
        print(f"\nOut-of-fold smoke predictions shape: {preds_df.shape}")
        print(preds_df.head(8).to_string(index=False))
        print("=" * 90 + "\n")
        return

    comparison_df = save_fusion_outputs(runs_df, preds_df, config_path=args.config)
    print("\n" + "=" * 100)
    print("EATD-CORPUS COMPARISON: Single-Modality vs. Late Fusion (mean ± std, plus seed_std)")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

