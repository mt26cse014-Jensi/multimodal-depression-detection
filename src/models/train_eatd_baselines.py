"""Train and evaluate EATD-Corpus single-modality baselines.

Models:
  - Logistic Regression (sklearn, class-weighted, fixed C or inner-CV tuned C)
  - MLP (1 hidden layer, weight decay, fixed epochs, class-weighted)

Protocol:
  - 5-fold CV from folds.json (participant-level splits)
  - 3 random seeds per fold
  - StandardScaler fit on train only
  - Optional stratified inner CV inside each outer training fold for LogReg C
  - Test fold NEVER used for tuning
  - Metrics: macro F1, depressed-class F1, accuracy, AUC-ROC

Usage:
    python -m src.models.train_eatd_baselines                                  # v1 config
    python -m src.models.train_eatd_baselines --config configs/eatd_baselines_v2.yaml
    python -m src.models.train_eatd_baselines --config configs/eatd_baselines_v2.yaml --smoke-test
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────

def load_config(config_path: str = "configs/eatd_preprocess.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Logistic Regression inner-CV tuning ──────────────────────────────

def select_best_logreg_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: Sequence[float],
    n_inner_folds: int = 5,
    scoring: str = "roc_auc",
    seed: int = 42,
    max_iter: int = 2000,
    solver: str = "lbfgs",
    class_weight: Optional[str] = "balanced",
) -> tuple[float, dict[float, float]]:
    """Select Logistic Regression regularisation strength C via stratified inner CV.

    Fits ``StandardScaler`` strictly inside each inner training split so there
    is zero scaling or label leakage from either inner validation or outer test sets.
    Ties are broken toward smaller ``C`` (stronger regularisation).

    Args:
        X_train: Unscaled outer training features of shape ``(n_train, D)``.
        y_train: Binary outer training labels of shape ``(n_train,)``.
        c_grid: Candidate regularisation strengths ``C``.
        n_inner_folds: Number of stratified inner CV folds.
        scoring: Inner CV selection metric (``"roc_auc"`` or ``"macro_f1"``).
        seed: Random seed for inner fold shuffling and solver.
        max_iter: Maximum solver iterations.
        solver: LogisticRegression solver name.
        class_weight: Class weighting passed to LogisticRegression.

    Returns:
        best_c: Selected regularisation strength ``C``.
        mean_scores: Mapping from each candidate ``C`` to mean inner-CV score.
    """
    c_sorted = sorted(float(c) for c in c_grid)
    if not c_sorted:
        raise ValueError("c_grid must not be empty")

    unique_classes, counts = np.unique(y_train, return_counts=True)
    if len(unique_classes) < 2:
        logger.warning("Only one class in outer train fold; defaulting C=1.0")
        return 1.0, {}

    actual_splits = min(int(n_inner_folds), int(counts.min()))
    if actual_splits < 2:
        logger.warning(
            "Too few minority samples (%d) for inner CV; defaulting C=1.0",
            int(counts.min()),
        )
        return (1.0 if 1.0 in c_sorted else c_sorted[len(c_sorted) // 2]), {}

    if actual_splits < n_inner_folds:
        logger.warning(
            "Reducing inner CV folds from %d to %d due to minority class count (%d)",
            n_inner_folds, actual_splits, int(counts.min()),
        )

    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X_train, y_train))

    mean_scores: dict[float, float] = {}
    best_c = c_sorted[0]
    best_score = -np.inf

    for c in c_sorted:
        fold_scores = []
        for inner_tr, inner_val in splits:
            scaler = StandardScaler()
            X_in_tr = scaler.fit_transform(X_train[inner_tr])
            X_in_val = scaler.transform(X_train[inner_val])
            y_in_tr = y_train[inner_tr]
            y_in_val = y_train[inner_val]

            clf = LogisticRegression(
                class_weight=class_weight,
                max_iter=max_iter,
                C=c,
                random_state=seed,
                solver=solver,
            )
            clf.fit(X_in_tr, y_in_tr)

            if scoring == "roc_auc":
                if len(np.unique(y_in_val)) > 1:
                    probs = clf.predict_proba(X_in_val)[:, 1]
                    score = float(roc_auc_score(y_in_val, probs))
                else:
                    score = float("nan")
            elif scoring == "macro_f1":
                preds = clf.predict(X_in_val)
                score = float(f1_score(y_in_val, preds, average="macro", zero_division=0))
            else:
                raise ValueError(f"Unsupported inner CV scoring: {scoring!r}")

            fold_scores.append(score)

        valid_scores = [s for s in fold_scores if not np.isnan(s)]
        mean_score = float(np.mean(valid_scores)) if valid_scores else float("nan")
        mean_scores[c] = mean_score

        # Strict '>' ensures ties break toward smaller C (stronger regularisation)
        if not np.isnan(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_c = c

    return best_c, mean_scores


# ── MLP model ────────────────────────────────────────────────────────

class DepMLP(nn.Module):
    """Single hidden-layer MLP for binary depression classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (no sigmoid)."""
        return self.net(x).squeeze(-1)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mlp_cfg: dict,
    seed: int,
) -> DepMLP:
    """Train MLP for a fixed number of epochs with weight decay."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    input_dim = X_train.shape[1]

    model = DepMLP(
        input_dim=input_dim,
        hidden_dim=mlp_cfg["hidden_dim"],
        dropout=mlp_cfg["dropout"],
    ).to(device)

    # Class-weighted loss: pos_weight = n_neg / n_pos
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=mlp_cfg["lr"],
        weight_decay=mlp_cfg["weight_decay"],
    )

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    batch_size = mlp_cfg.get("batch_size", 32)
    n_epochs = mlp_cfg["epochs"]
    n_samples = len(X_t)

    model.train()
    for _epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        X_t = X_t[perm]
        y_t = y_t[perm]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xb = X_t[start:end]
            yb = y_t[start:end]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def _predict_mlp(model: DepMLP, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and probabilities from a trained MLP.

    Returns:
        preds: (N,) binary predictions
        probs: (N,) probability of class 1
    """
    device = torch.device("cpu")
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return preds, probs


# ── Evaluation ───────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all required metrics."""
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "dep_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    # AUC-ROC requires both classes present
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc_roc"] = float("nan")
        logger.warning("Only one class in y_true — AUC-ROC is NaN")
    return metrics


def compute_majority_baseline(
    config_path: str = "configs/eatd_preprocess.yaml",
) -> pd.DataFrame:
    """Evaluate majority-class baseline (always predict 0 / not depressed).

    Args:
        config_path: Path to YAML config specifying manifest and folds.

    Returns:
        DataFrame with one row per fold containing fold metrics.
    """
    cfg = load_config(config_path)
    manifest = pd.read_csv(cfg["dataset"]["manifest_path"])
    pid_to_label = dict(zip(manifest["participant_id"], manifest["label"].astype(int)))

    with open(cfg["dataset"]["folds_path"], "r", encoding="utf-8") as f:
        folds = json.load(f)["folds"]

    rows = []
    for fold_info in folds:
        test_pids = fold_info["test"]
        y_true = np.array([pid_to_label[p] for p in test_pids], dtype=np.int64)
        y_pred = np.zeros_like(y_true)
        y_prob = np.zeros(len(y_true), dtype=np.float64)
        metrics = _compute_metrics(y_true, y_pred, y_prob)
        rows.append({"fold": fold_info["fold"], **metrics})

    return pd.DataFrame(rows)


# ── Main training loop ───────────────────────────────────────────────

def run_baselines(
    config_path: str = "configs/eatd_preprocess.yaml",
    smoke_test: bool = False,
    smoke_pids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run all baseline experiments and return results DataFrame.

    Args:
        config_path: Path to config YAML.
        smoke_test: If True, only run fold 0, seed 42, on smoke_pids.
        smoke_pids: Participant IDs for smoke test.

    Returns:
        DataFrame with per-run rows (and ``attrs["best_c_df"]`` when inner CV is used).
    """
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    mlp_cfg = train_cfg["mlp"]
    logreg_cfg = train_cfg.get("logreg", {})
    seeds = train_cfg["seeds"]
    class_weight = train_cfg.get("class_weight", "balanced")
    aggregation = cfg.get("features", {}).get("aggregation", "concat")

    use_inner_cv = "c_grid" in logreg_cfg and bool(logreg_cfg["c_grid"])
    c_grid = logreg_cfg.get("c_grid", [1.0])
    inner_cv_folds = int(logreg_cfg.get("inner_cv_folds", 5))
    inner_cv_scoring = logreg_cfg.get("inner_cv_scoring", "roc_auc")
    lr_max_iter = int(logreg_cfg.get("max_iter", 2000 if use_inner_cv else 1000))
    lr_solver = logreg_cfg.get("solver", "lbfgs")

    # Load folds
    with open(cfg["dataset"]["folds_path"], "r", encoding="utf-8") as f:
        folds_data = json.load(f)
    folds = folds_data["folds"]

    # Import embedding loaders
    from src.features.extract_text import load_text_embeddings
    from src.features.extract_audio import load_audio_embeddings

    # Define modality configs: (name, loader_func, loader_kwargs)
    audio_cfg = cfg["features"]["audio"]
    modalities = [
        ("text", load_text_embeddings, {"config_path": config_path, "aggregation": aggregation}),
    ]

    for key_name in ("model_name", "alt_model_name"):
        if key_name in audio_cfg:
            model_name = audio_cfg[key_name]
            short = model_name.split("/")[-1]
            modalities.append((
                f"audio_{short}",
                load_audio_embeddings,
                {"config_path": config_path, "encoder_key": key_name, "aggregation": aggregation},
            ))

    if smoke_test:
        folds = [folds[0]]
        seeds = [seeds[0]]

    results = []
    best_c_records = []

    for mod_name, loader, loader_kwargs in modalities:
        logger.info("=== Modality: %s (aggregation=%s) ===", mod_name, aggregation)

        try:
            X_all, y_all, pids_all = loader(**loader_kwargs)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", mod_name, e)
            continue

        pid_to_idx = {pid: i for i, pid in enumerate(pids_all)}

        for fold_info in folds:
            fold_idx = fold_info["fold"]
            train_pids = fold_info["train"]
            test_pids = fold_info["test"]

            # Filter to available participants (for smoke test)
            if smoke_pids is not None:
                smoke_set = set(smoke_pids)
                train_pids = [p for p in train_pids if p in smoke_set and p in pid_to_idx]
                test_pids = [p for p in test_pids if p in smoke_set and p in pid_to_idx]
                if len(train_pids) < 2 or len(test_pids) < 1:
                    available = [p for p in smoke_pids if p in pid_to_idx]
                    if len(available) < 4:
                        logger.warning("Not enough participants for smoke test")
                        continue
                    split_pt = max(2, int(round(len(available) * 0.7)))
                    train_pids = available[:split_pt]
                    test_pids = available[split_pt:]

            train_idx = [pid_to_idx[p] for p in train_pids if p in pid_to_idx]
            test_idx = [pid_to_idx[p] for p in test_pids if p in pid_to_idx]

            if len(train_idx) < 2 or len(test_idx) < 1:
                logger.warning(
                    "Fold %d: not enough participants (train=%d, test=%d), skipping",
                    fold_idx, len(train_idx), len(test_idx),
                )
                continue

            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_test, y_test = X_all[test_idx], y_all[test_idx]

            # StandardScaler fit on outer train only
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            for seed in seeds:
                # ── Logistic Regression ──
                if use_inner_cv:
                    best_c, inner_scores = select_best_logreg_c(
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
                    best_c_records.append({
                        "modality": mod_name,
                        "fold": fold_idx,
                        "seed": seed,
                        "best_c": best_c,
                        "best_inner_score": inner_scores.get(best_c, float("nan")),
                    })
                else:
                    best_c = float(logreg_cfg.get("c", 1.0))

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

                results.append({
                    "modality": mod_name,
                    "model": "logreg",
                    "fold": fold_idx,
                    "seed": seed,
                    **lr_metrics,
                })

                # ── MLP ──
                mlp_model = _train_mlp(X_train_s, y_train, mlp_cfg, seed)
                mlp_preds, mlp_probs = _predict_mlp(mlp_model, X_test_s)
                mlp_metrics = _compute_metrics(y_test, mlp_preds, mlp_probs)

                results.append({
                    "modality": mod_name,
                    "model": "mlp",
                    "fold": fold_idx,
                    "seed": seed,
                    **mlp_metrics,
                })

                logger.info(
                    "  Fold %d / seed %d — LR(C=%g): macF1=%.3f depF1=%.3f acc=%.3f auc=%.3f | "
                    "MLP: macF1=%.3f depF1=%.3f acc=%.3f auc=%.3f",
                    fold_idx, seed, best_c,
                    lr_metrics["macro_f1"], lr_metrics["dep_f1"],
                    lr_metrics["accuracy"], lr_metrics["auc_roc"],
                    mlp_metrics["macro_f1"], mlp_metrics["dep_f1"],
                    mlp_metrics["accuracy"], mlp_metrics["auc_roc"],
                )

    df = pd.DataFrame(results)
    df.attrs["best_c_df"] = pd.DataFrame(best_c_records)
    return df


# ── Aggregation and output ───────────────────────────────────────────

def _add_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add mean ± std summary rows per (modality, model) group."""
    metric_cols = ["macro_f1", "dep_f1", "accuracy", "auc_roc"]
    summaries = []

    for (mod, mdl), group in df.groupby(["modality", "model"]):
        row = {"modality": mod, "model": mdl, "fold": "mean", "seed": "-"}
        for col in metric_cols:
            vals = group[col].dropna()
            row[col] = f"{vals.mean():.4f} +/- {vals.std():.4f}"
        summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    return pd.concat([df, summary_df], ignore_index=True)


def _summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return formatted mean +/- std summary DataFrame for console reporting."""
    metric_cols = ["macro_f1", "dep_f1", "accuracy", "auc_roc"]
    rows = []
    for (mod, mdl), group in df.groupby(["modality", "model"]):
        row = {"modality": mod, "model": mdl}
        for col in metric_cols:
            vals = group[col].dropna()
            row[col] = f"{vals.mean():.4f} +/- {vals.std():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_barplot(df: pd.DataFrame, output_path: str, title: Optional[str] = None) -> None:
    """Create a grouped bar chart of mean metrics with std error bars."""
    metric_cols = ["macro_f1", "dep_f1", "accuracy", "auc_roc"]
    metric_labels = ["Macro F1", "Dep F1", "Accuracy", "AUC-ROC"]

    # Filter to numeric rows only
    df_num = df[df["fold"] != "mean"].copy()
    for col in metric_cols:
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

    groups = df_num.groupby(["modality", "model"])
    group_names = []
    means_dict = {m: [] for m in metric_cols}
    stds_dict = {m: [] for m in metric_cols}

    for (mod, mdl), group in groups:
        group_names.append(f"{mod}\n{mdl}")
        for col in metric_cols:
            vals = group[col].dropna()
            means_dict[col].append(vals.mean())
            stds_dict[col].append(vals.std())

    n_groups = len(group_names)
    n_metrics = len(metric_cols)
    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 2.5), 6))
    colors = plt.cm.Set2.colors

    for j, (col, label) in enumerate(zip(metric_cols, metric_labels)):
        offset = (j - n_metrics / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means_dict[col],
            width,
            yerr=stds_dict[col],
            label=label,
            color=colors[j % len(colors)],
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_ylabel("Score")
    ax.set_title(title or "EATD-Corpus Single-Modality Baselines (5-fold CV × 3 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=9)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved bar chart to %s", output_path)


def save_results(
    df: pd.DataFrame,
    config_path: str = "configs/eatd_preprocess.yaml",
) -> None:
    """Save results CSV and bar chart."""
    cfg = load_config(config_path)
    train_cfg = cfg["training"]

    results_csv = train_cfg["results_csv"]
    results_plot = train_cfg["results_plot"]
    include_summary_rows = train_cfg.get("include_summary_rows", True)

    df_to_save = _add_summary_rows(df) if include_summary_rows else df.copy()

    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_csv(results_csv, index=False)
    logger.info(
        "Saved results to %s (%d rows, include_summary_rows=%s)",
        results_csv, len(df_to_save), include_summary_rows,
    )

    best_c_df = df.attrs.get("best_c_df")
    if isinstance(best_c_df, pd.DataFrame) and not best_c_df.empty:
        best_c_path = train_cfg.get("results_best_c_csv")
        if best_c_path:
            Path(best_c_path).parent.mkdir(parents=True, exist_ok=True)
            best_c_df.to_csv(best_c_path, index=False)
            logger.info("Saved selected C values to %s (%d rows)", best_c_path, len(best_c_df))

    # Print summary
    summary = _summarize_df(df)
    print("\n" + "=" * 80)
    print("SUMMARY: Mean ± Std across 5 folds × 3 seeds")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80 + "\n")

    # Bar chart
    aggregation = cfg.get("features", {}).get("aggregation", "concat")
    plot_title = (
        f"EATD-Corpus Single-Modality Baselines ({aggregation} embeddings, 5-fold CV × 3 seeds)"
    )
    _plot_barplot(df, results_plot, title=plot_title)


# ── CLI entry point ──────────────────────────────────────────────────

def _get_smoke_test_pids(config_path: str) -> list[str]:
    """Return 10 stratified smoke-test participant IDs spanning fold 0 train and test.

    Selects 3 depressed + 3 control from fold 0 train (6 train, enabling 3-fold inner CV)
    and 2 depressed + 2 control from fold 0 test (4 test, ensuring both classes in test).
    """
    cfg = load_config(config_path)
    manifest = pd.read_csv(cfg["dataset"]["manifest_path"])
    with open(cfg["dataset"]["folds_path"], "r", encoding="utf-8") as f:
        folds = json.load(f)

    fold0_train = set(folds["folds"][0]["train"])
    fold0_test = set(folds["folds"][0]["test"])

    df_train = manifest[manifest.participant_id.isin(fold0_train)]
    df_test = manifest[manifest.participant_id.isin(fold0_test)]

    tr_dep = df_train[df_train.label == 1].participant_id.tolist()[:3]
    tr_ctrl = df_train[df_train.label == 0].participant_id.tolist()[:3]
    te_dep = df_test[df_test.label == 1].participant_id.tolist()[:2]
    te_ctrl = df_test[df_test.label == 0].participant_id.tolist()[:2]
    return tr_dep + tr_ctrl + te_dep + te_ctrl


def main():
    parser = argparse.ArgumentParser(description="Train EATD baseline classifiers.")
    parser.add_argument("--config", default="configs/eatd_preprocess.yaml")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick test on 10 participants, fold 0, seed 42 (does not write results CSV)",
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

    df = run_baselines(
        config_path=args.config,
        smoke_test=args.smoke_test,
        smoke_pids=smoke_pids,
    )

    if df.empty:
        logger.error("No results — check that embeddings are cached.")
        return

    if args.smoke_test:
        print("\n" + "=" * 80)
        print("SMOKE TEST RESULTS (10 participants, fold 0, seed 42 — results CSV unchanged)")
        print("=" * 80)
        print(df.to_string(index=False))
        best_c_df = df.attrs.get("best_c_df")
        if isinstance(best_c_df, pd.DataFrame) and not best_c_df.empty:
            print("\nSelected C (inner CV):")
            print(best_c_df.to_string(index=False))
        print("=" * 80 + "\n")
        return

    save_results(df, config_path=args.config)


if __name__ == "__main__":
    main()
