"""Unit tests for EATD baseline training utilities (Phase 2b)."""

import numpy as np
import yaml


def test_select_best_logreg_c_tie_breaks_smaller():
    """On linearly separable data where multiple C values tie at AUC=1.0, pick smallest C."""
    from src.models.train_eatd_baselines import select_best_logreg_c

    rng = np.random.default_rng(42)
    X_neg = rng.normal(-3.0, 0.5, size=(30, 8))
    X_pos = rng.normal(3.0, 0.5, size=(15, 8))
    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * 30 + [1] * 15)

    c_grid = [100.0, 1.0, 0.01, 10.0, 0.1]
    best_c, scores = select_best_logreg_c(X, y, c_grid=c_grid, n_inner_folds=5, seed=42)

    assert len(scores) == len(c_grid)
    assert scores[0.01] == 1.0
    assert best_c == 0.01


def test_select_best_logreg_c_valid_range():
    """Selected C must come from the provided grid with finite inner-CV score."""
    from src.models.train_eatd_baselines import select_best_logreg_c

    rng = np.random.default_rng(7)
    X = rng.normal(0.0, 1.0, size=(50, 16))
    y = np.array([0] * 35 + [1] * 15)
    X[y == 1, 0] += 1.5

    c_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_c, scores = select_best_logreg_c(X, y, c_grid=c_grid, n_inner_folds=5, seed=42)

    assert best_c in c_grid
    assert np.isfinite(scores[best_c])


def test_majority_baseline_values():
    """Majority baseline predicts all 0s -> dep_f1=0.0, auc_roc=0.5, accuracy ~0.815."""
    from src.models.train_eatd_baselines import compute_majority_baseline

    df = compute_majority_baseline("configs/eatd_baselines_v2.yaml")
    assert len(df) == 5
    assert (df["dep_f1"] == 0.0).all()
    assert np.allclose(df["auc_roc"], 0.5)
    assert 0.80 < df["accuracy"].mean() < 0.83
    assert 0.44 < df["macro_f1"].mean() < 0.46


def test_v2_config_preserves_v1_paths():
    """Verify v2 config does not point to v1 result files and disables summary rows."""
    with open("configs/eatd_baselines_v2.yaml", "r", encoding="utf-8") as f:
        v2 = yaml.safe_load(f)

    assert v2["features"]["aggregation"] == "mean"
    assert v2["training"]["include_summary_rows"] is False
    assert v2["training"]["results_csv"] == "results/eatd/eatd_baselines_v2.csv"
    assert v2["training"]["results_csv"] != "results/eatd/eatd_baselines.csv"
    assert v2["training"]["results_plot"] != "results/eatd/eatd_baselines_barplot.png"

