"""Unit tests for Phase 5 DAIC-WOZ evaluation, majority voting, and multi-task training."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch


def test_two_stage_majority_vote_and_ties():
    """Verify within-scale vote (with prob tie-break) and cross-scale 3-scale majority vote."""
    from src.models.daic_eval import aggregate_participant_predictions

    rows = []
    # Session 100: scale 30s -> 3/4 positive (1); scale 75s -> 2/4 tie with prob mean 0.65 (1);
    # scale 120s -> 1/4 positive (0) => cross-scale votes 1+1+0 = 2/3 -> final 1
    probs_30 = [0.8, 0.7, 0.6, 0.2]
    probs_75 = [0.9, 0.9, 0.4, 0.4]  # 2 >= 0.5, 2 < 0.5 -> tie; mean = 0.65 -> 1
    probs_120 = [0.7, 0.1, 0.1, 0.1]
    for scale, plist in [(30, probs_30), (75, probs_75), (120, probs_120)]:
        for p in plist:
            rows.append(
                {
                    "session_id": 100,
                    "scale": scale,
                    "y_true": 1.0,
                    "phq8_true": 14.0,
                    "y_prob": p,
                    "phq8_pred": 12.0,
                }
            )

    # Session 200: 0 across all 3 scales, unlabeled test (NaN)
    for scale in (30, 75, 120):
        for p in (0.1, 0.2):
            rows.append(
                {
                    "session_id": 200,
                    "scale": scale,
                    "y_true": math.nan,
                    "phq8_true": math.nan,
                    "y_prob": p,
                    "phq8_pred": 3.0,
                }
            )

    p_df, scale_dfs = aggregate_participant_predictions(pd.DataFrame(rows))
    assert set(scale_dfs.keys()) == {30, 75, 120}

    s100 = p_df[p_df["session_id"] == 100].iloc[0]
    assert s100["pred_30s"] == 1
    assert s100["pred_75s"] == 1
    assert s100["pred_120s"] == 0
    assert s100["scale_votes_sum"] == 2
    assert s100["y_pred"] == 1
    assert np.isclose(s100["phq8_pred"], 12.0)

    s200 = p_df[p_df["session_id"] == 200].iloc[0]
    assert s200["y_pred"] == 0
    assert math.isnan(s200["y_true"])
    assert math.isnan(s200["phq8_true"])


def test_bootstrap_participant_ci_brackets_mean():
    """Stratified participant bootstrap 95% CIs bracket point estimate cleanly."""
    from src.models.daic_eval import bootstrap_participant_ci

    rng = np.random.default_rng(0)
    seed_dfs = []
    y_true = np.array([1] * 10 + [0] * 20, dtype=float)
    phq_true = np.where(y_true == 1, 15.0, 3.0)

    for s in range(3):
        probs = np.clip(np.where(y_true == 1, 0.75, 0.25) + rng.normal(0, 0.12, size=30), 0.01, 0.99)
        preds = (probs >= 0.5).astype(int)
        phq_pred = np.clip(phq_true + rng.normal(0, 2.0, size=30), 0.0, 24.0)
        seed_dfs.append(
            pd.DataFrame(
                {
                    "session_id": np.arange(300, 330),
                    "y_true": y_true,
                    "phq8_true": phq_true,
                    "y_pred": preds,
                    "y_prob": probs,
                    "phq8_pred": phq_pred,
                }
            )
        )

    summary = bootstrap_participant_ci(seed_dfs, n_bootstrap=300, seed=42)
    for m in ("macro_f1", "dep_f1", "accuracy", "auc_roc", "phq8_rmse"):
        assert summary[m]["ci_low"] <= summary[m]["mean"] <= summary[m]["ci_high"]
        assert summary[m]["seed_std"] >= 0.0


def test_verify_no_split_overlap_raises():
    from src.models.train_daic import verify_no_split_overlap

    verify_no_split_overlap([301, 302], [303, 304])
    with pytest.raises(ValueError, match="leakage"):
        verify_no_split_overlap([301, 302], [302, 305])


def test_train_single_seed_multitask_end_to_end():
    """MultiTaskDepMLP trains with scaler fit on train only and outputs valid participant preds."""
    from src.models.train_daic import train_single_seed

    rng = np.random.default_rng(7)
    n_tr, n_dv, d = 60, 36, 16

    # Synthetic separable train & dev windows across 4 train and 2 dev sessions, 3 scales
    w_tr = pd.DataFrame(
        {
            "session_id": np.repeat([1, 2, 3, 4], 15),
            "window_id": [f"tr_{i}" for i in range(n_tr)],
            "scale": np.tile([30, 75, 120], 20),
            "y_true": np.repeat([1.0, 1.0, 0.0, 0.0], 15),
            "phq8_true": np.repeat([16.0, 14.0, 2.0, 3.0], 15),
        }
    )
    w_dv = pd.DataFrame(
        {
            "session_id": np.repeat([101, 102], 18),
            "window_id": [f"dv_{i}" for i in range(n_dv)],
            "scale": np.tile([30, 75, 120], 12),
            "y_true": np.repeat([1.0, 0.0], 18),
            "phq8_true": np.repeat([18.0, 1.0], 18),
        }
    )

    X_tr = rng.normal(0.0, 1.0, size=(n_tr, d)).astype(np.float32)
    X_tr[w_tr["y_true"] == 1.0, :4] += 2.0
    X_dv = rng.normal(0.0, 1.0, size=(n_dv, d)).astype(np.float32)
    X_dv[w_dv["y_true"] == 1.0, :4] += 2.0

    mlp_cfg = {
        "hidden_dim": 32,
        "dropout": 0.1,
        "epochs": 12,
        "patience": 5,
        "batch_size": 16,
        "lr": 1e-2,
        "weight_decay": 1e-3,
        "reg_loss_weight": 0.25,
    }
    out = train_single_seed(X_tr, w_tr, X_dv, w_dv, mlp_cfg=mlp_cfg, seed=42)
    p_df = out["dev_participant_df"]
    assert len(p_df) == 2
    assert set(p_df["session_id"]) == {101, 102}
    assert out["best_epoch"] >= 1
    assert np.all((p_df["phq8_pred"] >= 0.0) & (p_df["phq8_pred"] <= 24.0))

