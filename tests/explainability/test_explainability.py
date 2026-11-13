"""
Unit tests for Phase 7 Explainability Modules.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.explainability.shap_analysis import compute_exact_2group_shap
from src.explainability.shortcut_analysis import analyze_interview_temporal_shortcuts
from src.explainability.turn_occlusion import occlude_session_turns
from src.models.cross_modal_attention import SmallCrossModalModel


def test_exact_2group_shap_efficiency() -> None:
    """Verify that exact 2-group Shapley values strictly satisfy the efficiency axiom."""
    torch.manual_seed(42)
    np.random.seed(42)

    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2304, 1)

        def forward(self, x):
            return self.linear(x), self.linear(x)

    model = DummyMLP()
    model.eval()

    X_bg = np.random.randn(20, 2304).astype(np.float32)
    X_query = np.random.randn(5, 2304).astype(np.float32)

    phi_text, phi_audio, base_val = compute_exact_2group_shap(model, X_query, X_bg, text_dim=768)

    assert phi_text.shape == (5,)
    assert phi_audio.shape == (5,)

    with torch.no_grad():
        preds = torch.sigmoid(model(torch.tensor(X_query))[0]).squeeze(-1).numpy()

    deltas = preds - base_val
    shap_sums = phi_text + phi_audio

    np.testing.assert_allclose(shap_sums, deltas, atol=1e-5)


def test_turn_occlusion_scoring() -> None:
    """Verify that turn occlusion removes each turn and computes Delta_P correctly."""
    torch.manual_seed(42)
    model = SmallCrossModalModel(n_layers=1, d_in_text=768, d_in_audio=768, d_model=32, hidden_dim=32)
    model.eval()

    n_turns = 4
    n_windows = 6
    l_t = 5
    l_a = 10

    t_pad = torch.randn(n_turns, l_t, 768)
    a_pad = torch.randn(n_turns, l_a, 768)
    t_mask = torch.zeros(n_turns, l_t, dtype=torch.bool)
    a_mask = torch.zeros(n_turns, l_a, dtype=torch.bool)

    w_mat = torch.softmax(torch.randn(n_windows, n_turns), dim=1)

    session_data = {
        "session_id": 999,
        "t_pad": t_pad,
        "a_pad": a_pad,
        "t_mask": t_mask,
        "a_mask": a_mask,
        "w_mat": w_mat,
        "turn_ids": np.array([10, 11, 12, 13]),
        "start_times": np.array([0.0, 10.0, 20.0, 30.0]),
        "stop_times": np.array([5.0, 15.0, 25.0, 35.0]),
        "turn_durations": np.array([5.0, 5.0, 5.0, 5.0]),
    }

    df = occlude_session_turns(session_data, model)

    assert len(df) == n_turns
    assert set(df["turn_id"]) == {10, 11, 12, 13}
    assert "abs_delta_p" in df.columns
    assert "rank_in_session" in df.columns
    assert list(df["rank_in_session"]) == [1, 2, 3, 4]
    assert (df["abs_delta_p"] >= 0.0).all()
    # Ensure no text columns exist
    assert "text" not in df.columns


def test_shortcut_analysis_statistics() -> None:
    """Verify statistical metrics and quartile bucketing for interview temporal analysis."""
    occlusion_df = pd.DataFrame({
        "session_id": [101] * 8,
        "turn_id": list(range(8)),
        "relative_position": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        "abs_delta_p": [0.01, 0.02, 0.03, 0.04, 0.10, 0.12, 0.15, 0.20],
        "rank_in_session": list(range(8, 0, -1)),
    })

    res = analyze_interview_temporal_shortcuts(occlusion_df)

    assert res["spearman_r"] > 0.9  # Perfectly monotonic increase
    assert res["spearman_p"] < 0.01
    assert res["second_half_mean_imp"] > res["first_half_mean_imp"]
    assert res["half_ratio_second_to_first"] > 1.0
    assert "Q1 (0-25%)" in res["quartile_means"]
    assert "Q4 (75-100%)" in res["quartile_means"]

