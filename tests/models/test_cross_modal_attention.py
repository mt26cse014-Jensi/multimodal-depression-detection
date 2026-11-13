"""
Tests for Phase 6 Cross-Modal Attention Architecture.
"""

import pytest
import torch
import torch.nn as nn

from src.models.cross_modal_attention import (
    CrossModalAttentionBlock,
    SmallCrossModalModel,
)


def test_parameter_counts() -> None:
    """Verify that 1-layer and 2-layer CMA models match exact expected parameter counts."""
    m1 = SmallCrossModalModel(n_layers=1)
    m2 = SmallCrossModalModel(n_layers=2)

    p1 = m1.count_trainable_parameters()
    p2 = m2.count_trainable_parameters()

    assert p1 == 494_978, f"Expected 494,978 parameters for 1-layer CMA, got {p1}"
    assert p2 == 759_938, f"Expected 759,938 parameters for 2-layer CMA, got {p2}"


def test_cma_block_forward() -> None:
    """Verify forward pass of CrossModalAttentionBlock."""
    block = CrossModalAttentionBlock(d_model=64, n_heads=4, d_ff=128, dropout=0.0, attn_dropout=0.0)
    block.eval()

    t = torch.randn(2, 10, 64)
    a = torch.randn(2, 25, 64)

    t_out, a_out = block(t, a)
    assert t_out.shape == (2, 10, 64)
    assert a_out.shape == (2, 25, 64)


def test_forward_pass_and_gradient_flow() -> None:
    """Verify forward and backward passes through SmallCrossModalModel."""
    model = SmallCrossModalModel(n_layers=1, d_in_text=768, d_in_audio=768, d_model=128, hidden_dim=128)
    model.train()

    n_turns = 6
    n_windows = 8
    max_l_t = 12
    max_l_a = 30

    t = torch.randn(n_turns, max_l_t, 768)
    a = torch.randn(n_turns, max_l_a, 768)

    # Random masks
    t_mask = torch.zeros(n_turns, max_l_t, dtype=torch.bool)
    t_mask[:, 8:] = True  # last 4 tokens padded
    a_mask = torch.zeros(n_turns, max_l_a, dtype=torch.bool)
    a_mask[:, 20:] = True  # last 10 frames padded

    # Window turn matrix: random normalized rows
    w_mat = torch.softmax(torch.randn(n_windows, n_turns), dim=1)

    logits, phq_preds = model(t, a, w_mat, t_mask=t_mask, a_mask=a_mask)

    assert logits.shape == (n_windows, 1)
    assert phq_preds.shape == (n_windows, 1)

    # Backward pass
    target_labels = torch.randint(0, 2, (n_windows, 1)).float()
    target_phq = torch.rand(n_windows, 1) * 24.0

    bce_loss = nn.BCEWithLogitsLoss()(logits, target_labels)
    mse_loss = nn.MSELoss()(phq_preds, target_phq)
    total_loss = bce_loss + 0.1 * mse_loss

    total_loss.backward()

    # Check gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"


def test_duration_weighted_pooling_equivalence() -> None:
    """Verify that window matrix multiplication matches manual duration-weighted pooling."""
    n_turns = 4
    turn_embs = torch.randn(n_turns, 256)
    durations = [2.0, 3.0, 5.0, 1.0]

    # Window 0 uses turns 0 and 1: weights [2/5, 3/5, 0, 0]
    # Window 1 uses turns 1, 2, 3: weights [0, 3/9, 5/9, 1/9]
    w_mat = torch.tensor([
        [2.0 / 5.0, 3.0 / 5.0, 0.0, 0.0],
        [0.0, 3.0 / 9.0, 5.0 / 9.0, 1.0 / 9.0],
    ])

    pooled_mat = torch.matmul(w_mat, turn_embs)

    expected_w0 = (2.0 * turn_embs[0] + 3.0 * turn_embs[1]) / 5.0
    expected_w1 = (3.0 * turn_embs[1] + 5.0 * turn_embs[2] + 1.0 * turn_embs[3]) / 9.0

    assert torch.allclose(pooled_mat[0], expected_w0, atol=1e-6)
    assert torch.allclose(pooled_mat[1], expected_w1, atol=1e-6)
