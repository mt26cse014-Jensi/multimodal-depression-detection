"""
Phase 6: Cross-Modal Attention Model for DAIC-WOZ.

Architecture:
- Frozen text (RoBERTa-base, 768-d) and audio (WavLM-base-plus, 768-d) inputs per turn
- Linear projection to d_model (128-d)
- 1 or 2 Bidirectional Cross-Modal Attention Blocks (T -> A and A -> T MHA with 4 heads)
- Turn-level mean pooling over sequence dimension -> [h_text; h_audio] (256-d)
- Duration-weighted aggregation of turns into window embeddings (256-d)
- Shared MLP trunk (256 -> 128 -> BatchNorm1d -> ReLU -> Dropout(0.3))
- Dual output heads:
  * Binary classification head: Linear(128, 1) -> logit
  * PHQ-8 regression head: Linear(128, 1) -> normalized score
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalAttentionBlock(nn.Module):
    """Bidirectional cross-modal attention block.

    Text attends to audio frames (T -> A) and audio frames attend to text tokens (A -> T).
    Each branch uses multi-head attention followed by residual connections, LayerNorm,
    and a 2-layer feed-forward network with GELU activation.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Bidirectional cross-attention
        # T -> A: Query=Text, Key=Audio, Value=Audio
        self.mha_t2a = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        # A -> T: Query=Audio, Key=Text, Value=Text
        self.mha_a2t = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.drop_attn_t = nn.Dropout(dropout)
        self.drop_attn_a = nn.Dropout(dropout)

        self.ln_t1 = nn.LayerNorm(d_model)
        self.ln_a1 = nn.LayerNorm(d_model)

        # Feed-forward networks
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.ln_t2 = nn.LayerNorm(d_model)
        self.ln_a2 = nn.LayerNorm(d_model)

    def forward(
        self,
        t: torch.Tensor,
        a: torch.Tensor,
        t_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            t: Text embeddings, shape (batch_size, L_t, d_model).
            a: Audio frame embeddings, shape (batch_size, L_a, d_model).
            t_mask: Text key padding mask, shape (batch_size, L_t), True where padded.
            a_mask: Audio key padding mask, shape (batch_size, L_a), True where padded.
            return_attn_weights: If True, also returns attention weight matrices.

        Returns:
            Tuple of updated (t, a) representations, or (t, a, t_weights, a_weights).
        """
        # Text attends to Audio: query=t, key=a, value=a, key_padding_mask=a_mask
        t_attn, t_weights = self.mha_t2a(t, a, a, key_padding_mask=a_mask, need_weights=return_attn_weights)
        t = self.ln_t1(t + self.drop_attn_t(t_attn))

        # Audio attends to Text: query=a, key=t, value=t, key_padding_mask=t_mask
        a_attn, a_weights = self.mha_a2t(a, t, t, key_padding_mask=t_mask, need_weights=return_attn_weights)
        a = self.ln_a1(a + self.drop_attn_a(a_attn))

        # FFN sublayers
        t = self.ln_t2(t + self.ffn_t(t))
        a = self.ln_a2(a + self.ffn_a(a))

        if return_attn_weights:
            return t, a, t_weights, a_weights
        return t, a


class SmallCrossModalModel(nn.Module):
    """Compact cross-modal attention architecture for DAIC-WOZ.

    Follows the Phase 5 head capacity while adding cross-modal interaction:
    1. Projects 768-d frozen text and audio embeddings to d_model (128).
    2. Passes sequences through 1 or 2 CrossModalAttentionBlocks.
    3. Performs masked mean-pooling across tokens and frames to form turn representations (256-d).
    4. Aggregates turns into window embeddings via duration-weighted pooling.
    5. Feeds window embeddings into the shared MLP trunk and binary/regression heads.
    """

    def __init__(
        self,
        n_layers: int = 1,
        d_in_text: int = 768,
        d_in_audio: int = 768,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Input projections
        self.proj_text = nn.Linear(d_in_text, d_model)
        self.proj_audio = nn.Linear(d_in_audio, d_model)

        # Cross-modal attention blocks
        self.blocks = nn.ModuleList([
            CrossModalAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(n_layers)
        ])

        # Shared MLP trunk: identical to Phase 5 late fusion MLP
        # Input dim = 2 * d_model (concatenation of cross-attended text and audio)
        self.trunk = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Dual output heads
        self.classifier = nn.Linear(hidden_dim, 1)
        self.regressor = nn.Linear(hidden_dim, 1)

    def get_attention_maps(
        self,
        t: torch.Tensor,
        a: torch.Tensor,
        t_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract attention weight matrices from all CMA blocks.

        Returns:
            List of (t2a_weights, a2t_weights) per block.
            t2a_weights: (B, L_t, L_a)
            a2t_weights: (B, L_a, L_t)
        """
        t = self.proj_text(t)
        a = self.proj_audio(a)
        maps = []
        for block in self.blocks:
            t, a, t2a_w, a2t_w = block(t, a, t_mask=t_mask, a_mask=a_mask, return_attn_weights=True)
            maps.append((t2a_w, a2t_w))
        return maps

    def count_trainable_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_turns(
        self,
        t: torch.Tensor,
        a: torch.Tensor,
        t_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute fused turn embeddings for a batch of turns.

        Args:
            t: Text embeddings, shape (N_turns, L_t, d_in_text).
            a: Audio frame embeddings, shape (N_turns, L_a, d_in_audio).
            t_mask: Bool mask, shape (N_turns, L_t), True where padded.
            a_mask: Bool mask, shape (N_turns, L_a), True where padded.

        Returns:
            turn_embs: Fused turn embeddings, shape (N_turns, 2 * d_model).
        """
        # 1. Project to d_model
        t = self.proj_text(t)
        a = self.proj_audio(a)

        # 2. Cross-modal attention layers
        for block in self.blocks:
            t, a = block(t, a, t_mask=t_mask, a_mask=a_mask)

        # 3. Masked mean pooling across sequence dimension
        if t_mask is not None:
            t_valid = (~t_mask).unsqueeze(-1).float()
            t_pooled = (t * t_valid).sum(dim=1) / t_valid.sum(dim=1).clamp(min=1.0)
        else:
            t_pooled = t.mean(dim=1)

        if a_mask is not None:
            a_valid = (~a_mask).unsqueeze(-1).float()
            a_pooled = (a * a_valid).sum(dim=1) / a_valid.sum(dim=1).clamp(min=1.0)
        else:
            a_pooled = a.mean(dim=1)

        # Concatenate attended text and audio: (N_turns, 2 * d_model)
        turn_embs = torch.cat([t_pooled, a_pooled], dim=-1)
        return turn_embs

    def forward_windows(self, window_embs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and PHQ-8 regression predictions from window embeddings.

        Args:
            window_embs: Shape (N_windows, 2 * d_model).

        Returns:
            logits: Shape (N_windows, 1).
            phq_preds: Shape (N_windows, 1).
        """
        features = self.trunk(window_embs)
        logits = self.classifier(features)
        phq_preds = self.regressor(features)
        return logits, phq_preds

    def forward(
        self,
        t: torch.Tensor,
        a: torch.Tensor,
        window_turn_matrix: torch.Tensor,
        t_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass from turns to window predictions.

        Args:
            t: Text embeddings, shape (N_turns, L_t, d_in_text).
            a: Audio frame embeddings, shape (N_turns, L_a, d_in_audio).
            window_turn_matrix: Duration-weighted aggregation matrix, shape (N_windows, N_turns).
            t_mask: Text mask, shape (N_turns, L_t).
            a_mask: Audio mask, shape (N_turns, L_a).

        Returns:
            logits: (N_windows, 1)
            phq_preds: (N_windows, 1)
        """
        turn_embs = self.forward_turns(t, a, t_mask=t_mask, a_mask=a_mask)
        # Duration-weighted pooling over turns: (N_windows, N_turns) x (N_turns, 2*d_model) -> (N_windows, 2*d_model)
        window_embs = torch.matmul(window_turn_matrix, turn_embs)
        return self.forward_windows(window_embs)
