"""
Cross-Modal Attention Maps Extraction and Visualization.

Extracts bidirectional attention weight matrices (T -> A and A -> T)
from the SmallCrossModalModel for 3 representative example windows:
  1. High-confidence Depressed
  2. High-confidence Control
  3. Ambiguous / Borderline Case

License & Privacy:
Tokens are indexed numerically (Token 0, Token 1, ...).
No transcript text is displayed or stored in the figure or metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_cross_modal_attention_maps(
    model: nn.Module,
    session_data: Dict[str, Any],
    turn_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Extract bidirectional attention maps for a specific turn in a session.

    Args:
        model: Trained SmallCrossModalModel with get_attention_maps method.
        session_data: Output dict from build_session_data.
        turn_index: Integer index of turn within the session.

    Returns:
        t2a_matrix: (L_t, L_a) float32 attention weights (Text -> Audio).
        a2t_matrix: (L_a, L_t) float32 attention weights (Audio -> Text).
        n_tokens: Valid token count for this turn.
        n_frames: Valid audio frame count for this turn.
    """
    model.eval()
    device = next(model.parameters()).device

    t_pad = session_data["t_pad"].to(device)
    a_pad = session_data["a_pad"].to(device)
    t_mask = session_data["t_mask"].to(device)
    a_mask = session_data["a_mask"].to(device)

    with torch.no_grad():
        # Shape of return: list of (t2a_w, a2t_w) per block
        # t2a_w shape: (N_turns, L_t, L_a)
        # a2t_w shape: (N_turns, L_a, L_t)
        maps = model.get_attention_maps(t_pad, a_pad, t_mask=t_mask, a_mask=a_mask)
        # Use first layer CMA block
        t2a_w, a2t_w = maps[0]

        # Valid lengths
        valid_t = int((~t_mask[turn_index]).sum().item())
        valid_a = int((~a_mask[turn_index]).sum().item())

        t2a_turn = t2a_w[turn_index, :valid_t, :valid_a].cpu().numpy().astype(np.float32)
        a2t_turn = a2t_w[turn_index, :valid_a, :valid_t].cpu().numpy().astype(np.float32)

    return t2a_turn, a2t_turn, valid_t, valid_a


def plot_example_attention_heatmaps(
    examples: List[Dict[str, Any]],
    output_path: Path | str = "results/figures/attention_maps_3_examples.png",
    fps: float = 25.0,
) -> None:
    """Plot bidirectional attention heatmaps for 3 representative example cases.

    Args:
        examples: List of 3 dicts with:
            case_name: str (e.g. 'High-Confidence Depressed')
            session_id: int
            turn_id: int
            label: float
            pred_prob: float
            t2a_matrix: (L_t, L_a)
            a2t_matrix: (L_a, L_t)
            duration_sec: float
        output_path: Path to save PNG.
        fps: Audio frames per second (default 25.0).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_cases = len(examples)
    fig, axes = plt.subplots(n_cases, 2, figsize=(14, 4 * n_cases))
    if n_cases == 1:
        axes = np.array([axes])

    for i, ex in enumerate(examples):
        t2a = ex["t2a_matrix"]
        a2t = ex["a2t_matrix"]
        dur = ex.get("duration_sec", t2a.shape[1] / fps)
        time_extent = [0.0, dur, t2a.shape[0] - 0.5, -0.5]

        # Left: Text -> Audio
        ax_t2a = axes[i, 0]
        im1 = ax_t2a.imshow(t2a, aspect="auto", cmap="viridis", extent=time_extent, origin="upper")
        ax_t2a.set_title(
            f"{ex['case_name']} (Session {ex['session_id']}, Turn {ex['turn_id']})\n"
            f"True Label: {int(ex['label']) if pd.notna(ex['label']) else 'N/A'}, P(Dep): {ex['pred_prob']:.3f} | Text -> Audio",
            fontsize=10,
            fontweight="bold",
        )
        ax_t2a.set_xlabel("Audio Time (seconds)", fontsize=9)
        ax_t2a.set_ylabel("Text Token Index", fontsize=9)
        fig.colorbar(im1, ax=ax_t2a, fraction=0.046, pad=0.04)

        # Right: Audio -> Text
        ax_a2t = axes[i, 1]
        im2 = ax_a2t.imshow(a2t, aspect="auto", cmap="magma", extent=[-0.5, t2a.shape[0] - 0.5, dur, 0.0], origin="upper")
        ax_a2t.set_title(
            f"{ex['case_name']} | Audio -> Text Attention",
            fontsize=10,
            fontweight="bold",
        )
        ax_a2t.set_xlabel("Text Token Index", fontsize=9)
        ax_a2t.set_ylabel("Audio Time (seconds)", fontsize=9)
        fig.colorbar(im2, ax=ax_a2t, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved 3 example attention maps to %s", output_path)

