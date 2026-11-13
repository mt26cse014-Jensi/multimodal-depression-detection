"""
Turn-Level Occlusion Importance Analysis for Multimodal Depression Detection.

Quantifies the marginal predictive impact of individual participant turns
by measuring the change in predicted probability when each turn is omitted:
    Delta_P = P_baseline - P_occluded

License & Privacy:
Outputs contain ONLY numerical session IDs, turn IDs, timestamps, and impact scores.
NO transcript text is ever stored, logged, or printed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def occlude_session_turns(
    session_data: Dict[str, Any],
    model: nn.Module,
    phq8_max_score: float = 24.0,
) -> pd.DataFrame:
    """Compute turn-level occlusion impact for all participant turns in a single session.

    Args:
        session_data: Output dict from build_session_data containing padded tensors,
            turn metadata (turn_ids, start_times, stop_times, turn_durations), and w_mat.
        model: Trained SmallCrossModalModel.
        phq8_max_score: Maximum PHQ-8 score for scaling.

    Returns:
        DataFrame with columns: session_id, turn_id, start_time, stop_time,
        duration, relative_position, p_baseline, p_occluded, delta_p, abs_delta_p,
        delta_phq, rank_in_session.
    """
    model.eval()
    device = next(model.parameters()).device

    sid = int(session_data["session_id"])
    t_pad = session_data["t_pad"].to(device)
    a_pad = session_data["a_pad"].to(device)
    t_mask = session_data["t_mask"].to(device)
    a_mask = session_data["a_mask"].to(device)
    w_mat = session_data["w_mat"].to(device)

    turn_ids = np.asarray(session_data["turn_ids"], dtype=int)
    start_times = np.asarray(session_data["start_times"], dtype=float)
    stop_times = np.asarray(session_data["stop_times"], dtype=float)
    turn_durs = np.asarray(session_data["turn_durations"], dtype=float)
    n_turns = len(turn_ids)

    if n_turns == 0:
        return pd.DataFrame()

    t_end = float(stop_times.max()) if len(stop_times) > 0 and stop_times.max() > 0 else 1.0

    with torch.no_grad():
        # Baseline prediction: compute turn embeddings once
        turn_embs = model.forward_turns(t_pad, a_pad, t_mask=t_mask, a_mask=a_mask)

        # Baseline window forward
        base_win_embs = torch.matmul(w_mat, turn_embs)
        base_logits, base_phq_norm = model.forward_windows(base_win_embs)
        base_probs = torch.sigmoid(base_logits.squeeze(-1)).cpu().numpy()
        base_p_mean = float(base_probs.mean())
        base_phq_mean = float(base_phq_norm.squeeze(-1).mean().item() * phq8_max_score)

        records: List[Dict[str, Any]] = []

        # Occlude one turn at a time
        for k in range(n_turns):
            w_k = w_mat.clone()
            w_k[:, k] = 0.0

            # Row re-normalization
            row_sums = w_k.sum(dim=1, keepdim=True)
            w_k = torch.where(row_sums > 0, w_k / row_sums.clamp(min=1e-8), torch.zeros_like(w_k))

            win_embs_k = torch.matmul(w_k, turn_embs)
            logits_k, phq_k = model.forward_windows(win_embs_k)
            probs_k = torch.sigmoid(logits_k.squeeze(-1)).cpu().numpy()

            p_occ_mean = float(probs_k.mean())
            phq_occ_mean = float(phq_k.squeeze(-1).mean().item() * phq8_max_score)

            delta_p = base_p_mean - p_occ_mean
            delta_phq = base_phq_mean - phq_occ_mean

            rel_pos = float((start_times[k] + stop_times[k]) / (2.0 * t_end))

            records.append({
                "session_id": sid,
                "turn_id": int(turn_ids[k]),
                "start_time": float(start_times[k]),
                "stop_time": float(stop_times[k]),
                "duration": float(turn_durs[k]),
                "relative_position": float(np.clip(rel_pos, 0.0, 1.0)),
                "p_baseline": base_p_mean,
                "p_occluded": p_occ_mean,
                "delta_p": delta_p,
                "abs_delta_p": abs(delta_p),
                "delta_phq": delta_phq,
            })

    df = pd.DataFrame(records)
    # Sort and rank within session by absolute impact
    df = df.sort_values("abs_delta_p", ascending=False).reset_index(drop=True)
    df["rank_in_session"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def run_turn_occlusion_suite(
    sessions_data: Sequence[Dict[str, Any]],
    model: nn.Module,
    output_csv: Path | str = "results/turn_occlusion_importance.csv",
    top_k_csv: Path | str = "results/top_turns_per_participant.csv",
    top_k: int = 5,
    phq8_max_score: float = 24.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute turn occlusion importance analysis across a cohort of sessions.

    Saves:
        1. Complete turn importance CSV
        2. Filtered top-K turns per participant CSV
    """
    output_csv = Path(output_csv)
    top_k_csv = Path(top_k_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    top_k_csv.parent.mkdir(parents=True, exist_ok=True)

    all_dfs: List[pd.DataFrame] = []
    top_dfs: List[pd.DataFrame] = []

    for s_idx, s_data in enumerate(sessions_data, 1):
        sid = s_data["session_id"]
        logger.info("[%d/%d] Occluding turns for session %d...", s_idx, len(sessions_data), sid)
        s_df = occlude_session_turns(s_data, model, phq8_max_score=phq8_max_score)
        if not s_df.empty:
            all_dfs.append(s_df)
            top_dfs.append(s_df.head(top_k))

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    top_combined_df = pd.concat(top_dfs, ignore_index=True) if top_dfs else pd.DataFrame()

    if not combined_df.empty:
        combined_df.to_csv(output_csv, index=False)
        top_combined_df.to_csv(top_k_csv, index=False)
        logger.info(
            "Saved %d turn occlusion records to %s (top-%d saved to %s)",
            len(combined_df),
            output_csv,
            top_k,
            top_k_csv,
        )

    return combined_df, top_combined_df

