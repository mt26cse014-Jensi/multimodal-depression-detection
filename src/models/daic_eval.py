"""Participant-level aggregation, metrics, and bootstrap CIs for DAIC-WOZ (Phase 5).

Aggregation rule:
  1. Predict binary probability and PHQ-8 score for every window.
  2. Within each scale (30 s, 75 s, 120 s), aggregate per participant via majority vote
     over window binary predictions (ties broken by mean probability >= 0.5) and
     mean predicted probability / PHQ-8 score.
  3. Across the three scales, take the majority vote across scale votes for ONE
     final binary prediction per participant, mean scale probability for AUC-ROC,
     and mean scale PHQ-8 score for participant RMSE.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

METRIC_NAMES = ["macro_f1", "dep_f1", "accuracy", "auc_roc", "phq8_rmse"]


def aggregate_participant_predictions(
    window_preds_df: pd.DataFrame,
    scales: Sequence[int] = (30, 75, 120),
    threshold: float = 0.5,
    phq8_max_score: float = 24.0,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Aggregate window-level predictions into within-scale and cross-scale participant predictions.

    Required columns in ``window_preds_df``:
        ``session_id``, ``scale``, ``y_prob``, ``phq8_pred``
    Optional columns preserved when available:
        ``y_true``, ``phq8_true``

    Returns:
        participant_df: One row per participant (cross-scale majority vote + scale columns).
        scale_dfs: Mapping ``scale -> DataFrame`` with one row per participant present at that scale.
    """
    required = {"session_id", "scale", "y_prob", "phq8_pred"}
    missing = required - set(window_preds_df.columns)
    if missing:
        raise ValueError(f"window_preds_df missing required columns: {sorted(missing)}")

    df = window_preds_df.copy()
    df["session_id"] = df["session_id"].astype(int)
    df["scale"] = df["scale"].astype(int)
    df["win_pred"] = (df["y_prob"].to_numpy(dtype=float) >= threshold).astype(int)
    df["phq8_pred_clipped"] = np.clip(df["phq8_pred"].to_numpy(dtype=float), 0.0, phq8_max_score)

    has_y_true = "y_true" in df.columns
    has_phq8_true = "phq8_true" in df.columns

    scale_dfs: dict[int, pd.DataFrame] = {}
    for scale in scales:
        sub = df[df["scale"] == int(scale)]
        if sub.empty:
            continue

        rows = []
        for sid, grp in sub.groupby("session_id", sort=True):
            vote_frac = float(grp["win_pred"].mean())
            prob_mean = float(grp["y_prob"].mean())
            if vote_frac > 0.5:
                y_pred = 1
            elif vote_frac < 0.5:
                y_pred = 0
            else:
                y_pred = int(prob_mean >= threshold)

            phq8_mean = float(np.clip(grp["phq8_pred_clipped"].mean(), 0.0, phq8_max_score))
            y_true = float(grp["y_true"].iloc[0]) if has_y_true else math.nan
            phq8_true = float(grp["phq8_true"].iloc[0]) if has_phq8_true else math.nan

            rows.append(
                {
                    "session_id": int(sid),
                    "scale": int(scale),
                    "n_windows": int(len(grp)),
                    "y_true": y_true,
                    "phq8_true": phq8_true,
                    "y_pred": int(y_pred),
                    "y_prob": prob_mean,
                    "vote_frac": vote_frac,
                    "phq8_pred": phq8_mean,
                }
            )
        scale_dfs[int(scale)] = pd.DataFrame(rows)

    # Combine across scales per participant
    all_sids = sorted(df["session_id"].unique().tolist())
    participant_rows = []
    for sid in all_sids:
        scale_votes: list[int] = []
        scale_probs: list[float] = []
        scale_phq8: list[float] = []
        per_scale_cols: dict[str, float | int] = {}

        y_true = math.nan
        phq8_true = math.nan
        if has_y_true:
            y_true = float(df.loc[df["session_id"] == sid, "y_true"].iloc[0])
        if has_phq8_true:
            phq8_true = float(df.loc[df["session_id"] == sid, "phq8_true"].iloc[0])

        for scale in scales:
            s_int = int(scale)
            s_df = scale_dfs.get(s_int)
            if s_df is None:
                continue
            match = s_df[s_df["session_id"] == sid]
            if match.empty:
                continue
            r = match.iloc[0]
            scale_votes.append(int(r["y_pred"]))
            scale_probs.append(float(r["y_prob"]))
            scale_phq8.append(float(r["phq8_pred"]))
            per_scale_cols[f"pred_{s_int}s"] = int(r["y_pred"])
            per_scale_cols[f"prob_{s_int}s"] = float(r["y_prob"])
            per_scale_cols[f"phq8_{s_int}s"] = float(r["phq8_pred"])

        n_scales = len(scale_votes)
        mean_prob = float(np.mean(scale_probs)) if scale_probs else 0.0
        mean_phq8 = float(np.clip(np.mean(scale_phq8), 0.0, phq8_max_score)) if scale_phq8 else 0.0

        votes_sum = sum(scale_votes)
        if votes_sum > n_scales / 2.0:
            final_pred = 1
        elif votes_sum < n_scales / 2.0:
            final_pred = 0
        else:
            final_pred = int(mean_prob >= threshold)

        participant_rows.append(
            {
                "session_id": int(sid),
                "n_scales": int(n_scales),
                "y_true": y_true,
                "phq8_true": phq8_true,
                "y_pred": int(final_pred),
                "y_prob": mean_prob,
                "phq8_pred": mean_phq8,
                "scale_votes_sum": int(votes_sum),
                **per_scale_cols,
            }
        )

    return pd.DataFrame(participant_rows), scale_dfs


def compute_participant_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    phq8_true: Optional[np.ndarray] = None,
    phq8_pred: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute participant-level classification metrics and PHQ-8 RMSE."""
    y_t = np.asarray(y_true, dtype=float)
    valid_cls = ~np.isnan(y_t)
    if not valid_cls.any():
        return {m: float("nan") for m in METRIC_NAMES}

    yt = y_t[valid_cls].astype(int)
    yp = np.asarray(y_pred, dtype=int)[valid_cls]
    ypr = np.asarray(y_prob, dtype=float)[valid_cls]

    metrics = {
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "dep_f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
        "auc_roc": float(roc_auc_score(yt, ypr)) if len(np.unique(yt)) > 1 else float("nan"),
        "phq8_rmse": float("nan"),
    }

    if phq8_true is not None and phq8_pred is not None:
        rt = np.asarray(phq8_true, dtype=float)
        rp = np.asarray(phq8_pred, dtype=float)
        valid_reg = (~np.isnan(rt)) & (~np.isnan(rp))
        if valid_reg.any():
            metrics["phq8_rmse"] = float(np.sqrt(np.mean((rt[valid_reg] - rp[valid_reg]) ** 2)))

    return metrics


def _metrics_from_participant_df(df: pd.DataFrame) -> dict[str, float]:
    return compute_participant_metrics(
        y_true=df["y_true"].to_numpy(dtype=float),
        y_pred=df["y_pred"].to_numpy(dtype=int),
        y_prob=df["y_prob"].to_numpy(dtype=float),
        phq8_true=df["phq8_true"].to_numpy(dtype=float) if "phq8_true" in df.columns else None,
        phq8_pred=df["phq8_pred"].to_numpy(dtype=float) if "phq8_pred" in df.columns else None,
    )


def bootstrap_participant_ci(
    seed_participant_dfs: Sequence[pd.DataFrame],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute multi-seed mean, seed std, and stratified participant bootstrap 95% CIs.

    For each stratified participant bootstrap draw (preserving exact depressed vs
    control count), evaluates all seeds on the same drawn participant IDs and
    averages each metric across seeds.
    """
    if not seed_participant_dfs:
        raise ValueError("seed_participant_dfs must not be empty")

    # Align seeds on identical ordered labeled participants
    ref = seed_participant_dfs[0].dropna(subset=["y_true"]).sort_values("session_id").reset_index(drop=True)
    if ref.empty:
        return {m: {"mean": float("nan"), "seed_std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")} for m in METRIC_NAMES}

    sids = ref["session_id"].to_numpy(dtype=int)
    aligned: list[pd.DataFrame] = []
    for df in seed_participant_dfs:
        sub = df.set_index("session_id").loc[sids].reset_index()
        aligned.append(sub)

    # Full-dataset per-seed metrics
    seed_metrics = [_metrics_from_participant_df(df) for df in aligned]
    summary: dict[str, dict[str, float]] = {}

    y_ref = ref["y_true"].to_numpy(dtype=int)
    pos_idx = np.where(y_ref == 1)[0]
    neg_idx = np.where(y_ref == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0 or n_bootstrap < 1:
        for m in METRIC_NAMES:
            vals = np.array([sm[m] for sm in seed_metrics], dtype=float)
            valid = vals[~np.isnan(vals)]
            mean_v = float(np.mean(valid)) if len(valid) else float("nan")
            std_v = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
            summary[m] = {"mean": mean_v, "seed_std": std_v, "ci_low": float("nan"), "ci_high": float("nan")}
        return summary

    # Pre-extract numpy arrays for fast bootstrap evaluation
    arrs = [
        (
            df["y_true"].to_numpy(dtype=float),
            df["y_pred"].to_numpy(dtype=int),
            df["y_prob"].to_numpy(dtype=float),
            df["phq8_true"].to_numpy(dtype=float) if "phq8_true" in df.columns else None,
            df["phq8_pred"].to_numpy(dtype=float) if "phq8_pred" in df.columns else None,
        )
        for df in aligned
    ]

    rng = np.random.default_rng(seed)
    boot_vals: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}

    for _ in range(n_bootstrap):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        per_seed = {m: [] for m in METRIC_NAMES}
        for yt, yp, ypr, rt, rp in arrs:
            res = compute_participant_metrics(
                yt[b_idx],
                yp[b_idx],
                ypr[b_idx],
                rt[b_idx] if rt is not None else None,
                rp[b_idx] if rp is not None else None,
            )
            for m in METRIC_NAMES:
                if not np.isnan(res[m]):
                    per_seed[m].append(res[m])

        for m in METRIC_NAMES:
            if per_seed[m]:
                boot_vals[m].append(float(np.mean(per_seed[m])))

    for m in METRIC_NAMES:
        vals = np.array([sm[m] for sm in seed_metrics], dtype=float)
        valid = vals[~np.isnan(vals)]
        mean_v = float(np.mean(valid)) if len(valid) else float("nan")
        std_v = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        b_arr = np.asarray(boot_vals[m], dtype=float)
        ci_lo = float(np.percentile(b_arr, 2.5)) if len(b_arr) else float("nan")
        ci_hi = float(np.percentile(b_arr, 97.5)) if len(b_arr) else float("nan")
        summary[m] = {"mean": mean_v, "seed_std": std_v, "ci_low": ci_lo, "ci_high": ci_hi}

    return summary


def paired_bootstrap_gap_ci(
    base_seed_dfs: Sequence[pd.DataFrame],
    alt_seed_dfs: Sequence[pd.DataFrame],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Paired stratified bootstrap 95% CIs for ``gap = alt - base`` over common participants."""
    ref = base_seed_dfs[0].dropna(subset=["y_true"]).sort_values("session_id").reset_index(drop=True)
    if ref.empty:
        return {m: {"mean": float("nan"), "seed_std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")} for m in METRIC_NAMES}

    sids = ref["session_id"].to_numpy(dtype=int)
    base_aligned = [df.set_index("session_id").loc[sids].reset_index() for df in base_seed_dfs]
    alt_aligned = [df.set_index("session_id").loc[sids].reset_index() for df in alt_seed_dfs]

    n_pairs = min(len(base_aligned), len(alt_aligned))
    gaps_per_seed = []
    for k in range(n_pairs):
        mb = _metrics_from_participant_df(base_aligned[k])
        ma = _metrics_from_participant_df(alt_aligned[k])
        gaps_per_seed.append({m: ma[m] - mb[m] for m in METRIC_NAMES})

    y_ref = ref["y_true"].to_numpy(dtype=int)
    pos_idx = np.where(y_ref == 1)[0]
    neg_idx = np.where(y_ref == 0)[0]

    out: dict[str, dict[str, float]] = {}
    if len(pos_idx) == 0 or len(neg_idx) == 0 or n_bootstrap < 1:
        for m in METRIC_NAMES:
            vals = np.array([g[m] for g in gaps_per_seed], dtype=float)
            valid = vals[~np.isnan(vals)]
            out[m] = {
                "mean": float(np.mean(valid)) if len(valid) else float("nan"),
                "seed_std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
        return out

    base_arrs = [
        (
            df["y_true"].to_numpy(dtype=float),
            df["y_pred"].to_numpy(dtype=int),
            df["y_prob"].to_numpy(dtype=float),
            df["phq8_true"].to_numpy(dtype=float) if "phq8_true" in df.columns else None,
            df["phq8_pred"].to_numpy(dtype=float) if "phq8_pred" in df.columns else None,
        )
        for df in base_aligned[:n_pairs]
    ]
    alt_arrs = [
        (
            df["y_true"].to_numpy(dtype=float),
            df["y_pred"].to_numpy(dtype=int),
            df["y_prob"].to_numpy(dtype=float),
            df["phq8_true"].to_numpy(dtype=float) if "phq8_true" in df.columns else None,
            df["phq8_pred"].to_numpy(dtype=float) if "phq8_pred" in df.columns else None,
        )
        for df in alt_aligned[:n_pairs]
    ]

    rng = np.random.default_rng(seed)
    boot_gaps: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for _ in range(n_bootstrap):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        seed_diffs = {m: [] for m in METRIC_NAMES}
        for (yt_b, yp_b, ypr_b, rt_b, rp_b), (yt_a, yp_a, ypr_a, rt_a, rp_a) in zip(base_arrs, alt_arrs):
            mb = compute_participant_metrics(
                yt_b[b_idx], yp_b[b_idx], ypr_b[b_idx],
                rt_b[b_idx] if rt_b is not None else None,
                rp_b[b_idx] if rp_b is not None else None,
            )
            ma = compute_participant_metrics(
                yt_a[b_idx], yp_a[b_idx], ypr_a[b_idx],
                rt_a[b_idx] if rt_a is not None else None,
                rp_a[b_idx] if rp_a is not None else None,
            )
            for m in METRIC_NAMES:
                diff = ma[m] - mb[m]
                if not np.isnan(diff):
                    seed_diffs[m].append(diff)

        for m in METRIC_NAMES:
            if seed_diffs[m]:
                boot_gaps[m].append(float(np.mean(seed_diffs[m])))

    for m in METRIC_NAMES:
        vals = np.array([g[m] for g in gaps_per_seed], dtype=float)
        valid = vals[~np.isnan(vals)]
        b_arr = np.asarray(boot_gaps[m], dtype=float)
        out[m] = {
            "mean": float(np.mean(valid)) if len(valid) else float("nan"),
            "seed_std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "ci_low": float(np.percentile(b_arr, 2.5)) if len(b_arr) else float("nan"),
            "ci_high": float(np.percentile(b_arr, 97.5)) if len(b_arr) else float("nan"),
        }
    return out


def compute_daic_majority_baseline(
    manifest_df: pd.DataFrame,
    eval_session_ids: Sequence[int],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute majority-class (always 0) + train-mean PHQ-8 baseline on ``eval_session_ids``."""
    train_scores = manifest_df.loc[manifest_df["split"] == "train", "phq8_score"].dropna().astype(float)
    mean_phq8 = float(train_scores.mean()) if not train_scores.empty else 0.0

    sub = manifest_df[manifest_df["session_id"].astype(int).isin({int(s) for s in eval_session_ids})].copy()
    sub = sub.sort_values("session_id").reset_index(drop=True)
    dummy_df = pd.DataFrame(
        {
            "session_id": sub["session_id"].astype(int),
            "y_true": sub["phq8_binary"].astype(float),
            "phq8_true": sub["phq8_score"].astype(float),
            "y_pred": np.zeros(len(sub), dtype=int),
            "y_prob": np.zeros(len(sub), dtype=float),
            "phq8_pred": np.full(len(sub), mean_phq8, dtype=float),
        }
    )
    return bootstrap_participant_ci([dummy_df], n_bootstrap=n_bootstrap, seed=seed)

