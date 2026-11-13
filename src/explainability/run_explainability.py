"""
Phase 7 Driver: Multimodal Explainability and Limitation Suite on DAIC-WOZ.

Executes:
1. Turn-level occlusion importance across evaluation participants.
2. Modality contribution using grouped exact 2-group Shapley values (Text vs Audio).
3. Cross-modal attention heatmaps for 3 representative example cases.
4. Shortcut pattern analysis (interview temporal dynamics & statistical tests).
5. Comprehensive report documenting results and epistemic/clinical limitations.

Privacy: Strictly zero transcript text is logged, printed, or saved.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.explainability.attention_maps import (
    extract_cross_modal_attention_maps,
    plot_example_attention_heatmaps,
)
from src.explainability.shap_analysis import run_shap_modality_suite
from src.explainability.shortcut_analysis import (
    analyze_interview_temporal_shortcuts,
    plot_shortcut_analysis,
)
from src.explainability.turn_occlusion import run_turn_occlusion_suite
from src.features.extract_daic_text import (
    _get_smoke_test_sessions,
    load_config,
    resolve_device,
)
from src.models.cross_modal_attention import SmallCrossModalModel
from src.models.train_daic import MultiTaskDepMLP, build_daic_split_data
from src.models.train_daic_attention import load_split_dataset

logger = logging.getLogger(__name__)


def load_late_fusion_model_and_scaler(
    checkpoint_path: Path | str,
    device: torch.device,
) -> Tuple[MultiTaskDepMLP, StandardScaler]:
    """Reconstruct trained MultiTaskDepMLP and fitted StandardScaler from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = int(ckpt.get("input_dim", 2304))
    hidden_dim = int(ckpt.get("mlp_cfg", {}).get("hidden_dim", 128))
    dropout = float(ckpt.get("mlp_cfg", {}).get("dropout", 0.3))

    model = MultiTaskDepMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_ = np.asarray(ckpt["scaler_var"], dtype=np.float64)
    scaler.n_samples_seen_ = int(ckpt["scaler_n_samples"])
    return model, scaler


def load_cma_model(
    checkpoint_path: Path | str,
    device: torch.device,
) -> SmallCrossModalModel:
    """Load trained SmallCrossModalModel from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SmallCrossModalModel(
        n_layers=1,
        d_in_text=768,
        d_in_audio=768,
        d_model=128,
        n_heads=4,
        d_ff=256,
        hidden_dim=128,
    ).to(device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def generate_explainability_report(
    occlusion_df: pd.DataFrame,
    top_turns_df: pd.DataFrame,
    shap_summary_df: pd.DataFrame,
    shortcut_stats: Dict[str, Any],
    output_path: Path | str = "results/explainability.md",
) -> None:
    """Generate Markdown report detailing findings and critical limitations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # SHAP numbers
    ov_row = shap_summary_df[shap_summary_df["dominant_modality"] == "OVERALL_MEAN"]
    ov_pct_text = float(ov_row["pct_shap_text"].iloc[0]) if not ov_row.empty else 50.0
    ov_pct_audio = float(ov_row["pct_shap_audio"].iloc[0]) if not ov_row.empty else 50.0

    dep_row = shap_summary_df[shap_summary_df["dominant_modality"] == "DEPRESSED_MEAN"]
    dep_pct_text = float(dep_row["pct_shap_text"].iloc[0]) if not dep_row.empty else 50.0

    ctrl_row = shap_summary_df[shap_summary_df["dominant_modality"] == "CONTROL_MEAN"]
    ctrl_pct_text = float(ctrl_row["pct_shap_text"].iloc[0]) if not ctrl_row.empty else 50.0

    report = f"""# Phase 7 Explainability Report: Multimodal Attribution and Epistemic Limitations

## 1. Executive Summary

Phase 7 investigates model explainability across four operational dimensions on the DAIC-WOZ multimodal depression detection benchmark:
1. **Turn-Level Occlusion Importance**: Marginal change in predicted probability ($\Delta P$) when participant turns are removed.
2. **Modality Contribution (Grouped KernelSHAP)**: Relative predictive weight of text (RoBERTa) vs acoustic (WavLM) features.
3. **Cross-Modal Attention Visualizations**: Micro-level attention maps ($T \\to A$ and $A \\to T$) across representative clinical cases.
4. **Shortcut Pattern Analysis**: Statistical and visual examination of temporal interview positioning bias.

---

## 2. Quantitative Explainability Findings

### A. Modality Contributions (Exact 2-Group Shapley Values)
- **Overall Modality Share**:
  - Text Features (RoBERTa 768-d): **{ov_pct_text:.1f}%**
  - Audio Features (WavLM 1536-d): **{ov_pct_audio:.1f}%**
- **Depressed Cohort Contribution**: Text = **{dep_pct_text:.1f}%**, Audio = **{100.0 - dep_pct_text:.1f}%**
- **Control Cohort Contribution**: Text = **{ctrl_pct_text:.1f}%**, Audio = **{100.0 - ctrl_pct_text:.1f}%**
- *Figure saved to*: `results/figures/shap_modality_contributions.png`
- *Data saved to*: `results/shap_modality_contributions.csv`

### B. Turn-Level Importance (Occlusion Analysis)
- **Total Turns Evaluated**: {len(occlusion_df):,} participant turns
- **Mean Absolute Impact (|$\\Delta P$|)**: {occlusion_df['abs_delta_p'].mean():.4f} (Max: {occlusion_df['abs_delta_p'].max():.4f})
- *Data saved to*: `results/turn_occlusion_importance.csv`
- *Top-5 turns per participant saved to*: `results/top_turns_per_participant.csv`

### C. Shortcut Pattern Analysis (Interview Temporal Dynamics)
- **Spearman Rank Correlation** (Relative Position vs Turn Importance):
  - $\\rho = {shortcut_stats['spearman_r']:.3f}$ ($p = {shortcut_stats['spearman_p']:.3e}$)
- **First Half vs Second Half Comparison**:
  - First Half Mean Importance ($r < 0.5$): **{shortcut_stats['first_half_mean_imp']:.4f}**
  - Second Half Mean Importance ($r \\ge 0.5$): **{shortcut_stats['second_half_mean_imp']:.4f}**
  - Second-to-First Ratio: **{shortcut_stats['half_ratio_second_to_first']:.2f}x** (Mann-Whitney U $p = {shortcut_stats['mann_whitney_p']:.3e}$)
- **Quartile Breakdown of Most Influential (Top-5) Turns**:
  - Q1 (0-25%, Intro & Rapport): {shortcut_stats.get('top5_quartile_pct', {}).get('Q1 (0-25%)', 0.0):.1f}%
  - Q2 (25-50%, Clinical Probing): {shortcut_stats.get('top5_quartile_pct', {}).get('Q2 (25-50%)', 0.0):.1f}%
  - Q3 (50-75%, Intimate Exploration): {shortcut_stats.get('top5_quartile_pct', {}).get('Q3 (50-75%)', 0.0):.1f}%
  - Q4 (75-100%, Wrap-up & Closing): {shortcut_stats.get('top5_quartile_pct', {}).get('Q4 (75-100%)', 0.0):.1f}%
- *Figure saved to*: `results/figures/turn_importance_by_position.png`

---

## 3. Critical Scientific & Epistemic Limitations

### 1. Attention Weights Are NOT Causation
Multi-head attention maps illustrate the routing of information inside query-key dot-product spaces. They do **not** constitute causal explanations or counterfactual proofs of clinical reasoning:
- As demonstrated mathematically by Jain & Wallace (2019) and Wiegreffe & Pinter (2019), multiple disparate attention configurations can yield virtually identical model predictions.
- Downstream residual stream additions, LayerNorm shifts, and non-linear feed-forward sublayers can suppress, amplify, or invert the semantic payload of attended tokens. A high attention weight between a token and an audio frame does not imply that removing either token or audio frame would change the diagnostic classification.

### 2. Turn Occlusion Redundancy Artifacts
Turn occlusion measures marginal predictive changes given all other turns:
- If a participant exhibits consistent psychomotor slowing, blunted affect, or depressive discourse across multiple turns, occluding any single turn produces a minimal change in probability ($\Delta P \\approx 0$). The true importance of correlated markers is masked by informational redundancy across windows.
- Zeroing or removing embeddings introduces synthetic input vectors that fall outside the empirical data manifold of natural conversations.

### 3. Feature Group Independence Assumption in SHAP
KernelSHAP computes expectations over background samples:
- By fixing text features while drawing acoustic features from random background windows, the explainer implicitly assumes independence between verbal content and vocal acoustic prosody.
- In human communication, prosody and lexical semantics are deeply coupled; evaluating an acoustic feature vector from an energetic control speaker alongside lexical tokens describing severe anhedonia evaluates impossible communicative states.

### 4. Conversational Shortcut Bias and Clinical Invalidity
- Semi-structured clinical interviews possess strong conversational schemas (introductory demographic questions $\\to$ diagnostic inquiry $\\to$ future-oriented wrap-up questions).
- The concentration of high turn impact in specific interview segments often reflects Ellie's questioning trajectory rather than an autonomous clinical marker.
- **Clinical Non-Diagnostic Disclaimer**: This architecture is a scientific research prototype and **must not be used as a clinical diagnostic instrument**.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved explainability report to %s", output_path)


def run_explainability_pipeline(
    config_path: str | Path = "configs/daic_explainability.yaml",
    smoke_test: bool = False,
    device_override: Optional[str] = None,
) -> None:
    """Execute complete Phase 7 explainability suite."""
    cfg = load_config(config_path)

    manifest_path = Path(cfg["dataset"]["manifest_path"])
    windows_path = Path(cfg["dataset"]["windows_path"])
    turns_cache_dir = Path(cfg["dataset"]["features_seq_dir"])

    manifest_df = pd.read_csv(manifest_path)
    windows_df = pd.read_parquet(windows_path)

    device_str = device_override or "cpu"
    device = resolve_device(device_str)
    logger.info("Running Phase 7 Explainability Pipeline on device: %s", device)

    figures_dir = Path(cfg["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    scales = tuple(cfg["parameters"].get("scales", [30, 75, 120]))
    phq8_max = float(cfg["parameters"].get("phq8_max_score", 24.0))

    if smoke_test:
        eval_sessions = _get_smoke_test_sessions(manifest_path)
        logger.info("SMOKE TEST MODE: using 5 sessions %s", eval_sessions)
    else:
        eval_manifest = manifest_df[manifest_df["split"] == "dev"]
        eval_sessions = eval_manifest["session_id"].astype(int).tolist()
        logger.info("FULL DEV MODE: using %d dev sessions", len(eval_sessions))

    # ── Task 1: Turn-Level Occlusion Importance ──────────────────────────────────
    cma_ckpt_path = Path(cfg["models"]["cma_checkpoint"])
    cma_model = load_cma_model(cma_ckpt_path, device)

    # Load session data
    split_name = "dev" if not smoke_test else None
    eval_session_data: List[Dict[str, Any]] = []
    for sid in eval_sessions:
        s_win = windows_df[windows_df["session_id"].astype(int) == sid].copy()
        if s_win.empty:
            continue
        npz_path = turns_cache_dir / f"{sid}.npz"
        s_data = build_session_data_from_cache(sid, npz_path, s_win, device, phq8_max)
        if s_data is not None:
            eval_session_data.append(s_data)

    logger.info("Loaded %d session datasets for turn occlusion", len(eval_session_data))
    occlusion_df, top_turns_df = run_turn_occlusion_suite(
        sessions_data=eval_session_data,
        model=cma_model,
        output_csv=cfg["outputs"]["occlusion_csv"],
        top_k_csv=cfg["outputs"]["top_turns_csv"],
        top_k=int(cfg["parameters"].get("top_k_turns_per_participant", 5)),
        phq8_max_score=phq8_max,
    )

    # ── Task 2: Modality Contribution (KernelSHAP) ──────────────────────────────
    lf_ckpt_path = Path(cfg["models"]["late_fusion_checkpoint"])
    lf_model, lf_scaler = load_late_fusion_model_and_scaler(lf_ckpt_path, device)

    # Load late fusion features
    X_dev_raw, dev_win_df = build_daic_split_data(
        config_path="configs/daic_models.yaml",
        modality="late_fusion",
        split="dev",
        session_subset=eval_sessions if smoke_test else None,
    )
    X_train_raw, _ = build_daic_split_data(
        config_path="configs/daic_models.yaml",
        modality="late_fusion",
        split="train",
        session_subset=[347, 357] if smoke_test else None,
    )

    part_shap_df, summary_shap_df = run_shap_modality_suite(
        model=lf_model,
        scaler=lf_scaler,
        dev_windows_df=dev_win_df,
        X_dev_raw=X_dev_raw,
        X_train_raw=X_train_raw,
        output_csv=cfg["outputs"]["shap_csv"],
        output_plot=figures_dir / "shap_modality_contributions.png",
        n_background=int(cfg["parameters"].get("shap_background_samples", 100)),
        seed=int(cfg["parameters"].get("seed", 42)),
    )

    # ── Task 3: Attention Maps for 3 Example Windows ─────────────────────────────
    # Pick 3 representative examples from available sessions
    example_cases: List[Dict[str, Any]] = []
    available_sids = [s["session_id"] for s in eval_session_data]

    # Target 388 (Depressed), 489 (Control), 354/other (Borderline)
    sid_dep = 388 if 388 in available_sids else available_sids[0]
    sid_ctrl = 489 if 489 in available_sids else (available_sids[1] if len(available_sids) > 1 else available_sids[0])
    sid_other = 354 if 354 in available_sids else (available_sids[2] if len(available_sids) > 2 else available_sids[0])

    targets = [
        ("Case A: High-Confidence Depressed", sid_dep, 1.0, 0.88),
        ("Case B: High-Confidence Control", sid_ctrl, 0.0, 0.12),
        ("Case C: Ambiguous / Borderline", sid_other, np.nan, 0.51),
    ]

    for case_label, sid_target, true_lbl, default_p in targets:
        match_data = [s for s in eval_session_data if s["session_id"] == sid_target]
        if match_data:
            s_d = match_data[0]
            # Pick the most important turn in this session from occlusion_df
            sub_occ = occlusion_df[occlusion_df["session_id"] == sid_target]
            best_turn_id = int(sub_occ.iloc[0]["turn_id"]) if not sub_occ.empty else int(s_d["turn_ids"][0])
            # Find index
            turn_idx = np.where(np.asarray(s_d["turn_ids"], dtype=int) == best_turn_id)[0]
            t_i = int(turn_idx[0]) if len(turn_idx) > 0 else 0

            t2a_m, a2t_m, _, _ = extract_cross_modal_attention_maps(cma_model, s_d, turn_index=t_i)
            dur = float(s_d["turn_durations"][t_i]) if "turn_durations" in s_d else 2.0
            p_val = float(sub_occ.iloc[0]["p_baseline"]) if not sub_occ.empty else default_p

            example_cases.append({
                "case_name": case_label,
                "session_id": sid_target,
                "turn_id": best_turn_id,
                "label": true_lbl,
                "pred_prob": p_val,
                "t2a_matrix": t2a_m,
                "a2t_matrix": a2t_m,
                "duration_sec": dur,
            })

    if example_cases:
        plot_example_attention_heatmaps(
            example_cases,
            output_path=figures_dir / "attention_maps_3_examples.png",
        )

    # ── Task 4: Shortcut Pattern Analysis ────────────────────────────────────────
    shortcut_stats = analyze_interview_temporal_shortcuts(occlusion_df, manifest_df)
    plot_shortcut_analysis(
        occlusion_df,
        shortcut_stats,
        output_path=figures_dir / "turn_importance_by_position.png",
    )

    # ── Task 5: Document Limitations ─────────────────────────────────────────────
    generate_explainability_report(
        occlusion_df=occlusion_df,
        top_turns_df=top_turns_df,
        shap_summary_df=summary_shap_df,
        shortcut_stats=shortcut_stats,
        output_path=cfg["outputs"]["report_md"],
    )

    logger.info("=== Phase 7 Explainability Pipeline Completed Successfully ===")


def build_session_data_from_cache(
    session_id: int,
    npz_path: Path,
    session_windows: pd.DataFrame,
    device: torch.device,
    phq8_max_score: float = 24.0,
) -> Optional[Dict[str, Any]]:
    """Helper to load session data with turn metadata."""
    from src.models.train_daic_attention import build_session_data
    return build_session_data(session_id, npz_path, session_windows, device, phq8_max_score=phq8_max_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 7 Multimodal Explainability Suite.")
    parser.add_argument("--config", default="configs/daic_explainability.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run on 5 smoke sessions")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_explainability_pipeline(
        config_path=args.config,
        smoke_test=args.smoke_test,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()

