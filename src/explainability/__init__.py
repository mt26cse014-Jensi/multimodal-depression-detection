"""
Explainability and limitation analysis suite for multimodal depression detection (Phase 7).
"""

from src.explainability.turn_occlusion import (
    occlude_session_turns,
    run_turn_occlusion_suite,
)
from src.explainability.shap_analysis import (
    compute_exact_2group_shap,
    run_shap_modality_suite,
)
from src.explainability.attention_maps import (
    extract_cross_modal_attention_maps,
    plot_example_attention_heatmaps,
)
from src.explainability.shortcut_analysis import (
    analyze_interview_temporal_shortcuts,
    plot_shortcut_analysis,
)

__all__ = [
    "occlude_session_turns",
    "run_turn_occlusion_suite",
    "compute_exact_2group_shap",
    "run_shap_modality_suite",
    "extract_cross_modal_attention_maps",
    "plot_example_attention_heatmaps",
    "analyze_interview_temporal_shortcuts",
    "plot_shortcut_analysis",
]

