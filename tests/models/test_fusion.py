"""Unit and integration tests for Phase 3 EATD late-fusion pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


CONFIG_PATH = "configs/eatd_fusion.yaml"


def _has_all_cached_embeddings() -> bool:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cache_dir = Path(cfg["features"]["cache_dir"])
    xlsr = cfg["features"]["audio"]["model_name"].split("/")[-1]
    wavlm = cfg["features"]["audio"]["alt_model_name"].split("/")[-1]
    return (
        (cache_dir / "text").exists()
        and (cache_dir / f"audio_{xlsr}").exists()
        and (cache_dir / f"audio_{wavlm}").exists()
    )


@pytest.mark.skipif(not _has_all_cached_embeddings(), reason="Cached embeddings missing")
def test_build_fused_features_shapes_and_alignment():
    """Fused vectors must equal concatenated text and audio dims with identical PIDs/labels."""
    from src.features.extract_audio import load_audio_embeddings
    from src.features.extract_text import load_text_embeddings
    from src.models.train_eatd_fusion import build_fused_features

    X_t, y_t, pids_t = load_text_embeddings(CONFIG_PATH, aggregation="mean")
    X_a, y_a, pids_a = load_audio_embeddings(CONFIG_PATH, encoder_key="model_name", aggregation="mean")
    X_f, y_f, pids_f = build_fused_features(CONFIG_PATH, encoder_key="model_name", aggregation="mean")

    assert pids_f == pids_t == pids_a
    np.testing.assert_array_equal(y_f, y_t)
    np.testing.assert_array_equal(y_f, y_a)
    assert X_f.shape == (162, 768 + 2048)
    np.testing.assert_allclose(X_f[:, :768], X_t)
    np.testing.assert_allclose(X_f[:, 768:], X_a)
    assert not np.isnan(X_f).any()


@pytest.mark.skipif(not _has_all_cached_embeddings(), reason="Cached embeddings missing")
def test_build_fused_features_concat_shape():
    """Concat-mode fusion should have shape (162, 2304 + 4608) for WavLM."""
    from src.models.train_eatd_fusion import build_fused_features

    X_f, y_f, pids_f = build_fused_features(
        CONFIG_PATH, encoder_key="alt_model_name", aggregation="concat",
    )
    assert X_f.shape == (162, 2304 + 4608)
    assert len(pids_f) == len(y_f) == 162
    assert not np.isnan(X_f).any()


def test_build_comparison_table_combines_sources():
    """Comparison table builder uses primary mean aggregation consistently and marks concat secondary."""
    from src.models.train_eatd_fusion import build_comparison_table

    dummy_runs = pd.DataFrame([
        {
            "aggregation": agg,
            "modality": "fusion_text+xlsr",
            "model": "mlp",
            "fold": f,
            "seed": s,
            "macro_f1": 0.60 + 0.01 * s_idx,
            "dep_f1": 0.35,
            "accuracy": 0.80,
            "auc_roc": 0.65,
        }
        for agg in ("mean", "concat")
        for f in range(5)
        for s_idx, s in enumerate([42, 123, 456])
    ])

    comp_df, plot_df = build_comparison_table(dummy_runs, config_path=CONFIG_PATH)
    assert "majority_baseline" in set(comp_df["source"])
    assert "v2_baseline" in set(comp_df["source"])
    assert "v1_baseline" not in set(comp_df["source"])

    primary_fusion = comp_df[(comp_df["source"] == "v3_fusion") & (comp_df["role"] == "primary")]
    secondary_fusion = comp_df[(comp_df["source"] == "v3_fusion") & (comp_df["role"] == "secondary")]
    assert len(primary_fusion) == 1
    assert primary_fusion.iloc[0]["aggregation"] == "mean"
    assert primary_fusion.iloc[0]["n_runs"] == 15
    assert np.isclose(primary_fusion.iloc[0]["macro_f1_mean"], 0.61)
    assert primary_fusion.iloc[0]["macro_f1_seed_std"] > 0.0
    assert len(secondary_fusion) == 1
    assert secondary_fusion.iloc[0]["aggregation"] == "concat"
    assert not plot_df.empty


def test_fusion_config_paths_in_results_eatd():
    """Verify all Phase 3 output paths live under results/eatd/ and do not overwrite v1/v2."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tr = cfg["training"]
    for key in ("results_runs_csv", "results_comparison_csv", "results_predictions_csv", "results_plot"):
        assert tr[key].startswith("results/eatd/"), f"{key}={tr[key]} should be in results/eatd/"
    assert tr["results_runs_csv"] == "results/eatd/eatd_fusion.csv"
    assert tr["results_comparison_csv"] == "results/eatd/eatd_comparison.csv"
    assert tr["results_predictions_csv"] == "results/eatd/eatd_fusion_predictions.csv"
    assert tr["results_runs_csv"] not in (
        "results/eatd/eatd_baselines.csv",
        "results/eatd/eatd_baselines_v2.csv",
    )

