"""Tests for feature extraction and loading functions.

These tests run on cached embeddings (fast), not the extraction itself.
They verify shapes, label consistency, and NaN absence.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


# ── Helpers ──────────────────────────────────────────────────────────

CONFIG_PATH = "configs/eatd_preprocess.yaml"


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _has_cached_text() -> bool:
    """Check if text embeddings are cached."""
    cfg = _load_config()
    cache_dir = Path(cfg["features"]["cache_dir"]) / "text"
    return cache_dir.exists() and any(cache_dir.glob("*.npy"))


def _has_cached_audio(encoder_key: str = "model_name") -> bool:
    """Check if audio embeddings are cached for the given encoder."""
    cfg = _load_config()
    model_name = cfg["features"]["audio"][encoder_key]
    short_name = model_name.split("/")[-1]
    cache_dir = Path(cfg["features"]["cache_dir"]) / f"audio_{short_name}"
    return cache_dir.exists() and any(cache_dir.glob("*.npy"))


# ── Text embedding tests ────────────────────────────────────────────

@pytest.mark.skipif(not _has_cached_text(), reason="Text embeddings not cached yet")
class TestTextEmbeddings:
    """Tests for cached text embeddings."""

    def test_load_shape(self):
        """Text embeddings should be (N, 2304) for 3 × 768-dim Chinese RoBERTa."""
        from src.features.extract_text import load_text_embeddings
        X, y, pids = load_text_embeddings(CONFIG_PATH)
        assert X.ndim == 2
        assert X.shape[1] == 2304, f"Expected 2304-dim, got {X.shape[1]}"
        assert len(y) == len(pids) == X.shape[0]

    def test_labels_match_manifest(self):
        """Labels from embeddings must match the manifest."""
        from src.features.extract_text import load_text_embeddings
        cfg = _load_config()
        manifest = pd.read_csv(cfg["dataset"]["manifest_path"])
        X, y, pids = load_text_embeddings(CONFIG_PATH)

        for pid, label in zip(pids, y):
            expected = manifest.loc[manifest.participant_id == pid, "label"].values[0]
            assert label == expected, f"Label mismatch for {pid}: got {label}, expected {expected}"

    def test_no_nans(self):
        """Embeddings should not contain NaN values."""
        from src.features.extract_text import load_text_embeddings
        X, y, pids = load_text_embeddings(CONFIG_PATH)
        assert not np.isnan(X).any(), "Text embeddings contain NaN values"

    def test_all_participants_covered(self):
        """All manifest participants should have cached embeddings."""
        from src.features.extract_text import load_text_embeddings
        cfg = _load_config()
        manifest = pd.read_csv(cfg["dataset"]["manifest_path"])
        X, y, pids = load_text_embeddings(CONFIG_PATH)
        assert set(pids) == set(manifest.participant_id), "Not all participants have embeddings"


# ── Audio embedding tests ────────────────────────────────────────────

@pytest.mark.skipif(not _has_cached_audio("model_name"), reason="Primary audio embeddings not cached yet")
class TestPrimaryAudioEmbeddings:
    """Tests for cached audio embeddings (primary encoder: XLS-R 300m)."""

    def test_load_shape(self):
        """Audio embeddings should be (N, 6144) for XLS-R (3 × 2 × 1024)."""
        from src.features.extract_audio import load_audio_embeddings
        X, y, pids = load_audio_embeddings(CONFIG_PATH, encoder_key="model_name")
        assert X.ndim == 2
        assert X.shape[1] == 6144, f"Expected 6144-dim for XLS-R, got {X.shape[1]}"
        assert len(y) == len(pids) == X.shape[0]

    def test_labels_match_manifest(self):
        from src.features.extract_audio import load_audio_embeddings
        cfg = _load_config()
        manifest = pd.read_csv(cfg["dataset"]["manifest_path"])
        X, y, pids = load_audio_embeddings(CONFIG_PATH, encoder_key="model_name")

        for pid, label in zip(pids, y):
            expected = manifest.loc[manifest.participant_id == pid, "label"].values[0]
            assert label == expected, f"Label mismatch for {pid}"

    def test_no_nans(self):
        from src.features.extract_audio import load_audio_embeddings
        X, y, pids = load_audio_embeddings(CONFIG_PATH, encoder_key="model_name")
        assert not np.isnan(X).any(), "Audio embeddings contain NaN values"


@pytest.mark.skipif(not _has_cached_audio("alt_model_name"), reason="Alt audio embeddings not cached yet")
class TestAltAudioEmbeddings:
    """Tests for cached audio embeddings (comparison encoder: WavLM)."""

    def test_load_shape(self):
        """Audio embeddings should be (N, 4608) for WavLM (3 × 2 × 768)."""
        from src.features.extract_audio import load_audio_embeddings
        X, y, pids = load_audio_embeddings(CONFIG_PATH, encoder_key="alt_model_name")
        assert X.ndim == 2
        assert X.shape[1] == 4608, f"Expected 4608-dim for WavLM, got {X.shape[1]}"
        assert len(y) == len(pids) == X.shape[0]

    def test_no_nans(self):
        from src.features.extract_audio import load_audio_embeddings
        X, y, pids = load_audio_embeddings(CONFIG_PATH, encoder_key="alt_model_name")
        assert not np.isnan(X).any(), "Audio embeddings contain NaN values"


# ── Aggregation & mean-pooling tests ─────────────────────────────────

class TestAggregationUnit:
    """Unit tests for answer embedding aggregation helper."""

    def test_concat_mode_preserves_vector(self):
        from src.features.aggregation import aggregate_answer_embeddings
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        out = aggregate_answer_embeddings(vec, n_answers=3, mode="concat")
        np.testing.assert_allclose(out, vec)

    def test_mean_mode_without_mask(self):
        from src.features.aggregation import aggregate_answer_embeddings
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        out = aggregate_answer_embeddings(vec, n_answers=3, mode="mean")
        np.testing.assert_allclose(out, np.array([3.0, 4.0], dtype=np.float32))

    def test_mean_mode_with_dropped_mask(self):
        from src.features.aggregation import aggregate_answer_embeddings
        vec = np.array([2.0, 4.0, 0.0, 0.0, 6.0, 8.0], dtype=np.float32)
        out = aggregate_answer_embeddings(
            vec, n_answers=3, mode="mean", valid_mask=[True, False, True],
        )
        np.testing.assert_allclose(out, np.array([4.0, 6.0], dtype=np.float32))


@pytest.mark.skipif(not _has_cached_text(), reason="Text embeddings not cached yet")
def test_load_text_embeddings_mean():
    """Mean-aggregated text embeddings should have shape (162, 768) and match slice mean."""
    from src.features.extract_text import load_text_embeddings
    X_concat, y_concat, pids_concat = load_text_embeddings(CONFIG_PATH, aggregation="concat")
    X_mean, y_mean, pids_mean = load_text_embeddings(CONFIG_PATH, aggregation="mean")

    assert pids_concat == pids_mean
    np.testing.assert_array_equal(y_concat, y_mean)
    assert X_mean.shape == (len(pids_mean), 768)
    expected = X_concat.reshape(len(pids_mean), 3, 768).mean(axis=1)
    np.testing.assert_allclose(X_mean, expected, atol=1e-6)


@pytest.mark.skipif(not _has_cached_audio("model_name"), reason="Primary audio embeddings not cached yet")
def test_load_audio_embeddings_mean_handles_dropped_clips():
    """Mean audio loading skips dropped clips (v_23, v_73, v_79) and matches valid mean."""
    from src.features.extract_audio import load_audio_embeddings
    X_concat, _, pids = load_audio_embeddings(CONFIG_PATH, encoder_key="model_name", aggregation="concat")
    X_mean, _, _ = load_audio_embeddings(CONFIG_PATH, encoder_key="model_name", aggregation="mean")

    assert X_mean.shape == (len(pids), 2048)

    # v_79 had positive (slice 0) dropped -> average slices 1 and 2
    idx_v79 = pids.index("v_79")
    slices_v79 = X_concat[idx_v79].reshape(3, 2048)
    np.testing.assert_allclose(slices_v79[0], 0.0)
    np.testing.assert_allclose(X_mean[idx_v79], slices_v79[1:].mean(axis=0), atol=1e-6)

    # v_23 and v_73 had negative (slice 2) dropped -> average slices 0 and 1
    for pid in ("v_23", "v_73"):
        idx = pids.index(pid)
        slices = X_concat[idx].reshape(3, 2048)
        np.testing.assert_allclose(slices[2], 0.0)
        np.testing.assert_allclose(X_mean[idx], slices[:2].mean(axis=0), atol=1e-6)


@pytest.mark.skipif(not _has_cached_audio("alt_model_name"), reason="Alt audio embeddings not cached yet")
def test_load_alt_audio_embeddings_mean_shape():
    """Mean WavLM embeddings should have shape (162, 1536)."""
    from src.features.extract_audio import load_audio_embeddings
    X_mean, y_mean, pids = load_audio_embeddings(CONFIG_PATH, encoder_key="alt_model_name", aggregation="mean")
    assert X_mean.shape == (len(pids), 1536)
    assert not np.isnan(X_mean).any()

