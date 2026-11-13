"""Unit tests for Phase 5 DAIC-WOZ feature extraction and loading utilities."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml


def test_chunk_token_ids_splits_cleanly():
    from src.features.extract_daic_text import chunk_token_ids

    assert chunk_token_ids([], max_content_tokens=510) == [[]]
    assert chunk_token_ids([1, 2, 3], max_content_tokens=510) == [[1, 2, 3]]
    chunks = chunk_token_ids(list(range(1025)), max_content_tokens=510)
    assert [len(c) for c in chunks] == [510, 510, 5]
    assert [tok for ch in chunks for tok in ch] == list(range(1025))


def test_encode_texts_chunked_matches_single_pass_within_limit():
    """Single-chunk encoding matches direct masked mean pooling across all non-pad tokens."""
    from src.features.extract_daic_text import encode_texts_chunked

    class DummyTokenizer:
        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False):
            # Map each whitespace token deterministically to an integer > 2
            return [10 + (sum(ord(c) for c in tok) % 50) for tok in text.split()]

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.emb = torch.nn.Embedding(120, 4)
            with torch.no_grad():
                torch.manual_seed(0)
                self.emb.weight.normal_()

        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    tok = DummyTokenizer()
    enc = DummyEncoder().eval()

    short_text = "one two three four"
    long_text = " ".join(f"w{i}" for i in range(15))

    vecs = encode_texts_chunked(
        [short_text, "", long_text],
        tokenizer=tok,
        model=enc,
        device=torch.device("cpu"),
        max_length=8,   # forces 6 content tokens/chunk -> 3 chunks for 15 tokens
        batch_size=2,
    )
    assert vecs.shape == (3, 4)
    np.testing.assert_allclose(vecs[1], np.zeros(4, dtype=np.float32))

    # Short text has 4 tokens -> fits in 1 chunk (4 + 2 = 6 <= 8)
    ids = [tok.cls_token_id, *tok.encode(short_text), tok.sep_token_id]
    with torch.no_grad():
        expected_short = enc(torch.tensor([ids])).last_hidden_state.squeeze(0).mean(dim=0).numpy()
    np.testing.assert_allclose(vecs[0], expected_short, atol=1e-6)
    assert not np.isnan(vecs).any()


def test_combine_turn_moments_matches_raw_concat_pooling():
    """Analytical parallel moments across variable-length turns equal full-sequence mean+std."""
    from src.features.extract_daic_audio import (
        combine_turn_moments,
        compute_hidden_moments,
        pool_hidden_mean_std,
    )

    rng = torch.Generator().manual_seed(123)
    turn1 = torch.randn(1, 17, 8, generator=rng)
    turn2 = torch.randn(1, 41, 8, generator=rng)
    turn3 = torch.randn(1, 9, 8, generator=rng)

    m1, v1, c1 = compute_hidden_moments(turn1)
    m2, v2, c2 = compute_hidden_moments(turn2)
    m3, v3, c3 = compute_hidden_moments(turn3)

    combined = combine_turn_moments(
        np.stack([m1, m2, m3]),
        np.stack([v1, v2, v3]),
        np.array([c1, c2, c3]),
    )
    concat_frames = torch.cat([turn1, turn2, turn3], dim=1)
    expected = pool_hidden_mean_std(concat_frames)
    np.testing.assert_allclose(combined, expected, atol=1e-5)


def test_deduplicate_and_load_daic_cache_roundtrip(tmp_path):
    """Deduplicated session .npz caches align row-for-row back onto windows."""
    from src.features.extract_daic_text import (
        deduplicate_session_windows,
        load_daic_text_embeddings,
        make_window_key,
    )

    win_df = pd.DataFrame(
        {
            "session_id": [300, 300, 300],
            "window_id": ["300_030s_0000", "300_030s_0001", "300_030s_0002"],
            "scale": [30, 30, 30],
            "start_time": [10.0, 10.0, 50.0],
            "end_time": [40.0, 40.0, 80.0],
            "text": ["a", "a", "b"],
            "duration": [30.0, 30.0, 30.0],
            "label": [0.0, 0.0, 0.0],
            "split": ["dev", "dev", "dev"],
        }
    )
    uniq, inv = deduplicate_session_windows(win_df)
    assert len(uniq) == 2
    np.testing.assert_array_equal(inv, np.array([0, 0, 1]))

    win_path = tmp_path / "windows.parquet"
    win_df.to_parquet(str(win_path), index=False)

    cache_root = tmp_path / "features"
    model_dir = cache_root / "text_roberta-base"
    model_dir.mkdir(parents=True)
    emb_uniq = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np.savez(
        model_dir / "300.npz",
        window_keys=np.array([make_window_key(10.0, 40.0), make_window_key(50.0, 80.0)]),
        embeddings=emb_uniq,
    )

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"windows_path": str(win_path)},
                "features": {
                    "cache_dir": str(cache_root),
                    "text": {"model_name": "roberta-base"},
                },
            }
        ),
        encoding="utf-8",
    )

    X_loaded, df_loaded = load_daic_text_embeddings(
        config_path=str(cfg_path), split="dev", variant="participant", model_name_override="roberta-base"
    )
    assert X_loaded.shape == (3, 3)
    assert len(df_loaded) == 3
    np.testing.assert_allclose(X_loaded[0], emb_uniq[0])
    np.testing.assert_allclose(X_loaded[1], emb_uniq[0])
    np.testing.assert_allclose(X_loaded[2], emb_uniq[1])


def test_with_ellie_handles_no_ellie_session_safely():
    """Sessions 451/458/480 without Ellie turns must not crash ablation text builder."""
    from src.features.extract_daic_text import build_with_ellie_window_texts

    win_path = Path("data/processed/daic/windows.parquet")
    prep_path = Path("configs/daic_preprocess.yaml")
    if not win_path.exists() or not prep_path.exists():
        return

    with open(prep_path, "r", encoding="utf-8") as f:
        prep_cfg = yaml.safe_load(f)
    w = pd.read_parquet(str(win_path))
    sub = w[w["session_id"] == 451].head(3)
    if sub.empty:
        return
    texts = build_with_ellie_window_texts(451, sub, prep_cfg, include_speaker_tags=False)
    assert len(texts) == len(sub)
    assert all(isinstance(t, str) and len(t.strip()) > 0 for t in texts)

