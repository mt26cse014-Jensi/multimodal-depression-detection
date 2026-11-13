"""Extract and cache frozen WavLM/Wav2Vec2 audio embeddings for DAIC-WOZ windows (Phase 5).

Supports two extraction modes writing the exact same downstream per-session cache:
  - ``turn_moments`` (CPU-friendly): runs the frozen speech encoder once per cut
    16 kHz participant turn WAV, stores per-turn ``(mean, var_pop, n_frames)``,
    and analytically combines constituent turn moments into exact per-window
    ``(mean, std)`` vectors of shape ``(2 * hidden_size,)``.
  - ``window_concat`` (Colab GPU recommended): concatenates cut turn WAVs within
    each unique window interval, runs the speech encoder directly across the
    window waveform, and applies ``mean + std`` pooling over time.

Features:
  - Resumable per session with atomic writes (``.tmp.npz`` -> ``.npz``).
  - Deduplicates repeated ``(start_time, end_time)`` windows per session.
  - Colab-compatible: ``--device auto``, ``--fp16``, ``--split``, ``--sessions``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from transformers import AutoFeatureExtractor, AutoModel

from src.features.extract_daic_text import (
    _get_smoke_test_sessions,
    deduplicate_session_windows,
    load_config,
    make_window_key,
    model_short_name,
    resolve_device,
)

logger = logging.getLogger(__name__)


# ── Moments & pooling math ─────────────────────────────────────────────────────

def compute_hidden_moments(hidden_states: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute per-channel mean, population variance, and frame count from ``(1, T, D)``."""
    h = hidden_states.squeeze(0).float()  # (T, D)
    n_frames = int(h.shape[0])
    if n_frames == 0:
        d = int(h.shape[1])
        return np.zeros(d, dtype=np.float32), np.zeros(d, dtype=np.float32), 0
    mean = h.mean(dim=0).cpu().numpy().astype(np.float32)
    var_pop = h.var(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
    return mean, var_pop, n_frames


def combine_turn_moments(
    means: np.ndarray,
    vars_pop: np.ndarray,
    frame_counts: np.ndarray,
) -> np.ndarray:
    """Combine turn-level ``(mean, var_pop, n_frames)`` into global ``[mean, std]``.

    Uses exact first and second raw moments so combining turns reproduces
    concatenated-frame population ``mean`` and ``std`` identically.
    """
    if len(frame_counts) == 0 or int(np.sum(frame_counts)) <= 0:
        d = means.shape[1] if means.ndim == 2 and means.shape[1] > 0 else 768
        return np.zeros(2 * d, dtype=np.float32)

    weights = frame_counts.astype(np.float64)
    total = float(weights.sum())
    means_f = means.astype(np.float64)
    vars_f = np.maximum(vars_pop.astype(np.float64), 0.0)

    global_mean = np.sum(means_f * weights[:, None], axis=0) / total
    second_moment = np.sum((vars_f + means_f ** 2) * weights[:, None], axis=0) / total
    global_var = np.maximum(second_moment - global_mean ** 2, 0.0)
    global_std = np.sqrt(global_var)
    return np.concatenate([global_mean, global_std]).astype(np.float32)


def pool_hidden_mean_std(hidden_states: torch.Tensor) -> np.ndarray:
    """Compute ``[mean, std]`` (population std) over time for ``(1, T, D)``."""
    mean, var_pop, n_frames = compute_hidden_moments(hidden_states)
    if n_frames == 0:
        return np.zeros(2 * mean.shape[0], dtype=np.float32)
    std = np.sqrt(np.maximum(var_pop, 0.0)).astype(np.float32)
    return np.concatenate([mean, std]).astype(np.float32)


# ── Turn & audio helpers ───────────────────────────────────────────────────────

def load_turn_wav_16k(wav_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load a cut turn WAV clip as mono float32 at ``target_sr``."""
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != target_sr:
        t = torch.from_numpy(mono).unsqueeze(0)
        mono = torchaudio.transforms.Resample(sr, target_sr)(t).squeeze(0).numpy()
    return mono.astype(np.float32, copy=False)


def match_turns_to_window(
    turns_df: pd.DataFrame,
    start_time: float,
    end_time: float,
    tol: float = 1e-3,
) -> list[int]:
    """Return sorted row indices of ``turns_df`` falling inside ``[start_time, end_time]``."""
    t_starts = turns_df["start_time"].to_numpy(dtype=float)
    t_stops = turns_df["stop_time"].to_numpy(dtype=float)
    mask = (t_starts >= float(start_time) - tol) & (t_stops <= float(end_time) + tol)
    return np.where(mask)[0].tolist()


def _load_speech_encoder(model_name: str, device: torch.device) -> tuple[Any, Any]:
    """Load HF speech feature extractor and model, preferring local cache when cached."""
    try:
        fe = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except Exception:
        hf_token = os.environ.get("HF_TOKEN") or None
        fe = AutoFeatureExtractor.from_pretrained(model_name, token=hf_token)
        model = AutoModel.from_pretrained(model_name, token=hf_token)

    model.eval()
    model.to(device)
    return fe, model


def _audio_cache_dir(cache_root: Path, model_name: str) -> Path:
    return cache_root / f"audio_{model_short_name(model_name)}"


# ── Per-session extraction ─────────────────────────────────────────────────────

def _extract_session_turn_moments(
    sid: int,
    unique_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    turns_session_dir: Path,
    turn_cache_path: Path,
    feature_extractor: Any,
    model: torch.nn.Module,
    device: torch.device,
    target_sr: int,
    use_fp16: bool,
) -> np.ndarray:
    """Compute (or load cached) per-turn moments and combine per unique window."""
    hidden_size = int(model.config.hidden_size)
    turns_df = turns_df.sort_values("start_time").reset_index(drop=True)

    if turn_cache_path.exists():
        with np.load(turn_cache_path, allow_pickle=False) as d:
            turn_ids = d["turn_ids"].astype(int)
            means = d["means"].astype(np.float32)
            vars_pop = d["vars_pop"].astype(np.float32)
            counts = d["frame_counts"].astype(np.int64)
    else:
        n_turns = len(turns_df)
        turn_ids = turns_df["turn_id"].to_numpy(dtype=np.int64)
        means = np.zeros((n_turns, hidden_size), dtype=np.float32)
        vars_pop = np.zeros((n_turns, hidden_size), dtype=np.float32)
        counts = np.zeros(n_turns, dtype=np.int64)
        amp_enabled = bool(use_fp16 and device.type == "cuda")

        for idx, row in enumerate(turns_df.itertuples()):
            wav_path = turns_session_dir / f"{sid}_{int(row.turn_id):04d}.wav"
            if not wav_path.exists():
                continue
            wav = load_turn_wav_16k(wav_path, target_sr=target_sr)
            if len(wav) < int(0.05 * target_sr):
                continue

            inputs = feature_extractor(wav, sampling_rate=target_sr, return_tensors="pt", padding=False)
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                if amp_enabled:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(input_values)
                else:
                    outputs = model(input_values)
            m, v, c = compute_hidden_moments(outputs.last_hidden_state)
            means[idx] = m
            vars_pop[idx] = v
            counts[idx] = c

        turn_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = turn_cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, turn_ids=turn_ids, means=means, vars_pop=vars_pop, frame_counts=counts)
        tmp.replace(turn_cache_path)

    embs = np.zeros((len(unique_df), 2 * hidden_size), dtype=np.float32)
    for w_i, w_row in enumerate(unique_df.itertuples()):
        t_idx = match_turns_to_window(turns_df, w_row.start_time, w_row.end_time)
        if not t_idx:
            continue
        embs[w_i] = combine_turn_moments(means[t_idx], vars_pop[t_idx], counts[t_idx])
    return embs


def _extract_session_window_concat(
    sid: int,
    unique_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    turns_session_dir: Path,
    feature_extractor: Any,
    model: torch.nn.Module,
    device: torch.device,
    target_sr: int,
    max_duration_sec: float,
    use_fp16: bool,
) -> np.ndarray:
    """Concatenate constituent turn WAVs per window and run full-window WavLM."""
    hidden_size = int(model.config.hidden_size)
    turns_df = turns_df.sort_values("start_time").reset_index(drop=True)
    max_samples = int(max_duration_sec * target_sr)
    amp_enabled = bool(use_fp16 and device.type == "cuda")

    # Preload session turn WAVs once
    wav_cache: dict[int, np.ndarray] = {}
    for row in turns_df.itertuples():
        tid = int(row.turn_id)
        p = turns_session_dir / f"{sid}_{tid:04d}.wav"
        if p.exists():
            wav_cache[tid] = load_turn_wav_16k(p, target_sr=target_sr)

    embs = np.zeros((len(unique_df), 2 * hidden_size), dtype=np.float32)
    for w_i, w_row in enumerate(unique_df.itertuples()):
        t_idx = match_turns_to_window(turns_df, w_row.start_time, w_row.end_time)
        clips = [wav_cache[int(turns_df.iloc[j]["turn_id"])] for j in t_idx if int(turns_df.iloc[j]["turn_id"]) in wav_cache]
        if not clips:
            continue
        full_wav = np.concatenate(clips, axis=0)
        if len(full_wav) > max_samples:
            full_wav = full_wav[:max_samples]

        inputs = feature_extractor(full_wav, sampling_rate=target_sr, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_values)
            else:
                outputs = model(input_values)
        embs[w_i] = pool_hidden_mean_std(outputs.last_hidden_state)

    return embs


# ── Extraction & loading ───────────────────────────────────────────────────────

def extract_daic_audio_embeddings(
    config_path: str = "configs/daic_models.yaml",
    mode_override: Optional[str] = None,
    split_filter: Optional[str] = None,
    session_subset: Optional[Sequence[int]] = None,
    device_override: Optional[str] = None,
    fp16_override: Optional[bool] = None,
) -> Path:
    """Extract and cache per-session window audio embeddings for DAIC-WOZ."""
    cfg = load_config(config_path)
    feat_cfg = cfg["features"]
    audio_cfg = feat_cfg["audio"]

    model_name = audio_cfg["model_name"]
    mode = mode_override or audio_cfg.get("extraction_mode", "turn_moments")
    if mode not in ("turn_moments", "window_concat"):
        raise ValueError(f"Unsupported audio extraction mode: {mode!r}")

    target_sr = int(audio_cfg.get("sample_rate", 16000))
    max_dur = float(audio_cfg.get("max_duration_sec", 132.0))
    use_fp16 = bool(audio_cfg.get("fp16", False) if fp16_override is None else fp16_override)
    device = resolve_device(device_override or audio_cfg.get("device", "auto"))

    cache_root = Path(feat_cfg["cache_dir"])
    cache_dir = _audio_cache_dir(cache_root, model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    turn_moments_dir = cache_root / f"audio_{model_short_name(model_name)}_turn_moments"

    windows = pd.read_parquet(cfg["dataset"]["windows_path"])
    if split_filter is not None:
        windows = windows[windows["split"] == split_filter].copy()
    if session_subset is not None:
        sid_set = {int(s) for s in session_subset}
        windows = windows[windows["session_id"].astype(int).isin(sid_set)].copy()

    if windows.empty:
        logger.warning("No windows found for split=%s, sessions=%s", split_filter, session_subset)
        return cache_dir

    session_ids = sorted(windows["session_id"].astype(int).unique().tolist())
    pending = [sid for sid in session_ids if not (cache_dir / f"{sid}.npz").exists()]

    if not pending:
        logger.info("All %d DAIC audio sessions already cached in %s.", len(session_ids), cache_dir)
        return cache_dir

    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 4))

    logger.info(
        "Extracting DAIC audio embeddings (mode=%s): %d/%d sessions | model=%s | device=%s",
        mode, len(pending), len(session_ids), model_name, device,
    )

    fe, model = _load_speech_encoder(model_name, device=device)
    hidden_size = int(model.config.hidden_size)
    turns_root = Path(cfg["dataset"]["turns_dir"])

    meta_path = cache_dir / "metadata.json"
    meta_payload = {
        "model_name": model_name,
        "extraction_mode": mode,
        "sample_rate": target_sr,
        "hidden_size": hidden_size,
        "pooled_dim": 2 * hidden_size,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    t0 = time.time()
    for i, sid in enumerate(pending):
        sub = windows[windows["session_id"].astype(int) == sid].copy()
        unique_df, _ = deduplicate_session_windows(sub)

        turns_csv = turns_root / f"{sid}_turns.csv"
        if not turns_csv.exists():
            raise FileNotFoundError(f"Turn table not found for session {sid}: {turns_csv}")
        turns_df = pd.read_csv(turns_csv)
        turns_session_dir = turns_root / str(sid)

        if mode == "turn_moments":
            embs = _extract_session_turn_moments(
                sid=sid,
                unique_df=unique_df,
                turns_df=turns_df,
                turns_session_dir=turns_session_dir,
                turn_cache_path=turn_moments_dir / f"{sid}.npz",
                feature_extractor=fe,
                model=model,
                device=device,
                target_sr=target_sr,
                use_fp16=use_fp16,
            )
        else:
            embs = _extract_session_window_concat(
                sid=sid,
                unique_df=unique_df,
                turns_df=turns_df,
                turns_session_dir=turns_session_dir,
                feature_extractor=fe,
                model=model,
                device=device,
                target_sr=target_sr,
                max_duration_sec=max_dur,
                use_fp16=use_fp16,
            )

        out_path = cache_dir / f"{sid}.npz"
        tmp_path = cache_dir / f"{sid}.tmp.npz"
        np.savez(
            tmp_path,
            window_keys=np.array(unique_df["window_key"].tolist(), dtype=str),
            embeddings=embs.astype(np.float32, copy=False),
        )
        tmp_path.replace(out_path)

        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (len(pending) - i - 1) / rate if rate > 0 else 0.0
        logger.info(
            "  [%d/%d] session %d (%d unique windows) | %.1fs elapsed, ~%.0fs ETA",
            i + 1, len(pending), sid, len(unique_df), elapsed, eta,
        )

    return cache_dir


def load_daic_audio_embeddings(
    config_path: str = "configs/daic_models.yaml",
    split: Optional[str] = None,
    scales: Optional[Sequence[int]] = None,
    session_subset: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load cached DAIC audio embeddings aligned row-for-row with ``windows.parquet``."""
    cfg = load_config(config_path)
    cache_root = Path(cfg["features"]["cache_dir"])
    model_name = cfg["features"]["audio"]["model_name"]
    cache_dir = _audio_cache_dir(cache_root, model_name)

    windows = pd.read_parquet(cfg["dataset"]["windows_path"])
    if split is not None:
        windows = windows[windows["split"] == split].copy()
    if scales is not None:
        scale_set = {int(s) for s in scales}
        windows = windows[windows["scale"].astype(int).isin(scale_set)].copy()
    if session_subset is not None:
        sid_set = {int(s) for s in session_subset}
        windows = windows[windows["session_id"].astype(int).isin(sid_set)].copy()

    windows = windows.reset_index(drop=True)
    if windows.empty:
        raise ValueError(f"No windows match filter split={split}, scales={scales}, sessions={session_subset}")

    session_cache: dict[int, tuple[dict[str, int], np.ndarray]] = {}
    missing: list[int] = []
    for sid in sorted(windows["session_id"].astype(int).unique().tolist()):
        npz_path = cache_dir / f"{sid}.npz"
        if not npz_path.exists():
            missing.append(sid)
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            keys = [str(k) for k in data["window_keys"].tolist()]
            embs = data["embeddings"].astype(np.float32, copy=False)
        session_cache[sid] = ({k: i for i, k in enumerate(keys)}, embs)

    if missing:
        raise FileNotFoundError(
            f"Missing cached DAIC audio embeddings in {cache_dir} for "
            f"{len(missing)} sessions: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    first_emb = next(iter(session_cache.values()))[1]
    pooled_dim = int(first_emb.shape[1])
    X = np.empty((len(windows), pooled_dim), dtype=np.float32)

    for row_idx, row in enumerate(windows.itertuples()):
        sid = int(row.session_id)
        key = make_window_key(row.start_time, row.end_time)
        key_map, embs = session_cache[sid]
        if key not in key_map:
            raise KeyError(f"Window key {key} missing from {cache_dir / f'{sid}.npz'}")
        X[row_idx] = embs[key_map[key]]

    logger.info("Loaded DAIC audio embeddings (split=%s): X=%s", split or "all", X.shape)
    return X, windows


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DAIC-WOZ window audio embeddings.")
    parser.add_argument("--config", default="configs/daic_models.yaml")
    parser.add_argument("--mode", default=None, choices=["turn_moments", "window_concat"])
    parser.add_argument("--split", default=None, choices=["train", "dev", "test"])
    parser.add_argument("--sessions", nargs="+", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run on 5 stratified sessions only")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    sessions = args.sessions
    if args.smoke_test:
        sessions = _get_smoke_test_sessions(cfg["dataset"]["manifest_path"])
        logger.info("Smoke test sessions: %s", sessions)

    extract_daic_audio_embeddings(
        config_path=args.config,
        mode_override=args.mode,
        split_filter=args.split,
        session_subset=sessions,
        device_override=args.device,
        fp16_override=args.fp16 if args.fp16 else None,
    )


if __name__ == "__main__":
    main()

