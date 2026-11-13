"""
Phase 6: Turn-level sequence feature extractor for DAIC-WOZ.

Extracts sequence-level representations per turn:
- Text token embeddings from RoBERTa-base (float16, shape [L_k, 768])
- Audio frame embeddings from WavLM-base-plus downsampled to ~25 fps (float16, shape [T_k, 768])

Features are packed into contiguous arrays with offset indices per session and saved to:
  data/processed/daic/features_seq/turns/{session_id}.npz

License: Respects DAIC-WOZ restriction: no transcripts are saved or logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from src.features.extract_daic_text import (
    _get_smoke_test_sessions,
    load_config,
    resolve_device,
)

logger = logging.getLogger(__name__)


def extract_turn_sequences_for_session(
    session_id: int,
    turns_dir: Path | str,
    text_tokenizer: Any,
    text_model: torch.nn.Module,
    audio_feature_extractor: Any,
    audio_model: torch.nn.Module,
    device: torch.device,
    downsample_factor: int = 2,
    target_sample_rate: int = 16000,
) -> dict[str, np.ndarray]:
    """Extract sequence features for all turns of a single session.

    Returns a dict with:
        turn_ids: (N,) int32
        turn_durations: (N,) float32
        start_times: (N,) float32
        stop_times: (N,) float32
        text_embs: (sum(L), 768) float16
        text_offsets: (N + 1,) int32
        audio_embs: (sum(T), 768) float16
        audio_offsets: (N + 1,) int32
    """
    turns_dir = Path(turns_dir)
    turns_csv = turns_dir / f"{session_id}_turns.csv"
    if not turns_csv.exists():
        raise FileNotFoundError(f"Turns CSV not found for session {session_id}: {turns_csv}")

    turns_df = pd.read_csv(turns_csv)
    if turns_df.empty:
        logger.warning("Session %d has 0 turns in %s", session_id, turns_csv)
        return {
            "turn_ids": np.zeros(0, dtype=np.int32),
            "turn_durations": np.zeros(0, dtype=np.float32),
            "start_times": np.zeros(0, dtype=np.float32),
            "stop_times": np.zeros(0, dtype=np.float32),
            "text_embs": np.zeros((0, 768), dtype=np.float16),
            "text_offsets": np.zeros(1, dtype=np.int32),
            "audio_embs": np.zeros((0, 768), dtype=np.float16),
            "audio_offsets": np.zeros(1, dtype=np.int32),
        }

    session_wav_dir = turns_dir / str(session_id)

    turn_ids_list: list[int] = []
    durations_list: list[float] = []
    starts_list: list[float] = []
    stops_list: list[float] = []

    text_embs_list: list[np.ndarray] = []
    audio_embs_list: list[np.ndarray] = []

    text_offsets: list[int] = [0]
    audio_offsets: list[int] = [0]

    current_text_offset = 0
    current_audio_offset = 0

    text_model.eval()
    audio_model.eval()

    with torch.no_grad():
        for _, row in turns_df.iterrows():
            t_id = int(row["turn_id"])
            start_t = float(row["start_time"]) if pd.notna(row["start_time"]) else 0.0
            stop_t = float(row["stop_time"]) if pd.notna(row["stop_time"]) else 0.0
            dur = float(row["duration"]) if pd.notna(row["duration"]) else max(0.0, stop_t - start_t)
            raw_text = str(row["text"]).strip() if pd.notna(row["text"]) else ""

            # 1. Text token embeddings
            if not raw_text:
                tok_inputs = text_tokenizer(" ", return_tensors="pt")
            else:
                tok_inputs = text_tokenizer(
                    raw_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )

            input_ids = tok_inputs["input_ids"].to(device)
            attention_mask = tok_inputs["attention_mask"].to(device)
            text_outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
            # Shape: (1, L, 768) -> (L, 768)
            t_emb = text_outputs.last_hidden_state.squeeze(0).cpu().to(torch.float16).numpy()
            if t_emb.shape[0] == 0:
                t_emb = np.zeros((1, 768), dtype=np.float16)

            # 2. Audio frame embeddings
            wav_path = session_wav_dir / f"{session_id}_{t_id:04d}.wav"
            if wav_path.exists():
                try:
                    wav, sr = sf.read(str(wav_path))
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    if sr != target_sample_rate:
                        logger.warning("Resampling %s from %d to %d", wav_path, sr, target_sample_rate)
                        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                        wav = resampler(torch.from_numpy(wav).float()).numpy()
                except Exception as exc:
                    logger.warning("Error reading %s: %s; using zero audio frame", wav_path, exc)
                    wav = np.zeros(target_sample_rate // 4, dtype=np.float32)
            else:
                logger.warning("Missing turn WAV %s; using zero audio frame", wav_path)
                wav = np.zeros(target_sample_rate // 4, dtype=np.float32)

            # Ensure minimum length for WavLM receptive field (400 samples)
            if len(wav) < 400:
                wav = np.pad(wav, (0, 400 - len(wav)), mode="constant")

            audio_inputs = audio_feature_extractor(
                wav,
                sampling_rate=target_sample_rate,
                return_tensors="pt",
                padding=False,
            )
            input_values = audio_inputs.input_values.to(device)
            audio_outputs = audio_model(input_values)
            # Shape: (1, T_raw, 768)
            frames = audio_outputs.last_hidden_state
            if downsample_factor > 1 and frames.shape[1] >= downsample_factor:
                # 1D avg pool along time dimension: (1, 768, T) -> (1, 768, T // factor)
                frames_down = F.avg_pool1d(
                    frames.transpose(1, 2),
                    kernel_size=downsample_factor,
                    stride=downsample_factor,
                ).transpose(1, 2)
            else:
                frames_down = frames

            a_emb = frames_down.squeeze(0).cpu().to(torch.float16).numpy()
            if a_emb.shape[0] == 0:
                a_emb = np.zeros((1, 768), dtype=np.float16)

            # Append turn
            turn_ids_list.append(t_id)
            durations_list.append(dur)
            starts_list.append(start_t)
            stops_list.append(stop_t)

            text_embs_list.append(t_emb)
            current_text_offset += t_emb.shape[0]
            text_offsets.append(current_text_offset)

            audio_embs_list.append(a_emb)
            current_audio_offset += a_emb.shape[0]
            audio_offsets.append(current_audio_offset)

    concatenated_text = (
        np.concatenate(text_embs_list, axis=0).astype(np.float16)
        if text_embs_list
        else np.zeros((0, 768), dtype=np.float16)
    )
    concatenated_audio = (
        np.concatenate(audio_embs_list, axis=0).astype(np.float16)
        if audio_embs_list
        else np.zeros((0, 768), dtype=np.float16)
    )

    return {
        "turn_ids": np.array(turn_ids_list, dtype=np.int32),
        "turn_durations": np.array(durations_list, dtype=np.float32),
        "start_times": np.array(starts_list, dtype=np.float32),
        "stop_times": np.array(stops_list, dtype=np.float32),
        "text_embs": concatenated_text,
        "text_offsets": np.array(text_offsets, dtype=np.int32),
        "audio_embs": concatenated_audio,
        "audio_offsets": np.array(audio_offsets, dtype=np.int32),
    }


def extract_sequences_dataset(
    config_path: str | Path = "configs/daic_attention.yaml",
    session_subset: Sequence[int] | None = None,
    split_filter: str | None = None,
    device_override: str | None = None,
    downsample_factor: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run turn-level sequence feature extraction across DAIC-WOZ sessions."""
    cfg = load_config(config_path)

    manifest_path = Path(cfg["dataset"]["manifest_path"])
    turns_dir = Path(cfg["dataset"]["turns_dir"])
    cache_root = Path(cfg["features"]["cache_dir"])
    turns_cache_dir = cache_root / "turns"
    turns_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    if split_filter is not None:
        manifest = manifest[manifest["split"] == split_filter]
    if session_subset is not None:
        session_set = set(int(s) for s in session_subset)
        manifest = manifest[manifest["session_id"].astype(int).isin(session_set)]

    # Only process sessions that have transcripts and audio
    manifest = manifest[manifest["has_transcript"] == True]  # noqa: E712
    sessions = manifest["session_id"].astype(int).tolist()
    total_sessions = len(sessions)

    device_str = device_override or cfg["features"]["text"].get("device", "auto")
    device = resolve_device(device_str)
    logger.info("Extracting sequence features on device %s for %d sessions", device, total_sessions)

    # Load frozen encoders
    text_model_name = cfg["features"]["text"]["model_name"]
    audio_model_name = cfg["features"]["audio"]["model_name"]
    logger.info("Loading text encoder %s...", text_model_name)
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, local_files_only=True)
        text_model = AutoModel.from_pretrained(text_model_name, local_files_only=True)
    except Exception:
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_model = AutoModel.from_pretrained(text_model_name)
    text_model.to(device).eval()

    logger.info("Loading audio encoder %s...", audio_model_name)
    try:
        audio_fe = AutoFeatureExtractor.from_pretrained(audio_model_name, local_files_only=True)
        audio_model = AutoModel.from_pretrained(audio_model_name, local_files_only=True)
    except Exception:
        audio_fe = AutoFeatureExtractor.from_pretrained(audio_model_name)
        audio_model = AutoModel.from_pretrained(audio_model_name)
    audio_model.to(device).eval()

    target_sr = int(cfg["features"]["audio"].get("sample_rate", 16000))
    ds_factor = downsample_factor or int(cfg["features"]["audio"].get("downsample_factor", 2))

    start_time_all = time.time()
    processed_count = 0
    skipped_count = 0
    total_turns_extracted = 0

    for idx, session_id in enumerate(sessions, 1):
        out_path = turns_cache_dir / f"{session_id}.npz"
        if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
            try:
                # Quick validation that it can be loaded
                with np.load(out_path) as data:
                    _ = data["turn_ids"]
                logger.info(
                    "[%d/%d] Session %d already cached at %s (skipping)",
                    idx,
                    total_sessions,
                    session_id,
                    out_path.name,
                )
                skipped_count += 1
                continue
            except Exception:
                logger.warning("Corrupted cache file for %d; re-extracting", session_id)

        t0 = time.time()
        logger.info("[%d/%d] Extracting turn sequences for session %d...", idx, total_sessions, session_id)
        seq_dict = extract_turn_sequences_for_session(
            session_id=session_id,
            turns_dir=turns_dir,
            text_tokenizer=text_tokenizer,
            text_model=text_model,
            audio_feature_extractor=audio_fe,
            audio_model=audio_model,
            device=device,
            downsample_factor=ds_factor,
            target_sample_rate=target_sr,
        )

        # Atomic write via temporary file
        tmp_path = turns_cache_dir / f"{session_id}.tmp.npz"
        np.savez_compressed(tmp_path, **seq_dict)
        tmp_path.replace(out_path)

        dt = time.time() - t0
        n_turns = len(seq_dict["turn_ids"])
        total_turns_extracted += n_turns
        processed_count += 1
        file_size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Saved session %d (%d turns, text_tokens=%d, audio_frames=%d, size=%.2f MB) in %.2fs",
            session_id,
            n_turns,
            len(seq_dict["text_embs"]),
            len(seq_dict["audio_embs"]),
            file_size_mb,
            dt,
        )

    elapsed_all = time.time() - start_time_all
    logger.info(
        "Finished extraction: %d processed, %d skipped out of %d in %.1fs (total turns: %d)",
        processed_count,
        skipped_count,
        total_sessions,
        elapsed_all,
        total_turns_extracted,
    )

    manifest_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_model": text_model_name,
        "audio_model": audio_model_name,
        "downsample_factor": ds_factor,
        "total_sessions_requested": total_sessions,
        "processed_sessions": processed_count,
        "skipped_sessions": skipped_count,
        "total_turns_extracted": total_turns_extracted,
        "elapsed_seconds": elapsed_all,
    }
    with open(turns_cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)

    return manifest_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DAIC-WOZ turn-level sequence embeddings.")
    parser.add_argument("--config", default="configs/daic_attention.yaml")
    parser.add_argument("--split", default=None, choices=["train", "dev", "test"])
    parser.add_argument("--sessions", nargs="+", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run on 5 stratified sessions only")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    sessions = args.sessions
    if args.smoke_test:
        sessions = _get_smoke_test_sessions(cfg["dataset"]["manifest_path"])
        logger.info("Smoke test sessions selected: %s", sessions)

    extract_sequences_dataset(
        config_path=args.config,
        session_subset=sessions,
        split_filter=args.split,
        device_override=args.device,
        downsample_factor=args.downsample_factor,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
