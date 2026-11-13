"""Extract audio embeddings from frozen speech encoders for EATD-Corpus.

Loads a pretrained speech model (configurable via eatd_preprocess.yaml),
processes each participant's three audio clips (positive, neutral, negative)
through the frozen encoder, applies mean + std pooling over the time axis,
concatenates the three vectors → one embedding per participant, and caches
as .npy files.

Supports two encoders:
  - facebook/wav2vec2-xls-r-300m  (multilingual, 1024-dim → 6144 per participant)
  - microsoft/wavlm-base-plus     (768-dim → 4608 per participant)

Features:
  - Runs on CPU or GPU (auto-detect or config)
  - Processes one clip at a time (batch_size=1) for low-memory systems
  - Resumable: skips participants with existing .npy files
  - 60-second safety cap on audio length (configurable)
  - Handles dropped clips (from Phase 1) with zero vectors

Usage:
    python -m src.features.extract_audio                                 # primary encoder
    python -m src.features.extract_audio --encoder-key alt_model_name    # comparison encoder
    python -m src.features.extract_audio --smoke-test                    # 10 participants
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
from transformers import AutoModel, AutoFeatureExtractor

from src.features.aggregation import aggregate_answer_embeddings

logger = logging.getLogger(__name__)


# ── Configuration helpers ────────────────────────────────────────────

def load_config(config_path: str = "configs/eatd_preprocess.yaml") -> dict:
    """Load and return the YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' / 'cpu' / 'cuda' to a torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _model_short_name(model_name: str) -> str:
    """Extract a filesystem-safe short name from a HuggingFace model ID.

    Examples:
        'facebook/wav2vec2-xls-r-300m' → 'wav2vec2-xls-r-300m'
        'microsoft/wavlm-base-plus'    → 'wavlm-base-plus'
    """
    return model_name.split("/")[-1]


# ── Core extraction ─────────────────────────────────────────────────

def _mean_std_pool(hidden_states: torch.Tensor) -> np.ndarray:
    """Mean + std pooling over time dimension.

    Args:
        hidden_states: (1, T, D) — encoder output for one audio clip.

    Returns:
        1-D numpy array of shape (2*D,) — concatenation of mean and std.
    """
    h = hidden_states.squeeze(0)  # (T, D)
    mean = h.mean(dim=0)          # (D,)
    std = h.std(dim=0)            # (D,)
    return torch.cat([mean, std]).cpu().numpy().astype(np.float32)


def extract_audio_embeddings(
    config_path: str = "configs/eatd_preprocess.yaml",
    participant_subset: Optional[list[str]] = None,
    encoder_key: str = "model_name",
) -> None:
    """Extract and cache audio embeddings for EATD participants.

    Args:
        config_path: Path to the YAML config file.
        participant_subset: If given, only process these participant IDs.
        encoder_key: Which model to use from the config's audio section.
                     'model_name' for primary, 'alt_model_name' for comparison.
    """
    cfg = load_config(config_path)
    audio_cfg = cfg["features"]["audio"]
    model_name = audio_cfg[encoder_key]
    target_sr = audio_cfg["sample_rate"]
    max_dur = audio_cfg.get("max_duration_sec", 60.0)
    device = _resolve_device(audio_cfg["device"])

    short_name = _model_short_name(model_name)
    cache_dir = Path(cfg["features"]["cache_dir"]) / f"audio_{short_name}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(cfg["dataset"]["manifest_path"])

    # Filter to requested subset
    if participant_subset is not None:
        manifest = manifest[manifest.participant_id.isin(participant_subset)]
        if manifest.empty:
            logger.warning("No matching participants in subset: %s", participant_subset)
            return

    # Identify which participants still need extraction
    to_extract = []
    for _, row in manifest.iterrows():
        out_path = cache_dir / f"{row.participant_id}.npy"
        if out_path.exists():
            logger.debug("Skipping %s (cached)", row.participant_id)
        else:
            to_extract.append(row)

    if not to_extract:
        logger.info("All %d participants already cached — nothing to do.", len(manifest))
        return

    logger.info(
        "Extracting audio embeddings for %d / %d participants "
        "(model: %s, device: %s, max_dur: %.0fs)",
        len(to_extract), len(manifest), model_name, device, max_dur,
    )

    # Set torch threads for CPU efficiency
    torch.set_num_threads(4)

    # Load model and feature extractor (frozen)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # Determine hidden_dim from model config
    hidden_dim = model.config.hidden_size
    pooled_dim = 2 * hidden_dim  # mean + std
    logger.info("Model hidden_dim=%d, pooled_dim=%d per clip, total=%d per participant",
                hidden_dim, pooled_dim, pooled_dim * 3)

    valences = ["positive", "neutral", "negative"]
    max_samples = int(max_dur * target_sr)
    start_time = time.time()
    truncated_count = 0

    for i, row in enumerate(to_extract):
        pid = row.participant_id
        embeddings = []

        for valence in valences:
            # Check if this clip was dropped in Phase 1
            dropped_col = f"audio_{valence}_dropped"
            if row[dropped_col]:
                logger.debug("Dropped clip %s/%s — using zero vector", pid, valence)
                embeddings.append(np.zeros(pooled_dim, dtype=np.float32))
                continue

            audio_path = row[f"audio_{valence}_path"]
            if pd.isna(audio_path) or not Path(audio_path).exists():
                logger.warning("Missing audio file %s/%s — using zero vector", pid, valence)
                embeddings.append(np.zeros(pooled_dim, dtype=np.float32))
                continue

            # Load audio with torchaudio
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed (Phase 1 should already be 16 kHz)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)

            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Apply safety cap on duration
            if waveform.shape[1] > max_samples:
                logger.info(
                    "Truncating %s/%s from %.1fs to %.1fs",
                    pid, valence, waveform.shape[1] / target_sr, max_dur,
                )
                waveform = waveform[:, :max_samples]
                truncated_count += 1

            # Prepare input for the model
            audio_np = waveform.squeeze(0).numpy()
            inputs = feature_extractor(
                audio_np,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                outputs = model(input_values)

            vec = _mean_std_pool(outputs.last_hidden_state)
            embeddings.append(vec)

        # Concatenate 3 × pooled_dim
        combined = np.concatenate(embeddings)
        out_path = cache_dir / f"{pid}.npy"
        np.save(out_path, combined)

        if (i + 1) % 10 == 0 or (i + 1) == len(to_extract):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(to_extract) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] %.1f sec elapsed, ~%.0f sec remaining",
                i + 1, len(to_extract), elapsed, eta,
            )

    total = time.time() - start_time
    logger.info(
        "Audio extraction complete: %d participants in %.1f sec (%d clips truncated at %.0fs)",
        len(to_extract), total, truncated_count, max_dur,
    )


# ── Loading cached embeddings ────────────────────────────────────────

def load_audio_embeddings(
    config_path: str = "configs/eatd_preprocess.yaml",
    encoder_key: str = "model_name",
    aggregation: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all cached audio embeddings for a specific encoder.

    Args:
        config_path: Path to the YAML config file.
        encoder_key: Which model's cache to load ('model_name' or 'alt_model_name').
        aggregation: How to combine the 3 answer embeddings per participant
            (``"concat"`` or ``"mean"``). Defaults to ``features.aggregation``
            in config, or ``"concat"`` if omitted. When ``"mean"``, dropped
            clips (``audio_{valence}_dropped == True``) are excluded from the average.

    Returns:
        X: (N, D) float32 array of embeddings.
        y: (N,) int array of binary labels.
        pids: list of N participant IDs (same order).

    Raises:
        FileNotFoundError: if any participant's .npy is missing.
    """
    cfg = load_config(config_path)
    if aggregation is None:
        aggregation = cfg.get("features", {}).get("aggregation", "concat")

    model_name = cfg["features"]["audio"][encoder_key]
    short_name = _model_short_name(model_name)
    cache_dir = Path(cfg["features"]["cache_dir"]) / f"audio_{short_name}"
    manifest = pd.read_csv(cfg["dataset"]["manifest_path"])

    valences = ["positive", "neutral", "negative"]
    X_list, y_list, pid_list = [], [], []
    missing = []

    for _, row in manifest.iterrows():
        npy_path = cache_dir / f"{row.participant_id}.npy"
        if not npy_path.exists():
            missing.append(row.participant_id)
            continue
        raw_vec = np.load(npy_path)
        valid_mask = [not bool(row[f"audio_{v}_dropped"]) for v in valences]
        vec = aggregate_answer_embeddings(
            raw_vec,
            n_answers=len(valences),
            mode=aggregation,
            valid_mask=valid_mask,
        )
        X_list.append(vec)
        y_list.append(int(row.label))
        pid_list.append(row.participant_id)

    if missing:
        raise FileNotFoundError(
            f"Missing cached audio embeddings ({short_name}) for {len(missing)} participants: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    logger.info(
        "Loaded audio embeddings (%s, aggregation=%s): X=%s, y=%s, %d participants",
        short_name, aggregation, X.shape, y.shape, len(pid_list),
    )
    return X, y, pid_list


# ── CLI entry point ──────────────────────────────────────────────────

def _get_smoke_test_pids(config_path: str) -> list[str]:
    """Return 10 smoke-test participant IDs: 5 depressed + 5 control from fold 0 train."""
    cfg = load_config(config_path)
    manifest = pd.read_csv(cfg["dataset"]["manifest_path"])
    with open(cfg["dataset"]["folds_path"], "r") as f:
        folds = json.load(f)

    fold0_train = set(folds["folds"][0]["train"])
    df_train = manifest[manifest.participant_id.isin(fold0_train)]
    dep = df_train[df_train.label == 1].participant_id.tolist()[:5]
    ctrl = df_train[df_train.label == 0].participant_id.tolist()[:5]
    return dep + ctrl


def main():
    parser = argparse.ArgumentParser(description="Extract EATD audio embeddings.")
    parser.add_argument("--config", default="configs/eatd_preprocess.yaml")
    parser.add_argument("--subset", nargs="+", help="Specific participant IDs")
    parser.add_argument("--smoke-test", action="store_true", help="Run on 10 participants only")
    parser.add_argument(
        "--encoder-key", default="model_name",
        choices=["model_name", "alt_model_name"],
        help="Which encoder to use from config (default: model_name = XLS-R)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    subset = args.subset
    if args.smoke_test:
        subset = _get_smoke_test_pids(args.config)
        logger.info("Smoke test mode: %s", subset)

    extract_audio_embeddings(
        config_path=args.config,
        participant_subset=subset,
        encoder_key=args.encoder_key,
    )


if __name__ == "__main__":
    main()

