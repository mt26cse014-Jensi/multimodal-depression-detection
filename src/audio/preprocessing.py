"""Audio preprocessing functions for EATD-Corpus.

Handles:
- Loading audio from WAV format
- Multi-channel to mono conversion (averaging channels)
- Resampling to target sample rate (default 16 kHz)
- Leading and trailing silence trimming using frame-level RMS energy (VAD)
- Dropping clips shorter than minimum duration threshold (default 1.0s)
- Saving 16-bit PCM WAV output
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal

logger = logging.getLogger(__name__)


def load_audio(file_path: Path | str) -> Tuple[np.ndarray, int]:
    """Load an audio WAV file.

    Returns:
        audio: 1D or 2D numpy array normalized to float32 in range [-1.0, 1.0]
        sr: Sample rate in Hz
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    sr, data = wav.read(str(path))

    # Convert integer types to float32 in [-1.0, 1.0]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio dtype: {data.dtype}")

    return audio, sr


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging across channels.

    Args:
        audio: 1D (mono) or 2D array (samples, channels) or (channels, samples)

    Returns:
        1D mono float32 array
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # If shape is (channels, samples) with channels <= 2
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return np.mean(audio, axis=0).astype(np.float32)
        else:
            return np.mean(audio, axis=1).astype(np.float32)
    else:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {audio.shape}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to target sample rate if needed."""
    if orig_sr == target_sr:
        return audio

    num_samples = int(round(len(audio) * float(target_sr) / orig_sr))
    resampled = scipy.signal.resample(audio, num_samples)
    return resampled.astype(np.float32)


def trim_silence(
    signal: np.ndarray,
    sr: int = 16000,
    top_db: float = 30.0,
    frame_len_ms: float = 30.0,
    hop_len_ms: float = 10.0,
) -> np.ndarray:
    """Trim leading and trailing silence from audio using frame-level RMS energy.

    Args:
        signal: 1D float32 audio array
        sr: Sample rate in Hz
        top_db: The threshold (in decibels) below reference to consider as silence
        frame_len_ms: Frame length in milliseconds
        hop_len_ms: Hop length in milliseconds

    Returns:
        Trimmed 1D audio array
    """
    if len(signal) == 0:
        return signal

    frame_length = int(round(sr * frame_len_ms / 1000.0))
    hop_length = int(round(sr * hop_len_ms / 1000.0))

    if len(signal) <= frame_length:
        return signal

    # Number of frames
    num_frames = (len(signal) - frame_length) // hop_length + 1
    if num_frames <= 0:
        return signal

    # Construct frames using sliding window stride
    shape = (num_frames, frame_length)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

    # Frame RMS energy
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    max_rms = float(np.max(rms))

    # If maximum RMS is extremely low (ambient silence), consider the entire signal silent
    if max_rms < 1e-4:
        return np.array([], dtype=signal.dtype)

    # Energy in dB relative to maximum
    db = 20.0 * np.log10(np.maximum(rms, 1e-12) / max_rms)

    # Active frames exceeding threshold
    active_indices = np.where(db >= -top_db)[0]
    if len(active_indices) == 0:
        return np.array([], dtype=signal.dtype)

    start_sample = active_indices[0] * hop_length
    end_sample = min(len(signal), active_indices[-1] * hop_length + frame_length)

    return signal[start_sample:end_sample]


def save_wav_16bit(file_path: Path | str, signal: np.ndarray, sr: int = 16000) -> None:
    """Save 1D float32 audio signal as 16-bit PCM WAV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clip to prevent wrap-around distortion
    clipped = np.clip(signal, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    wav.write(str(path), sr, int16_data)


def preprocess_clip(
    input_path: Path | str,
    output_path: Optional[Path | str] = None,
    target_sr: int = 16000,
    top_db: float = 30.0,
    min_duration: float = 1.0,
) -> Dict[str, Any]:
    """Preprocess a single audio clip:

    1. Load WAV file
    2. Convert to mono
    3. Resample to target_sr if required
    4. Trim leading and trailing silence
    5. Drop if trimmed duration < min_duration
    6. Save to output_path if not dropped

    Returns:
        Dictionary with processing metadata:
        {
            "orig_duration": float,
            "duration": float,
            "dropped": bool,
            "reason": str | None,
            "output_path": str | None,
            "sample_rate": int,
            "channels": 1
        }
    """
    audio, orig_sr = load_audio(input_path)
    orig_duration = len(audio) / float(orig_sr)

    # Convert to mono
    mono = convert_to_mono(audio)

    # Resample
    resampled = resample_audio(mono, orig_sr, target_sr)

    # Trim leading and trailing silence
    trimmed = trim_silence(resampled, sr=target_sr, top_db=top_db)
    trimmed_duration = len(trimmed) / float(target_sr) if len(trimmed) > 0 else 0.0

    if trimmed_duration < min_duration:
        logger.info(
            "Dropping clip %s: trimmed duration %.3fs < threshold %.1fs (orig: %.3fs)",
            input_path,
            trimmed_duration,
            min_duration,
            orig_duration,
        )
        return {
            "orig_duration": round(orig_duration, 4),
            "duration": round(trimmed_duration, 4),
            "dropped": True,
            "reason": f"Duration {trimmed_duration:.3f}s < {min_duration:.1f}s threshold",
            "output_path": None,
            "sample_rate": target_sr,
            "channels": 1,
        }

    out_str = None
    if output_path is not None:
        save_wav_16bit(output_path, trimmed, sr=target_sr)
        out_str = str(output_path).replace("\\", "/")

    return {
        "orig_duration": round(orig_duration, 4),
        "duration": round(trimmed_duration, 4),
        "dropped": False,
        "reason": None,
        "output_path": out_str,
        "sample_rate": target_sr,
        "channels": 1,
    }

