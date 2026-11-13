"""Colab-compatible batch extraction script for DAIC-WOZ text and audio embeddings.

Supports resumable extraction across train/dev/test splits with optional GPU fp16
acceleration and automatic MentalBERT -> RoBERTa/BERT fallback.

Example Colab usage:
    !python scripts/colab_extract_daic.py --split dev --device cuda --fp16 --audio-mode window_concat
    !python scripts/colab_extract_daic.py --split train --device cuda --fp16 --audio-mode window_concat
    !python scripts/colab_extract_daic.py --split test --device cuda --fp16 --audio-mode window_concat
"""

from __future__ import annotations

import argparse
import logging

from src.features.extract_daic_audio import extract_daic_audio_embeddings
from src.features.extract_daic_text import _get_smoke_test_sessions, extract_daic_text_embeddings, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable DAIC-WOZ feature extraction.")
    parser.add_argument("--config", default="configs/daic_models.yaml")
    parser.add_argument("--split", default=None, choices=["train", "dev", "test"])
    parser.add_argument("--sessions", nargs="+", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audio-mode", default=None, choices=["turn_moments", "window_concat"])
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-ablation-text", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    sessions = args.sessions
    if args.smoke_test:
        sessions = _get_smoke_test_sessions(cfg["dataset"]["manifest_path"])

    if not args.skip_text:
        extract_daic_text_embeddings(
            config_path=args.config,
            variant="participant",
            split_filter=args.split,
            session_subset=sessions,
            device_override=args.device,
            batch_size_override=args.batch_size,
            fp16_override=args.fp16,
        )

    if not args.skip_text and not args.skip_ablation_text:
        extract_daic_text_embeddings(
            config_path=args.config,
            variant="with_ellie",
            split_filter=args.split,
            session_subset=sessions,
            device_override=args.device,
            batch_size_override=args.batch_size,
            fp16_override=args.fp16,
        )

    if not args.skip_audio:
        extract_daic_audio_embeddings(
            config_path=args.config,
            mode_override=args.audio_mode,
            split_filter=args.split,
            session_subset=sessions,
            device_override=args.device,
            fp16_override=args.fp16,
        )


if __name__ == "__main__":
    main()

