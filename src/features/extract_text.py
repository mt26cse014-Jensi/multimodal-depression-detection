"""Extract text embeddings from frozen Chinese BERT for EATD-Corpus.

Loads a pretrained Chinese language model (configurable via eatd_preprocess.yaml),
runs each participant's three cleaned answers (positive, neutral, negative)
through the frozen encoder, mean-pools the last hidden state over non-padding
tokens, concatenates the three 768-dim vectors → one 2304-dim vector per
participant, and caches as .npy files.

Usage:
    python -m src.features.extract_text                          # all
    python -m src.features.extract_text --subset t_1 t_10 v_52   # specific
    python -m src.features.extract_text --smoke-test              # 10 participants
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
import yaml
from transformers import AutoModel, AutoTokenizer

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


# ── Core extraction ─────────────────────────────────────────────────

def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """Mean-pool last hidden state over non-padding tokens.

    Args:
        last_hidden: (1, seq_len, hidden_dim)
        attention_mask: (1, seq_len)

    Returns:
        1-D numpy array of shape (hidden_dim,)
    """
    mask = attention_mask.unsqueeze(-1).float()          # (1, seq_len, 1)
    summed = (last_hidden * mask).sum(dim=1)             # (1, hidden_dim)
    counts = mask.sum(dim=1).clamp(min=1)                # (1, 1)
    return (summed / counts).squeeze(0).cpu().numpy()    # (hidden_dim,)


def extract_text_embeddings(
    config_path: str = "configs/eatd_preprocess.yaml",
    participant_subset: Optional[list[str]] = None,
) -> None:
    """Extract and cache text embeddings for EATD participants.

    Args:
        config_path: Path to the YAML config file.
        participant_subset: If given, only process these participant IDs.
    """
    cfg = load_config(config_path)
    text_cfg = cfg["features"]["text"]
    model_name = text_cfg["model_name"]
    max_length = text_cfg["max_length"]
    device = _resolve_device(text_cfg["device"])

    cache_dir = Path(cfg["features"]["cache_dir"]) / "text"
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
        "Extracting text embeddings for %d / %d participants (model: %s, device: %s)",
        len(to_extract), len(manifest), model_name, device,
    )

    # Set torch threads for CPU efficiency
    torch.set_num_threads(4)

    # Load model and tokeniser (frozen)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    valences = ["positive", "neutral", "negative"]
    start_time = time.time()

    for i, row in enumerate(to_extract):
        pid = row.participant_id
        embeddings = []

        for valence in valences:
            text = row[f"text_{valence}_clean"]
            if pd.isna(text) or str(text).strip() == "":
                # Fallback: zero vector (should be rare)
                logger.warning("Empty text for %s/%s — using zero vector", pid, valence)
                embeddings.append(np.zeros(model.config.hidden_size, dtype=np.float32))
                continue

            encoded = tokenizer(
                str(text),
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)

            vec = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(vec.astype(np.float32))

        # Concatenate 3 × hidden_dim → e.g. 3 × 768 = 2304
        combined = np.concatenate(embeddings)
        out_path = cache_dir / f"{pid}.npy"
        np.save(out_path, combined)

        if (i + 1) % 20 == 0 or (i + 1) == len(to_extract):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(to_extract) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] %.1f sec elapsed, ~%.0f sec remaining",
                i + 1, len(to_extract), elapsed, eta,
            )

    total = time.time() - start_time
    logger.info("Text extraction complete: %d participants in %.1f sec", len(to_extract), total)


# ── Loading cached embeddings ────────────────────────────────────────

def load_text_embeddings(
    config_path: str = "configs/eatd_preprocess.yaml",
    aggregation: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all cached text embeddings.

    Args:
        config_path: Path to the YAML config file.
        aggregation: How to combine the 3 answer embeddings per participant
            (``"concat"`` or ``"mean"``). Defaults to ``features.aggregation``
            in config, or ``"concat"`` if omitted.

    Returns:
        X: (N, 2304) for ``"concat"`` or (N, 768) for ``"mean"`` float32 array.
        y: (N,) int array of binary labels.
        pids: list of N participant IDs (same order).

    Raises:
        FileNotFoundError: if any participant's .npy is missing.
    """
    cfg = load_config(config_path)
    if aggregation is None:
        aggregation = cfg.get("features", {}).get("aggregation", "concat")

    cache_dir = Path(cfg["features"]["cache_dir"]) / "text"
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
        valid_mask = [
            not (pd.isna(row[f"text_{v}_clean"]) or str(row[f"text_{v}_clean"]).strip() == "")
            for v in valences
        ]
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
            f"Missing cached text embeddings for {len(missing)} participants: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    logger.info(
        "Loaded text embeddings (aggregation=%s): X=%s, y=%s, %d participants",
        aggregation, X.shape, y.shape, len(pid_list),
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
    parser = argparse.ArgumentParser(description="Extract EATD text embeddings.")
    parser.add_argument("--config", default="configs/eatd_preprocess.yaml")
    parser.add_argument("--subset", nargs="+", help="Specific participant IDs")
    parser.add_argument("--smoke-test", action="store_true", help="Run on 10 participants only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    subset = args.subset
    if args.smoke_test:
        subset = _get_smoke_test_pids(args.config)
        logger.info("Smoke test mode: %s", subset)

    extract_text_embeddings(config_path=args.config, participant_subset=subset)


if __name__ == "__main__":
    main()

