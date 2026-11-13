"""Extract and cache text embeddings for DAIC-WOZ windows (Phase 5).

Supports two text variants:
  - ``participant``: participant-only window text from ``windows.parquet`` (headline).
  - ``with_ellie``: participant + Ellie turns aligned to each window's
    ``[start_time, end_time]`` interval (for the interviewer ablation only).

Features:
  - Attempts ``mental/mental-bert-base-uncased`` first (using ``HF_TOKEN`` if set)
    and automatically falls back to ``roberta-base`` (or ``bert-base-uncased``) if
    gated or inaccessible, logging the resolved model and storing metadata.
  - Handles long windows (>512 tokens) via non-overlapping 510-token content chunks
    with token-count-weighted mean pooling across chunks.
  - Deduplicates identical ``(session_id, start_time, end_time)`` windows per session
    before encoding, saving one resumable ``.npz`` file per session.
  - Colab-compatible: ``--device auto``, ``--fp16``, ``--batch-size``, ``--split``.

Privacy: never logs or prints raw transcript text.
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
import torch
import yaml
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.data.parse_daic_transcripts import parse_session

logger = logging.getLogger(__name__)


# ── Configuration & device helpers ──────────────────────────────────────────────

def load_config(config_path: str = "configs/daic_models.yaml") -> dict[str, Any]:
    """Load and return YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve ``'auto'`` / ``'cpu'`` / ``'cuda'`` to a ``torch.device``."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def model_short_name(model_name: str) -> str:
    """Extract filesystem-safe model name from a HuggingFace repo ID."""
    return model_name.split("/")[-1]


def make_window_key(start_time: float, end_time: float) -> str:
    """Canonical string key for deduplicating windows inside a session."""
    return f"{float(start_time):.4f}|{float(end_time):.4f}"


# ── Model resolution (MentalBERT + fallback) ───────────────────────────────────

def resolve_text_model_name(
    preferred_model: str = "mental/mental-bert-base-uncased",
    fallback_model: str = "roberta-base",
    alt_fallback_model: str = "bert-base-uncased",
    token: Optional[str] = None,
) -> tuple[str, bool, str]:
    """Resolve an accessible HuggingFace text model, falling back if gated/offline.

    Args:
        preferred_model: Primary model ID (default: MentalBERT).
        fallback_model: First fallback model ID if primary fails.
        alt_fallback_model: Second fallback model ID if first fallback fails.
        token: Optional HF token; defaults to ``HF_TOKEN`` env var if set.

    Returns:
        (resolved_model_name, fell_back, reason_note)
    """
    hf_token = token if token is not None else os.environ.get("HF_TOKEN") or None
    candidates = [preferred_model]
    for fb in (fallback_model, alt_fallback_model):
        if fb and fb not in candidates:
            candidates.append(fb)

    # First check local cache for any candidate if offline, while preserving priority order
    errors: list[str] = []
    for idx, candidate in enumerate(candidates):
        try:
            AutoConfig.from_pretrained(candidate, token=hf_token)
            AutoTokenizer.from_pretrained(candidate, token=hf_token)
            if idx == 0:
                return candidate, False, "primary model accessible"
            reason = f"fallback from {preferred_model} ({errors[0]})"
            logger.warning("Using fallback text encoder %r: %s", candidate, reason)
            return candidate, True, reason
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:180]}"
            errors.append(err_msg)
            logger.info("Candidate text model %r unavailable (%s)", candidate, err_msg)

        # Retry cached local files before giving up on this candidate
        try:
            AutoConfig.from_pretrained(candidate, local_files_only=True)
            AutoTokenizer.from_pretrained(candidate, local_files_only=True)
            if idx == 0:
                return candidate, False, "primary model loaded from local cache"
            reason = f"fallback from {preferred_model} using local cache ({errors[0]})"
            logger.warning("Using cached fallback text encoder %r: %s", candidate, reason)
            return candidate, True, reason
        except Exception:
            pass

    raise RuntimeError(
        f"None of the candidate text models could be loaded ({candidates}). "
        f"Errors: {errors}"
    )


# ── Chunked tokenization & pooling ─────────────────────────────────────────────

def chunk_token_ids(
    token_ids: Sequence[int],
    max_content_tokens: int = 510,
) -> list[list[int]]:
    """Split raw token IDs (without special tokens) into fixed-max chunks.

    Args:
        token_ids: Full sequence of content token IDs.
        max_content_tokens: Maximum non-special tokens per chunk (e.g. 512 - 2 = 510).

    Returns:
        List of non-empty token ID slices (or ``[[]]`` if ``token_ids`` is empty).
    """
    if max_content_tokens < 1:
        raise ValueError(f"max_content_tokens must be >= 1, got {max_content_tokens}")
    ids = list(token_ids)
    if not ids:
        return [[]]
    return [ids[i : i + max_content_tokens] for i in range(0, len(ids), max_content_tokens)]


def encode_texts_chunked(
    texts: Sequence[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16,
    use_fp16: bool = False,
) -> np.ndarray:
    """Encode arbitrarily long texts using chunking + token-weighted mean pooling.

    For texts whose tokenized length + 2 special tokens <= ``max_length``, a single
    forward pass matches standard masked mean pooling over ``last_hidden_state``.
    For longer texts, content tokens are partitioned into chunks of
    ``max_length - 2`` and chunk representations are averaged weighted by valid
    token counts per chunk.

    Returns:
        Float32 array of shape ``(len(texts), hidden_size)``.
    """
    hidden_size = int(model.config.hidden_size)
    n_texts = len(texts)
    if n_texts == 0:
        return np.zeros((0, hidden_size), dtype=np.float32)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    n_special = int(cls_id is not None) + int(sep_id is not None)
    max_content = max(1, int(max_length) - n_special)

    # Expand every non-empty text into one or more bounded chunks
    chunk_seqs: list[list[int]] = []
    chunk_parents: list[int] = []
    empty_mask = np.zeros(n_texts, dtype=bool)

    for idx, raw in enumerate(texts):
        cleaned = "" if raw is None or pd.isna(raw) else str(raw).strip()
        if not cleaned:
            empty_mask[idx] = True
            continue
        raw_ids = tokenizer.encode(cleaned, add_special_tokens=False)
        if not raw_ids:
            empty_mask[idx] = True
            continue
        for slice_ids in chunk_token_ids(raw_ids, max_content_tokens=max_content):
            seq = []
            if cls_id is not None:
                seq.append(int(cls_id))
            seq.extend(int(t) for t in slice_ids)
            if sep_id is not None:
                seq.append(int(sep_id))
            chunk_seqs.append(seq)
            chunk_parents.append(idx)

    sum_vecs = np.zeros((n_texts, hidden_size), dtype=np.float64)
    token_totals = np.zeros(n_texts, dtype=np.float64)

    if not chunk_seqs:
        return sum_vecs.astype(np.float32)

    # Sort chunks by length before batching to minimize padding overhead on CPU/GPU
    order = sorted(range(len(chunk_seqs)), key=lambda i: len(chunk_seqs[i]))
    amp_enabled = bool(use_fp16 and device.type == "cuda")

    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        batch_seqs = [chunk_seqs[i] for i in batch_indices]
        max_len = max(len(s) for s in batch_seqs)

        input_ids = torch.full((len(batch_seqs), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(batch_seqs), max_len), dtype=torch.long)
        for b_i, s in enumerate(batch_seqs):
            input_ids[b_i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn_mask[b_i, : len(s)] = 1

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)

            hidden = outputs.last_hidden_state.float()  # (B, L, D)
            mask_f = attn_mask.unsqueeze(-1).float()    # (B, L, 1)
            chunk_sums = (hidden * mask_f).sum(dim=1).cpu().numpy()   # (B, D)
            chunk_counts = mask_f.sum(dim=1).squeeze(-1).cpu().numpy()  # (B,)

        for b_i, orig_chunk_idx in enumerate(batch_indices):
            parent = chunk_parents[orig_chunk_idx]
            sum_vecs[parent] += chunk_sums[b_i]
            token_totals[parent] += chunk_counts[b_i]

    out = np.zeros((n_texts, hidden_size), dtype=np.float32)
    valid = token_totals > 0
    out[valid] = (sum_vecs[valid] / token_totals[valid, None]).astype(np.float32)
    return out


# ── Ablation window text alignment (participant + Ellie) ───────────────────────

def build_with_ellie_window_texts(
    session_id: int,
    windows_sub: pd.DataFrame,
    preprocess_cfg: dict[str, Any],
    include_speaker_tags: bool = False,
) -> list[str]:
    """Construct participant + Ellie window texts aligned to ``[start_time, end_time]``.

    Uses the exact Phase 4 cleaning rules from ``parse_daic_transcripts.parse_session``.
    Sessions with zero Ellie lines (451, 458, 480) cleanly return participant text.
    """
    raw_iw = preprocess_cfg.get("interruption_windows", {})
    iw_map = {int(k): v for k, v in raw_iw.items()}
    zips_dir = Path(preprocess_cfg["zips_dir"]) if preprocess_cfg.get("zips_dir") else None

    turn_df = parse_session(
        session_id=int(session_id),
        daic_dir=Path(preprocess_cfg["daic_dir"]),
        transcript_format_threshold=int(preprocess_cfg.get("transcript_format_threshold", 363)),
        small_talk_cutoff_sec=float(preprocess_cfg.get("small_talk_cutoff_sec", 60.0)),
        interruption_windows=iw_map.get(int(session_id)),
        zips_dir=zips_dir,
    )

    if turn_df.empty:
        return [str(t) for t in windows_sub["text"].tolist()]

    kept = turn_df[(~turn_df["dropped"]) & (turn_df["text"].astype(str).str.strip() != "")].copy()
    if kept.empty:
        return [str(t) for t in windows_sub["text"].tolist()]

    kept = kept.sort_values(["start_time", "turn_id"]).reset_index(drop=True)
    t_starts = kept["start_time"].to_numpy(dtype=float)
    t_stops = kept["stop_time"].to_numpy(dtype=float)
    speakers = kept["speaker"].astype(str).tolist()
    utterances = kept["text"].astype(str).tolist()

    out_texts: list[str] = []
    eps = 1e-4
    for row in windows_sub.itertuples():
        w_start = float(row.start_time) - eps
        w_end = float(row.end_time) + eps
        overlap_idx = np.where((t_stops >= w_start) & (t_starts <= w_end))[0]
        if len(overlap_idx) == 0:
            out_texts.append(str(row.text))
            continue
        if include_speaker_tags:
            parts = [f"[{speakers[i]}] {utterances[i]}" for i in overlap_idx]
        else:
            parts = [utterances[i] for i in overlap_idx]
        joined = " ".join(p for p in parts if p.strip())
        out_texts.append(joined if joined else str(row.text))

    return out_texts


def deduplicate_session_windows(
    windows_sub: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Deduplicate windows inside a session by ``(start_time, end_time)``."""
    keys = [make_window_key(s, e) for s, e in zip(windows_sub["start_time"], windows_sub["end_time"])]
    key_to_pos: dict[str, int] = {}
    unique_rows: list[int] = []
    inverse = np.empty(len(windows_sub), dtype=np.int64)

    for i, k in enumerate(keys):
        if k not in key_to_pos:
            key_to_pos[k] = len(unique_rows)
            unique_rows.append(i)
        inverse[i] = key_to_pos[k]

    unique_df = windows_sub.iloc[unique_rows].copy().reset_index(drop=True)
    unique_df["window_key"] = [keys[i] for i in unique_rows]
    return unique_df, inverse


# ── Extraction & loading ───────────────────────────────────────────────────────

def _get_smoke_test_sessions(manifest_path: str | Path) -> list[int]:
    """Return 5 stratified smoke-test session IDs (train dep/ctrl, dev dep/ctrl, test)."""
    # Verified fast sessions covering all splits and classes
    preferred = [347, 357, 388, 489, 354]
    manifest = pd.read_csv(manifest_path)
    present = set(manifest["session_id"].astype(int).tolist())
    if all(s in present for s in preferred):
        return preferred

    chosen: list[int] = []
    for split in ("train", "dev"):
        sub = manifest[manifest["split"] == split]
        for lbl in (1, 0):
            cand = sub[sub["phq8_binary"] == lbl]["session_id"].astype(int).tolist()
            if cand:
                chosen.append(cand[0])
    test_cand = manifest[manifest["split"] == "test"]["session_id"].astype(int).tolist()
    if test_cand:
        chosen.append(test_cand[0])
    return chosen


def _variant_cache_dir(
    cache_root: Path,
    model_name: str,
    variant: str,
) -> Path:
    short = model_short_name(model_name)
    suffix = "" if variant == "participant" else "_with_ellie"
    return cache_root / f"text_{short}{suffix}"


def _resolve_cached_or_config_model(
    cfg: dict[str, Any],
    variant: str,
) -> tuple[str, bool, str]:
    """Check metadata.json first for consistent cached model, otherwise resolve HF."""
    feat_cfg = cfg["features"]
    text_cfg = feat_cfg["text"]
    cache_root = Path(feat_cfg["cache_dir"])

    # Prefer existing metadata for this variant or participant variant so both match identically
    check_variants = [variant] if variant == "participant" else [variant, "participant"]
    for v in check_variants:
        for candidate in (
            text_cfg["model_name"],
            text_cfg.get("fallback_model_name", "roberta-base"),
            text_cfg.get("alt_fallback_model_name", "bert-base-uncased"),
        ):
            meta_path = _variant_cache_dir(cache_root, candidate, v) / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                resolved = meta.get("resolved_model", candidate)
                return resolved, bool(meta.get("fell_back", False)), meta.get("resolution_note", "from metadata")

    return resolve_text_model_name(
        preferred_model=text_cfg["model_name"],
        fallback_model=text_cfg.get("fallback_model_name", "roberta-base"),
        alt_fallback_model=text_cfg.get("alt_fallback_model_name", "bert-base-uncased"),
    )


def extract_daic_text_embeddings(
    config_path: str = "configs/daic_models.yaml",
    variant: str = "participant",
    split_filter: Optional[str] = None,
    session_subset: Optional[Sequence[int]] = None,
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    fp16_override: Optional[bool] = None,
) -> Path:
    """Extract and cache per-session window text embeddings for DAIC-WOZ.

    Returns the cache directory path containing ``{session_id}.npz`` and ``metadata.json``.
    """
    if variant not in ("participant", "with_ellie"):
        raise ValueError(f"Unsupported text variant {variant!r}; expected 'participant' or 'with_ellie'")

    cfg = load_config(config_path)
    feat_cfg = cfg["features"]
    text_cfg = feat_cfg["text"]
    cache_root = Path(feat_cfg["cache_dir"])

    resolved_model, fell_back, reason_note = _resolve_cached_or_config_model(cfg, variant=variant)

    cache_dir = _variant_cache_dir(cache_root, resolved_model, variant)
    cache_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_parquet(cfg["dataset"]["windows_path"])
    if split_filter is not None:
        windows = windows[windows["split"] == split_filter].copy()
    if session_subset is not None:
        target_set = {int(s) for s in session_subset}
        windows = windows[windows["session_id"].astype(int).isin(target_set)].copy()

    if windows.empty:
        logger.warning("No matching windows found for split=%s, sessions=%s", split_filter, session_subset)
        return cache_dir

    session_ids = sorted(windows["session_id"].astype(int).unique().tolist())
    pending = [sid for sid in session_ids if not (cache_dir / f"{sid}.npz").exists()]

    preprocess_cfg = None
    if variant == "with_ellie":
        with open(cfg["dataset"]["preprocess_config_path"], "r", encoding="utf-8") as f:
            preprocess_cfg = yaml.safe_load(f)

    if not pending:
        logger.info("All %d DAIC %s text sessions already cached in %s.", len(session_ids), variant, cache_dir)
        return cache_dir

    device = resolve_device(device_override or text_cfg.get("device", "auto"))
    batch_size = int(batch_size_override or text_cfg.get("batch_size", 16))
    max_length = int(text_cfg.get("max_length", 512))
    use_fp16 = bool(text_cfg.get("fp16", False) if fp16_override is None else fp16_override)
    include_tags = bool(text_cfg.get("ablation_include_speaker_tags", False))

    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 4))

    logger.info(
        "Extracting DAIC text embeddings (%s): %d/%d sessions | model=%s | device=%s | bs=%d",
        variant, len(pending), len(session_ids), resolved_model, device, batch_size,
    )

    hf_token = os.environ.get("HF_TOKEN") or None
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, local_files_only=True)
        model = AutoModel.from_pretrained(resolved_model, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, token=hf_token)
        model = AutoModel.from_pretrained(resolved_model, token=hf_token)

    model.eval()
    model.to(device)
    hidden_size = int(model.config.hidden_size)

    meta_path = cache_dir / "metadata.json"
    meta_payload = {
        "preferred_model": text_cfg["model_name"],
        "resolved_model": resolved_model,
        "fell_back": fell_back,
        "resolution_note": reason_note,
        "variant": variant,
        "hidden_size": hidden_size,
        "max_length": max_length,
        "ablation_include_speaker_tags": include_tags,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    t0 = time.time()
    for i, sid in enumerate(pending):
        sub = windows[windows["session_id"].astype(int) == sid].copy()
        unique_df, _ = deduplicate_session_windows(sub)

        if variant == "participant":
            texts = [str(t) for t in unique_df["text"].tolist()]
        else:
            texts = build_with_ellie_window_texts(
                session_id=sid,
                windows_sub=unique_df,
                preprocess_cfg=preprocess_cfg,
                include_speaker_tags=include_tags,
            )

        embs = encode_texts_chunked(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
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

        if (i + 1) % 10 == 0 or (i + 1) == len(pending):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(pending) - i - 1) / rate if rate > 0 else 0.0
            logger.info(
                "  [%d/%d] session %d (%d unique windows) | %.1fs elapsed, ~%.0fs ETA",
                i + 1, len(pending), sid, len(unique_df), elapsed, eta,
            )

    return cache_dir


def load_daic_text_embeddings(
    config_path: str = "configs/daic_models.yaml",
    split: Optional[str] = None,
    variant: str = "participant",
    scales: Optional[Sequence[int]] = None,
    session_subset: Optional[Sequence[int]] = None,
    model_name_override: Optional[str] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load cached DAIC text embeddings aligned row-for-row with ``windows.parquet``.

    Returns:
        X: Float32 matrix of shape ``(N_windows, hidden_size)``.
        windows_df: Filtered ``windows.parquet`` DataFrame matching ``X`` rows.
    """
    cfg = load_config(config_path)
    cache_root = Path(cfg["features"]["cache_dir"])

    if model_name_override:
        model_name = model_name_override
    else:
        model_name, _, _ = _resolve_cached_or_config_model(cfg, variant=variant)

    cache_dir = _variant_cache_dir(cache_root, model_name, variant)

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
            f"Missing cached DAIC text ({variant}) embeddings in {cache_dir} for "
            f"{len(missing)} sessions: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    first_emb = next(iter(session_cache.values()))[1]
    hidden_dim = int(first_emb.shape[1])
    X = np.empty((len(windows), hidden_dim), dtype=np.float32)

    for row_idx, row in enumerate(windows.itertuples()):
        sid = int(row.session_id)
        key = make_window_key(row.start_time, row.end_time)
        key_map, embs = session_cache[sid]
        if key not in key_map:
            raise KeyError(f"Window key {key} missing from {cache_dir / f'{sid}.npz'}")
        X[row_idx] = embs[key_map[key]]

    logger.info(
        "Loaded DAIC text embeddings (%s, split=%s): X=%s",
        variant, split or "all", X.shape,
    )
    return X, windows


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DAIC-WOZ window text embeddings.")
    parser.add_argument("--config", default="configs/daic_models.yaml")
    parser.add_argument("--variant", default="participant", choices=["participant", "with_ellie"])
    parser.add_argument("--split", default=None, choices=["train", "dev", "test"])
    parser.add_argument("--sessions", nargs="+", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run on 5 stratified sessions only")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=None)
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

    extract_daic_text_embeddings(
        config_path=args.config,
        variant=args.variant,
        split_filter=args.split,
        session_subset=sessions,
        device_override=args.device,
        batch_size_override=args.batch_size,
        fp16_override=args.fp16 if args.fp16 else None,
    )


if __name__ == "__main__":
    main()

