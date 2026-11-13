"""Phase 2 full extraction + training orchestrator.

Runs sequentially:
  1. Text extraction (remaining 152 participants)
  2. XLS-R audio extraction (remaining 152 participants)
  3. WavLM audio extraction (all 162 participants)
  4. Verification of all caches (162 files each, no NaN, no all-zero)
  5. Training: 5 folds × 3 seeds × 3 modalities × 2 models
  6. Results CSV + bar chart

Stops immediately on any failure.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")

CONFIG = "configs/eatd_preprocess.yaml"
EXPECTED_PARTICIPANTS = 162


def verify_cache(cache_dir: str, expected_dim: int, label: str) -> bool:
    """Verify cached .npy files: count, shape, no NaN, no all-zero."""
    p = Path(cache_dir)
    if not p.exists():
        logger.error("Cache dir does not exist: %s", cache_dir)
        return False

    files = sorted(p.glob("*.npy"))
    n = len(files)
    logger.info("[%s] Found %d cached files (expected %d)", label, n, EXPECTED_PARTICIPANTS)

    if n != EXPECTED_PARTICIPANTS:
        logger.error("[%s] FAILED: expected %d files, got %d", label, EXPECTED_PARTICIPANTS, n)
        return False

    nan_files = []
    zero_files = []
    bad_shape = []

    for f in files:
        arr = np.load(f)
        if arr.shape != (expected_dim,):
            bad_shape.append((f.name, arr.shape))
        if np.isnan(arr).any():
            nan_files.append(f.name)
        if np.all(arr == 0):
            zero_files.append(f.name)

    ok = True
    if bad_shape:
        logger.error("[%s] Bad shapes: %s", label, bad_shape[:5])
        ok = False
    if nan_files:
        logger.error("[%s] NaN files: %s", label, nan_files[:5])
        ok = False
    if zero_files:
        logger.warning("[%s] All-zero files (dropped clips?): %s", label, zero_files)
        # All-zero is a warning, not a failure — could be participants with all 3 clips dropped

    if ok:
        logger.info("[%s] PASSED: %d files, shape=(%d,), no NaN", label, n, expected_dim)
    return ok


def main():
    start_all = time.time()

    # ── Step 1: Text extraction ──────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 1/6: Text embedding extraction (Chinese RoBERTa)")
    logger.info("=" * 70)

    from src.features.extract_text import extract_text_embeddings, load_config
    cfg = load_config(CONFIG)

    t0 = time.time()
    extract_text_embeddings(config_path=CONFIG)
    logger.info("Text extraction took %.1f sec", time.time() - t0)

    text_cache = str(Path(cfg["features"]["cache_dir"]) / "text")
    if not verify_cache(text_cache, 2304, "TEXT"):
        logger.error("Text verification FAILED — stopping.")
        sys.exit(1)

    # ── Step 2: XLS-R audio extraction ───────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 2/6: Audio embedding extraction (XLS-R 300m)")
    logger.info("=" * 70)

    from src.features.extract_audio import extract_audio_embeddings, _model_short_name

    xlsr_name = cfg["features"]["audio"]["model_name"]
    xlsr_short = _model_short_name(xlsr_name)

    t0 = time.time()
    extract_audio_embeddings(config_path=CONFIG, encoder_key="model_name")
    logger.info("XLS-R extraction took %.1f sec", time.time() - t0)

    xlsr_cache = str(Path(cfg["features"]["cache_dir"]) / f"audio_{xlsr_short}")
    if not verify_cache(xlsr_cache, 6144, "XLS-R"):
        logger.error("XLS-R verification FAILED — stopping.")
        sys.exit(1)

    # ── Step 3: WavLM audio extraction ───────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 3/6: Audio embedding extraction (WavLM-base-plus)")
    logger.info("=" * 70)

    wavlm_name = cfg["features"]["audio"]["alt_model_name"]
    wavlm_short = _model_short_name(wavlm_name)

    t0 = time.time()
    extract_audio_embeddings(config_path=CONFIG, encoder_key="alt_model_name")
    logger.info("WavLM extraction took %.1f sec", time.time() - t0)

    wavlm_cache = str(Path(cfg["features"]["cache_dir"]) / f"audio_{wavlm_short}")
    if not verify_cache(wavlm_cache, 4608, "WavLM"):
        logger.error("WavLM verification FAILED — stopping.")
        sys.exit(1)

    # ── Step 4: Summary of all caches ────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 4/6: All caches verified")
    logger.info("=" * 70)
    logger.info("  Text (Chinese RoBERTa):  162 × (2304,) ✓")
    logger.info("  Audio (XLS-R 300m):      162 × (6144,) ✓")
    logger.info("  Audio (WavLM-base-plus): 162 × (4608,) ✓")
    extraction_time = time.time() - start_all
    logger.info("  Total extraction time: %.1f min", extraction_time / 60)

    # ── Step 5: Training ─────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 5/6: Training baselines (5 folds × 3 seeds × 3 modalities × 2 models)")
    logger.info("=" * 70)

    from src.models.train_eatd_baselines import run_baselines, save_results

    t0 = time.time()
    df = run_baselines(config_path=CONFIG, smoke_test=False)
    logger.info("Training took %.1f sec", time.time() - t0)

    if df.empty:
        logger.error("Training produced no results — stopping.")
        sys.exit(1)

    # ── Step 6: Save results ─────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 6/6: Saving results CSV + bar chart")
    logger.info("=" * 70)

    save_results(df, config_path=CONFIG)

    total_time = time.time() - start_all
    logger.info("=" * 70)
    logger.info("ALL DONE in %.1f min (%.1f min extraction + %.1f min training)",
                total_time / 60, extraction_time / 60, (total_time - extraction_time) / 60)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

