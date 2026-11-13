"""Phase 4 full pipeline runner for DAIC-WOZ preprocessing.

Steps:
  1. Parse all transcripts (text variants + cleaning log)
  2. Cut all participant-turn audio (resumable, skips existing)
  3. Build windows.parquet (3 scales: 30s / 75s / 120s)
  4. Run the full test suite
  5. Print statistics table

Usage:
    .venv\\Scripts\\python run_phase4.py
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PYTHON = sys.executable
CONFIG = "configs/daic_preprocess.yaml"


def step(label: str) -> None:
    log.info("=" * 60)
    log.info("STEP: %s", label)
    log.info("=" * 60)


def run(cmd: list[str]) -> None:
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        log.error("Command failed (exit %d) after %.1fs: %s", result.returncode, elapsed, " ".join(cmd))
        sys.exit(result.returncode)
    log.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])

    # ── Step 0: Build/refresh manifest ────────────────────────────────────────
    step("0 / 4  Build DAIC-WOZ manifest")
    run([PYTHON, "-m", "src.data.build_daic_manifest", "--config", CONFIG])

    # ── Step 1: Parse all transcripts ─────────────────────────────────────────
    step("1 / 4  Parse all transcripts")
    run([PYTHON, "-m", "src.data.parse_daic_transcripts", "--config", CONFIG])

    # ── Step 2: Cut all audio ──────────────────────────────────────────────────
    step("2 / 4  Cut participant-turn audio (resumable)")
    run([PYTHON, "-m", "src.data.cut_daic_audio", "--config", CONFIG])

    # ── Step 3: Build windows ──────────────────────────────────────────────────
    step("3 / 4  Build windows.parquet")
    run([PYTHON, "-m", "src.data.build_daic_windows", "--config", CONFIG])

    # ── Step 4: Run tests ──────────────────────────────────────────────────────
    step("4 / 4  Run full test suite")
    run([PYTHON, "-m", "pytest", "--tb=short", "-q"])

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase 4 pipeline complete.")
    stats_path = Path("results/daic/daic_preprocessing_stats.txt")
    if stats_path.exists():
        print("\n" + stats_path.read_text(encoding="utf-8"))

    # Verify windows.parquet
    wp = processed_dir / "windows.parquet"
    if wp.exists():
        df = pd.read_parquet(str(wp))
        log.info("windows.parquet: %d rows, %d sessions, %d scales",
                 len(df), df["session_id"].nunique(), df["scale"].nunique())

    # Verify text files
    text_dir = processed_dir / "text"
    p_files = list(text_dir.glob("*_participant.txt"))
    log.info("Participant text files: %d", len(p_files))

    # Verify turn tables
    turns_dir = processed_dir / "turns"
    turn_csvs = list(turns_dir.glob("*_turns.csv"))
    log.info("Turn table CSVs: %d", len(turn_csvs))

