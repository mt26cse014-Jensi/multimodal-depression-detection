"""Utilities for aggregating per-answer embeddings into participant vectors."""

from typing import Optional, Sequence

import numpy as np


def aggregate_answer_embeddings(
    embedding: np.ndarray,
    n_answers: int = 3,
    mode: str = "concat",
    valid_mask: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """Aggregate a participant's concatenated answer embeddings.

    Cached embeddings store ``n_answers`` equal-sized answer vectors concatenated
    along a single 1-D array of shape ``(n_answers * dim,)``.

    Args:
        embedding: 1-D array of concatenated answer embeddings.
        n_answers: Number of answer segments concatenated in ``embedding``.
        mode: Aggregation strategy:
            - ``"concat"``: return the concatenated vector unchanged.
            - ``"mean"``: reshape to ``(n_answers, dim)`` and average across
              valid answers (or all answers if ``valid_mask`` is ``None``).
        valid_mask: Optional boolean sequence of length ``n_answers`` indicating
            which answers are valid (e.g., not dropped during preprocessing).
            Only used when ``mode == "mean"``.

    Returns:
        1-D float32 numpy array of shape ``(n_answers * dim,)`` for
        ``"concat"`` or ``(dim,)`` for ``"mean"``.
    """
    if embedding.ndim != 1:
        raise ValueError(f"Expected 1-D embedding, got shape {embedding.shape}")
    if n_answers < 1 or embedding.shape[0] % n_answers != 0:
        raise ValueError(
            f"Embedding length {embedding.shape[0]} is not divisible by n_answers={n_answers}"
        )

    if mode == "concat":
        return embedding.astype(np.float32, copy=False)

    if mode != "mean":
        raise ValueError(f"Unsupported aggregation mode: {mode!r}. Expected 'concat' or 'mean'.")

    parts = embedding.reshape(n_answers, -1)
    if valid_mask is None:
        return parts.mean(axis=0).astype(np.float32)

    if len(valid_mask) != n_answers:
        raise ValueError(
            f"valid_mask length ({len(valid_mask)}) must match n_answers ({n_answers})"
        )

    mask = np.asarray(valid_mask, dtype=bool)
    if not mask.any():
        return np.zeros(parts.shape[1], dtype=np.float32)

    return parts[mask].mean(axis=0).astype(np.float32)

