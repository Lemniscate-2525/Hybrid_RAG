"""Deterministic exact and token-similarity deduplication for question candidates."""
import re
from collections import defaultdict
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def normalize_question(question: str) -> str:
    return " ".join(_TOKEN_RE.findall(question.lower()))

def _tokens(question: str) -> set[str]:
    return set(_TOKEN_RE.findall(question.lower()))

def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0

class QuestionDeduplicator:
    """Greedy near-duplicate removal; preserves the first (usually highest-priority) item."""

    def __init__(self, similarity_threshold: float = 0.88):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1].")
        self.threshold = similarity_threshold

    def deduplicate(
        self, candidates: Iterable[dict[str, Any]], question_key: str = "question"
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        kept: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        exact_seen: dict[str, dict[str, Any]] = {}
        token_sets: list[set[str]] = []

        for candidate in candidates:
            question = str(candidate.get(question_key, "")).strip()
            normalized = normalize_question(question)
            if not normalized:
                duplicates.append({**candidate, "duplicate_reason": "empty_question"})
                continue
            if normalized in exact_seen:
                duplicates.append({
                    **candidate,
                    "duplicate_reason": "exact_match",
                    "duplicate_of": exact_seen[normalized].get("question"),
                })
                continue

            current_tokens = _tokens(question)
            match_index = next(
                (i for i, prior in enumerate(token_sets)
                 if _jaccard(current_tokens, prior) >= self.threshold),
                None,
            )
            if match_index is not None:
                duplicates.append({
                    **candidate,
                    "duplicate_reason": "token_similarity",
                    "duplicate_of": kept[match_index].get("question"),
                })
                continue

            exact_seen[normalized] = candidate
            kept.append(candidate)
            token_sets.append(current_tokens)
        return kept, duplicates
