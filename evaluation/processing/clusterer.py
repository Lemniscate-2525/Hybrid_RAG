"""Lightweight deterministic topic clustering without external ML dependencies."""
import re
from collections import Counter
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

def _terms(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))

class QuestionClusterer:
    """Greedy Jaccard clustering, useful as a first-pass organizer for review."""

    def __init__(self, similarity_threshold: float = 0.22):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1].")
        self.threshold = similarity_threshold

    def cluster(self, candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        for candidate in candidates:
            question = str(candidate.get("question", ""))
            terms = _terms(question)
            best_cluster = None
            best_score = 0.0
            for cluster in clusters:
                union = terms | cluster["_terms"]
                score = len(terms & cluster["_terms"]) / len(union) if union else 0.0
                if score > best_score:
                    best_score, best_cluster = score, cluster
            if best_cluster is not None and best_score >= self.threshold:
                best_cluster["items"].append(candidate)
                best_cluster["_terms"].update(terms)
            else:
                clusters.append({"cluster_id": f"cluster_{len(clusters) + 1:03d}", "items": [candidate], "_terms": set(terms)})

        output = []
        for cluster in clusters:
            counts = Counter(
                str(item.get("category", item.get("scenario_id", "uncategorized")))
                for item in cluster["items"]
            )
            output.append({
                "cluster_id": cluster["cluster_id"],
                "size": len(cluster["items"]),
                "dominant_category": counts.most_common(1)[0][0],
                "question_ids": [item.get("id") for item in cluster["items"] if item.get("id")],
                "questions": [item.get("question", "") for item in cluster["items"]],
            })
        return output
