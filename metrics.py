"""Chunk-level information-retrieval metrics for RAG evaluation."""
import math
from typing import Dict, List, Set, Union, Sequence

Ids = Union[List[str], Set[str]]

def precision_at_k(retrieved_ids: List[str], relevant_ids: Ids, k: int = 5) -> float:
    """Relevant retrieved items divided by K (standard fixed-denominator P@K)."""
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    return round(sum(item in relevant for item in retrieved_ids[:k]) / k, 4)

def recall_at_k(retrieved_ids: List[str], relevant_ids: Ids, k: int = 5) -> float:
    """Fraction of all known relevant items found in the first K results."""
    relevant = set(relevant_ids)
    if k <= 0 or not relevant:
        return 0.0
    return round(len(set(retrieved_ids[:k]) & relevant) / len(relevant), 4)

def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Ids, k: int = 0) -> float:
    """Reciprocal rank of the first relevant result; k=0 means no cutoff."""
    relevant = set(relevant_ids)
    candidates = retrieved_ids[:k] if k > 0 else retrieved_ids
    for rank, item in enumerate(candidates, start=1):
        if item in relevant:
            return round(1.0 / rank, 4)
    return 0.0

def mean_reciprocal_rank(all_retrieved: List[List[str]], all_relevant: List[List[str]]) -> float:
    if not all_retrieved or len(all_retrieved) != len(all_relevant):
        return 0.0
    scores = [reciprocal_rank(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)]
    return round(sum(scores) / len(scores), 4)

def dcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, float], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    return sum(
        (2.0 ** float(relevance_map.get(item, 0.0)) - 1.0) / math.log2(rank + 1)
        for rank, item in enumerate(retrieved_ids[:k], start=1)
    )

def ndcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, float], k: int = 5) -> float:
    if not relevance_map or k <= 0:
        return 0.0
    actual = dcg_at_k(retrieved_ids, relevance_map, k)
    ideal_ids = sorted(relevance_map, key=lambda item: relevance_map[item], reverse=True)
    ideal = dcg_at_k(ideal_ids, relevance_map, k)
    return round(actual / ideal, 4) if ideal else 0.0

def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Ids, k: int = 5) -> float:
    if k <= 0:
        return 0.0
    return float(bool(set(retrieved_ids[:k]) & set(relevant_ids)))

def evaluate_retrieval(retrieved_ids: List[str], qrels: Dict[str, float], k: int = 5) -> Dict[str, float]:
    """Evaluate chunk IDs against graded qrels: {chunk_id: relevance_grade}."""
    relevant = {chunk_id for chunk_id, grade in qrels.items() if float(grade) > 0}
    return {
        f"precision@{k}": precision_at_k(retrieved_ids, relevant, k),
        f"recall@{k}": recall_at_k(retrieved_ids, relevant, k),
        f"mrr@{k}": reciprocal_rank(retrieved_ids, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, qrels, k),
        f"hit_rate@{k}": hit_rate_at_k(retrieved_ids, relevant, k),
    }
