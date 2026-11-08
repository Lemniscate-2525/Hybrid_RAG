"""
Information Retrieval (IR) & RAG Evaluation Metrics.
Implements standard mathematical metrics: Precision@K, Recall@K, NDCG@K, and MRR.
"""
import math
from typing import List, Dict, Set, Union

def precision_at_k(retrieved_ids: List[str], ground_truth_relevant_ids: Union[List[str], Set[str]], k: int = 5) -> float:
    """
    Calculate Precision@K:
    Proportion of retrieved documents in the top-k that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    relevant_set = set(ground_truth_relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(hits / k, 4)

def recall_at_k(retrieved_ids: List[str], ground_truth_relevant_ids: Union[List[str], Set[str]], k: int = 5) -> float:
    """
    Calculate Recall@K:
    Proportion of total relevant documents that are successfully retrieved in top-k.
    """
    relevant_set = set(ground_truth_relevant_ids)
    if not relevant_set or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(min(hits / len(relevant_set), 1.0), 4)

def reciprocal_rank(retrieved_ids: List[str], ground_truth_relevant_ids: Union[List[str], Set[str]]) -> float:
    """
    Calculate Reciprocal Rank (RR):
    1 / rank of the first relevant document retrieved. Returns 0 if no relevant document found.
    """
    relevant_set = set(ground_truth_relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0

def mean_reciprocal_rank(all_retrieved: List[List[str]], all_relevant: List[List[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.
    """
    if not all_retrieved:
        return 0.0
    scores = [reciprocal_rank(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)]
    return round(sum(scores) / len(scores), 4)

def dcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, float], k: int = 5) -> float:
    """
    Compute Discounted Cumulative Gain at rank K.
    Formula: sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        rel = relevance_map.get(doc_id, 0.0)
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)
    return dcg

def ndcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, float], k: int = 5) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    Compares the DCG of retrieved items against the Ideal DCG (IDCG).
    Returns a score between 0.0 and 1.0.
    """
    if not relevance_map or k <= 0:
        return 0.0

    actual_dcg = dcg_at_k(retrieved_ids, relevance_map, k)

    # Ideal DCG: sort all known relevant documents descending by their relevance score
    ideal_sorted_ids = sorted(relevance_map.keys(), key=lambda x: relevance_map[x], reverse=True)
    ideal_dcg = dcg_at_k(ideal_sorted_ids, relevance_map, k)

    if ideal_dcg <= 0.0:
        return 0.0

    return round(min(actual_dcg / ideal_dcg, 1.0), 4)

