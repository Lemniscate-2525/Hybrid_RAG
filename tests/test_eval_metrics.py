from metrics import (
    precision_at_k, recall_at_k, reciprocal_rank,
    ndcg_at_k, hit_rate_at_k, evaluate_retrieval,
)

def test_precision_at_k_uses_fixed_k_denominator():
    assert precision_at_k(["a"], {"a"}, k=5) == 0.2

def test_recall_at_k_uses_all_relevant_chunks():
    assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

def test_reciprocal_rank_respects_cutoff():
    assert reciprocal_rank(["x", "a"], {"a"}, k=1) == 0.0
    assert reciprocal_rank(["x", "a"], {"a"}, k=2) == 0.5

def test_ndcg_uses_graded_relevance():
    score = ndcg_at_k(["low", "high"], {"high": 3, "low": 1}, k=2)
    assert 0.0 < score < 1.0

def test_hit_rate():
    assert hit_rate_at_k(["x", "a"], {"a"}, k=2) == 1.0
    assert hit_rate_at_k(["x"], {"a"}, k=1) == 0.0

def test_evaluate_retrieval_returns_all_metrics():
    result = evaluate_retrieval(["a", "b"], {"a": 3, "b": 1}, k=2)
    assert set(result) == {"precision@2", "recall@2", "mrr@2", "ndcg@2", "hit_rate@2"}
