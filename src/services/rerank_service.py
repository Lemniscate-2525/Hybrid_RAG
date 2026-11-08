"""
Reranker Service: Singleton Cross-Encoder Reranker.
Scores and sorts candidate passages using ms-marco-MiniLM-L-6-v2.
"""
from typing import List, Dict, Any
import threading
import logging
from sentence_transformers import CrossEncoder
from src.config import settings

logger = logging.getLogger(__name__)

class RerankerService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RerankerService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            logger.info("Initializing RerankerService (Cross-Encoder)...")
            self.model = CrossEncoder(settings.RERANKER_MODEL_NAME)
            self._initialized = True
            logger.info("RerankerService initialized successfully.")

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate chunk dictionaries based on query-passage relevance.
        Each candidate chunk should contain 'text' and other metadata.
        """
        if not candidate_chunks:
            return []

        pairs = [(query, chunk.get("text", "")) for chunk in candidate_chunks]
        scores = self.model.predict(pairs)

        ranked = []
        for chunk, score in zip(candidate_chunks, scores):
            updated_chunk = dict(chunk)
            updated_chunk["rerank_score"] = float(score)
            ranked.append(updated_chunk)

        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

reranker_service = RerankerService()

