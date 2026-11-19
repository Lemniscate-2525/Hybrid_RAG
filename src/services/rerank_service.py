"""
Reranker Service: Lazy Singleton Cross-Encoder Reranker.
Scores and sorts candidate passages using ms-marco-MiniLM-L-6-v2.
Supports lightweight CPU ONNX runtime (FastEmbed) to eliminate heavy PyTorch/CUDA
overhead and maintain memory usage under 512MB RAM.
"""
from typing import List, Dict, Any, Optional
import threading
import logging
from src.config import settings

logger = logging.getLogger(__name__)

class RerankerService:
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super(RerankerService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._singleton_lock:
            if self._initialized:
                return
            self._model = None
            self._backend: Optional[str] = None
            self._model_lock = threading.Lock()
            self._initialized = True
            logger.info("RerankerService configured (model will be lazily initialized on demand).")

    @property
    def is_loaded(self) -> bool:
        """Check if reranker model is currently loaded in memory."""
        return self._model is not None

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def _get_model(self):
        """Thread-safe lazy initialization for the Cross-Encoder model."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    # Prefer ONNX TextCrossEncoder for low-memory CPU production
                    if getattr(settings, "USE_ONNX_RERANKER", True):
                        try:
                            from fastembed.rerank.cross_encoder import TextCrossEncoder
                            model_name = getattr(settings, "ONNX_RERANKER_MODEL_NAME", "Xenova/ms-marco-MiniLM-L-6-v2")
                            logger.info("Lazily loading ONNX Cross-Encoder model: %s", model_name)
                            self._model = TextCrossEncoder(model_name=model_name)
                            self._backend = "fastembed_onnx"
                            logger.info("ONNX Cross-Encoder loaded successfully (memory-efficient).")
                            return self._model
                        except Exception as e:
                            logger.warning("Could not load FastEmbed ONNX Cross-Encoder (%s). Falling back to sentence-transformers.", e)

                    # Fallback to SentenceTransformers CrossEncoder
                    try:
                        from sentence_transformers import CrossEncoder
                        model_name = getattr(settings, "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                        logger.info("Lazily loading PyTorch CrossEncoder: %s", model_name)
                        self._model = CrossEncoder(model_name)
                        self._backend = "sentence_transformers"
                        logger.info("PyTorch CrossEncoder loaded successfully.")
                        return self._model
                    except Exception as e:
                        logger.error("Failed to load any Cross-Encoder implementation: %s", e)
                        raise RuntimeError(f"Cross-Encoder initialization failed: {e}") from e

        return self._model

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunk dictionaries based on query-passage relevance.
        Each candidate chunk must contain 'text'.
        """
        if not candidate_chunks:
            return []

        model = self._get_model()
        passages = [chunk.get("text", "") for chunk in candidate_chunks]

        if self._backend == "fastembed_onnx":
            # FastEmbed TextCrossEncoder.rerank returns scores iterator
            raw_scores = list(model.rerank(query, passages))
            scores = [float(s) for s in raw_scores]
        else:
            # SentenceTransformers CrossEncoder.predict expects pairs
            pairs = [(query, text) for text in passages]
            scores = [float(s) for s in model.predict(pairs)]

        ranked = []
        for chunk, score in zip(candidate_chunks, scores):
            updated_chunk = dict(chunk)
            updated_chunk["rerank_score"] = score
            ranked.append(updated_chunk)

        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

reranker_service = RerankerService()
