"""
Embedding Service: Lazy Singleton manager for Dense and BM42 Sparse Embeddings.
Defers model loading until first query/ingestion request to ensure fast startup
and low memory consumption (< 512MB RAM constraint).
"""
from typing import List, Tuple, Optional
import threading
import logging
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.http import models as qmodels
from src.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._singleton_lock:
            if self._initialized:
                return
            self._dense_model: Optional[TextEmbedding] = None
            self._sparse_model: Optional[SparseTextEmbedding] = None
            self._dense_lock = threading.Lock()
            self._sparse_lock = threading.Lock()
            self._initialized = True
            logger.info("EmbeddingService configured (models will be lazily initialized on demand).")

    @property
    def is_dense_loaded(self) -> bool:
        """Check if dense model is currently loaded in memory."""
        return self._dense_model is not None

    @property
    def is_sparse_loaded(self) -> bool:
        """Check if sparse model is currently loaded in memory."""
        return self._sparse_model is not None

    def _get_dense_model(self) -> TextEmbedding:
        """Thread-safe lazy initialization for the dense embedding model."""
        if self._dense_model is None:
            with self._dense_lock:
                if self._dense_model is None:
                    logger.info("Lazily loading Dense Embedding Model: %s", settings.DENSE_MODEL_NAME)
                    try:
                        self._dense_model = TextEmbedding(model_name=settings.DENSE_MODEL_NAME)
                        logger.info("Dense Embedding Model loaded successfully.")
                    except Exception as e:
                        logger.error("Failed to load Dense Embedding Model: %s", e)
                        raise RuntimeError(f"Dense embedding initialization failed: {e}") from e
        return self._dense_model

    def _get_sparse_model(self) -> SparseTextEmbedding:
        """Thread-safe lazy initialization for the BM42 sparse embedding model."""
        if self._sparse_model is None:
            with self._sparse_lock:
                if self._sparse_model is None:
                    logger.info("Lazily loading BM42 Sparse Embedding Model: %s", settings.SPARSE_MODEL_NAME)
                    try:
                        self._sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
                        logger.info("BM42 Sparse Embedding Model loaded successfully.")
                    except Exception as e:
                        logger.error("Failed to load BM42 Sparse Embedding Model: %s", e)
                        raise RuntimeError(f"BM42 sparse embedding initialization failed: {e}") from e
        return self._sparse_model

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Compute dense embeddings for a list of texts."""
        if not texts:
            return []
        model = self._get_dense_model()
        sanitized = [t if t.strip() else " " for t in texts]
        embeddings = list(model.embed(sanitized))
        return [emb.tolist() for emb in embeddings]

    def embed_sparse(self, texts: List[str]) -> List[qmodels.SparseVector]:
        """Compute BM42 sparse representations for a list of texts."""
        if not texts:
            return []
        model = self._get_sparse_model()
        sanitized = [t if t.strip() else " " for t in texts]
        embeddings = list(model.embed(sanitized))
        sparse_vectors = []
        for emb in embeddings:
            if hasattr(emb, "indices") and hasattr(emb, "values"):
                sparse_vectors.append(
                    qmodels.SparseVector(
                        indices=emb.indices.tolist(),
                        values=emb.values.tolist()
                    )
                )
            else:
                sparse_vectors.append(qmodels.SparseVector(indices=[], values=[]))
        return sparse_vectors

    def embed_single_query(self, query: str) -> Tuple[List[float], qmodels.SparseVector]:
        """Compute both dense and sparse representations for a single query."""
        dense = self.embed_dense([query])[0]
        sparse = self.embed_sparse([query])[0]
        return dense, sparse

embedding_service = EmbeddingService()
