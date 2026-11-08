"""
Embedding Service: Singleton manager for Dense and BM42 Sparse Embeddings.
Reuses model instances across requests for high throughput and reduced memory.
"""
from typing import List, Tuple
import threading
import logging
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.http import models as qmodels
from src.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            logger.info("Initializing EmbeddingService (Dense + BM42 Sparse)...")
            self.dense_model = TextEmbedding(model_name=settings.DENSE_MODEL_NAME)
            self.sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
            self._initialized = True
            logger.info("EmbeddingService initialized successfully.")

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Compute dense embeddings for a list of texts."""
        sanitized = [t if t.strip() else " " for t in texts]
        embeddings = list(self.dense_model.embed(sanitized))
        return [emb.tolist() for emb in embeddings]

    def embed_sparse(self, texts: List[str]) -> List[qmodels.SparseVector]:
        """Compute BM42 sparse representations for a list of texts."""
        sanitized = [t if t.strip() else " " for t in texts]
        embeddings = list(self.sparse_model.embed(sanitized))
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

