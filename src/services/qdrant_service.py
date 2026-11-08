"""
Qdrant Vector Database Service.
Handles multi-tenant collection creation, indexing, payload filtering,
and hybrid search (dense + BM42 sparse + Reciprocal Rank Fusion).
"""
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.config import settings

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        if settings.QDRANT_URL:
            logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=30
            )
        else:
            logger.warning("No QDRANT_URL configured. Initializing local embedded Qdrant in data/qdrant_local")
            local_path = str(settings.UPLOAD_DIR.parent / "qdrant_local")
            self.client = QdrantClient(path=local_path)

        self.collection_name = settings.COLLECTION_NAME
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create the hybrid collection and payload indexes if not already present."""
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": qmodels.VectorParams(
                            size=384,
                            distance=qmodels.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": qmodels.SparseVectorParams(
                            index=qmodels.SparseIndexParams(on_disk=False)
                        )
                    }
                )
                # Create payload indexes on remote Qdrant
                if settings.QDRANT_URL:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="user_id",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="doc_id",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="file_name",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                    )
                logger.info("Qdrant collection and payload indexes created successfully.")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {e}")

    def upsert_chunks(
        self,
        user_id: str,
        doc_id: str,
        file_name: str,
        chunks: List[Dict[str, Any]],
        dense_vectors: List[List[float]],
        sparse_vectors: List[qmodels.SparseVector]
    ) -> int:
        """
        Upsert document chunks into Qdrant with deterministic UUIDs and tenant metadata.
        """
        points = []
        for i, (chunk, dense_emb, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
            # Generate deterministic UUID based on user_id, doc_id, and chunk_index to prevent collisions
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}:{doc_id}:{i}"))
            payload = {
                "user_id": user_id,
                "doc_id": doc_id,
                "file_name": file_name,
                "chunk_index": i,
                "text": chunk.get("text", ""),
                "page_label": chunk.get("page_label", "1"),
            }
            point = qmodels.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_emb,
                    "sparse": sparse_vec,
                },
                payload=payload
            )
            points.append(point)

        # Upsert in batches of 64
        batch_size = 64
        total_upserted = 0
        for b_start in range(0, len(points), batch_size):
            batch = points[b_start : b_start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            total_upserted += len(batch)

        logger.info(f"Upserted {total_upserted} points for user {user_id}, doc {doc_id}")
        return total_upserted

    def hybrid_search(
        self,
        user_id: str,
        dense_query: List[float],
        sparse_query: qmodels.SparseVector,
        filename: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using dense and BM42 sparse prefetch with Reciprocal Rank Fusion.
        CRITICAL: Always enforces user_id filter to maintain multi-tenant isolation.
        """
        # Multi-tenant filter: user_id MUST match
        filter_conditions = [
            qmodels.FieldCondition(
                key="user_id",
                match=qmodels.MatchValue(value=user_id)
            )
        ]

        if filename:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="file_name",
                    match=qmodels.MatchValue(value=filename)
                )
            )

        query_filter = qmodels.Filter(must=filter_conditions)

        prefetch_sparse = qmodels.Prefetch(
            query=sparse_query,
            using="sparse",
            limit=limit,
        )
        prefetch_dense = qmodels.Prefetch(
            query=dense_query,
            using="dense",
            limit=limit,
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[prefetch_sparse, prefetch_dense],
            query_filter=query_filter,
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit
        )

        candidate_chunks = []
        for point in results.points:
            chunk = dict(point.payload)
            chunk["chunk_id"] = str(point.id)
            chunk["score"] = float(point.score) if point.score is not None else 0.0
            candidate_chunks.append(chunk)

        return candidate_chunks

    def delete_document_points(self, user_id: str, doc_id: str) -> None:
        """Delete all points associated with a specific document for a verified user."""
        filter_conditions = [
            qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)),
            qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))
        ]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=filter_conditions)
            )
        )
        logger.info(f"Deleted points in Qdrant for doc {doc_id} of user {user_id}")

qdrant_service = QdrantService()

