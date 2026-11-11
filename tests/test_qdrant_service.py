
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.services.qdrant_service import QdrantService


def create_test_service():
    """Create an isolated in-memory Qdrant service."""
    service = QdrantService.__new__(QdrantService)

    service.client = QdrantClient(":memory:")
    service.collection_name = "test_hybrid_collection"

    service.client.create_collection(
        collection_name=service.collection_name,
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

    return service


def test_scroll_chunks_returns_actual_chunk_ids():
    service = create_test_service()

    try:
        user_id = "test-user-001"
        doc_id = "test-doc-001"

        chunks = [
            {"text": "Database management systems", "page_label": "1"},
            {"text": "SQL databases and indexing", "page_label": "2"}
        ]

        dense_vectors = [[0.1] * 384, [0.2] * 384]

        sparse_vectors = [
            qmodels.SparseVector(indices=[1], values=[0.5]),
            qmodels.SparseVector(indices=[2], values=[0.8])
        ]

        service.upsert_chunks(
            user_id=user_id,
            doc_id=doc_id,
            file_name="test.pdf",
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors
        )

        results, next_offset = service.scroll_chunks(
            user_id=user_id,
            doc_id=doc_id,
            limit=10
        )

        assert len(results) == 2
        assert all(result["chunk_id"] for result in results)
        assert all(result["user_id"] == user_id for result in results)
        assert all(result["doc_id"] == doc_id for result in results)
        assert next_offset is None

    finally:
        service.client.close()


def test_scroll_chunks_enforces_tenant_isolation():
    service = create_test_service()

    try:
        chunks = [{"text": "Private document content", "page_label": "1"}]

        service.upsert_chunks(
            user_id="user-A",
            doc_id="doc-A",
            file_name="private.pdf",
            chunks=chunks,
            dense_vectors=[[0.1] * 384],
            sparse_vectors=[
                qmodels.SparseVector(indices=[1], values=[0.5])
            ]
        )

        results, _ = service.scroll_chunks(
            user_id="user-B",
            limit=10
        )

        assert results == []

    finally:
        service.client.close()
