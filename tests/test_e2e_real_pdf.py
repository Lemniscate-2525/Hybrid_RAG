"""
End-to-End Live Integration Test using actual repository PDF data.
Verifies PDF ingestion, dual embedding, Qdrant hybrid retrieval, reranking, and citation generation.
"""
from pathlib import Path
import pytest
from tests.test_multitenancy import get_auth_token

BASE_DIR = Path(__file__).resolve().parent.parent

def test_e2e_real_pdf_lifecycle(client):
    pdf_path = BASE_DIR / "data" / "Adaptive-RAG.pdf"
    if not pdf_path.exists():
        pytest.skip("Adaptive-RAG.pdf not found in data directory")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # 1. Register & Authenticate User
    token = get_auth_token(client, "e2e_user@example.com", "SecureE2EPass123!")
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload actual PDF
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("Adaptive-RAG.pdf", pdf_bytes, "application/pdf")},
        headers=headers
    )
    assert upload_res.status_code == 201
    doc_data = upload_res.json()
    assert doc_data["original_filename"] == "Adaptive-RAG.pdf"
    assert doc_data["chunk_count"] > 0
    doc_id = doc_data["id"]

    # 3. Query the indexed document
    query_payload = {
        "query": "What is the core idea of Adaptive RAG?",
        "filename": "Adaptive-RAG.pdf",
        "top_k": 5
    }
    query_res = client.post("/api/v1/rag/query", json=query_payload, headers=headers)
    assert query_res.status_code == 200
    rag_data = query_res.json()

    assert rag_data["query"] == "What is the core idea of Adaptive RAG?"
    assert len(rag_data["answer"]) > 0
    assert rag_data["is_abstention"] is False
    assert len(rag_data["citations"]) > 0
    assert rag_data["citations"][0]["file_name"] == "Adaptive-RAG.pdf"
    assert "relevance_score" in rag_data["citations"][0]
    assert rag_data["duration_ms"] > 0

    # 4. Clean up / Delete document
    del_res = client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

