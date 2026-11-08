"""
Integration & Security Tests for Multi-Tenant Document Isolation (Phase 1 & 2).
"""
import io
import pytest
from pypdf import PdfWriter

def create_sample_pdf(text: str = "This is a test PDF document for Hybrid RAG evaluation.") -> bytes:
    """Generate a valid in-memory PDF with text content."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, text)
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def get_auth_token(client, email, password):
    client.post("/api/v1/auth/register", json={"email": email, "password": password})
    res = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    return res.json()["access_token"]

def test_document_upload_and_isolation(client):
    try:
        from reportlab.pdfgen import canvas
        pdf_bytes = create_sample_pdf("Architecture of Adaptive Retrieval Augmented Generation systems.")
    except ImportError:
        # Fallback minimal valid PDF without reportlab
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 55 >>\nstream\nBT /F1 12 Tf 100 700 Td (Sample test document content.) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000208 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n314\n%%EOF"

    token_a = get_auth_token(client, "user_a@example.com", "PasswordUserA123!")
    token_b = get_auth_token(client, "user_b@example.com", "PasswordUserB456!")

    # 1. User A uploads a valid PDF
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test_doc.pdf", pdf_bytes, "application/pdf")},
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert upload_res.status_code == 201
    doc_a = upload_res.json()
    assert doc_a["original_filename"] == "test_doc.pdf"
    assert doc_a["chunk_count"] >= 1
    doc_id_a = doc_a["id"]

    # 2. User A can list their document
    list_a = client.get("/api/v1/documents/", headers={"Authorization": f"Bearer {token_a}"})
    assert list_a.status_code == 200
    assert any(d["id"] == doc_id_a for d in list_a.json()["documents"])

    # 3. User B lists documents -> Must NOT see User A's document
    list_b = client.get("/api/v1/documents/", headers={"Authorization": f"Bearer {token_b}"})
    assert list_b.status_code == 200
    assert not any(d["id"] == doc_id_a for d in list_b.json()["documents"])

    # 4. User B attempts to delete User A's document -> 404 Not Found (Cross-tenant access blocked)
    del_res = client.delete(f"/api/v1/documents/{doc_id_a}", headers={"Authorization": f"Bearer {token_b}"})
    assert del_res.status_code == 404

    # 5. User A deletes their own document -> 200 OK
    del_res_a = client.delete(f"/api/v1/documents/{doc_id_a}", headers={"Authorization": f"Bearer {token_a}"})
    assert del_res_a.status_code == 200
    assert del_res_a.json()["deleted"] is True

def test_upload_magic_bytes_validation(client):
    token = get_auth_token(client, "magic_check@example.com", "PasswordCheck123!")

    # Fake PDF: text disguised as PDF
    fake_pdf = b"This is plainly a text file and not a PDF."
    res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("fake.pdf", fake_pdf, "application/pdf")},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 415
    assert "PDF signature" in res.json()["detail"]

