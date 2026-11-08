"""
Document Management Endpoints: Upload, Listing, and Deletion.
Enforces strict multi-tenant authorization at every operation.
"""
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db.models import User, DocumentRecord
from src.api.schemas import DocumentResponse, DocumentListResponse, DeleteDocumentResponse
from src.security.auth import get_current_user
from src.services.ingestion_service import ingestion_service
from src.services.qdrant_service import qdrant_service

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and index a PDF document for the authenticated user.
    Validates magic bytes, enforces page limits, generates BM42 & dense embeddings,
    and isolates chunks by user_id in Qdrant.
    """
    doc_record = await ingestion_service.ingest_document(file=file, user=current_user, db=db)
    return doc_record

@router.get("/", response_model=DocumentListResponse)
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all documents owned by the authenticated user."""
    docs = db.query(DocumentRecord).filter(DocumentRecord.user_id == current_user.id).order_by(DocumentRecord.created_at.desc()).all()
    return DocumentListResponse(total=len(docs), documents=docs)

@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a document owned by the authenticated user.
    Removes the file from disk, purges points from Qdrant, and deletes DB metadata.
    Cross-user deletion attempts are strictly rejected.
    """
    doc = db.query(DocumentRecord).filter(
        DocumentRecord.id == doc_id,
        DocumentRecord.user_id == current_user.id
    ).first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or you do not have permission to delete it."
        )

    # Delete points from Qdrant
    qdrant_service.delete_document_points(user_id=current_user.id, doc_id=doc_id)

    # Delete physical file
    try:
        file_path = Path(doc.stored_path)
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass

    # Delete database record
    db.delete(doc)
    db.commit()

    return DeleteDocumentResponse(
        id=doc_id,
        deleted=True,
        message=f"Document '{doc.original_filename}' and its embeddings were permanently deleted."
    )

