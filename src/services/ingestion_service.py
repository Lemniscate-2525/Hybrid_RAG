"""
Document Ingestion Service.
Handles magic-byte validation, page limit checks, text extraction,
punctuation-preserving cleaning, chunking, and dual embedding indexing into Qdrant.
"""
import io
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import filetype
from pypdf import PdfReader
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from src.config import settings
from src.db.models import User, DocumentRecord
from src.services.embedding_service import embedding_service
from src.services.qdrant_service import qdrant_service

class IngestionService:
    def __init__(self):
        self.splitter = SentenceSplitter(chunk_size=1500, chunk_overlap=200)

    def clean_text(self, text: str) -> str:
        """
        Clean text while strictly preserving punctuation, sentence boundaries,
        and technical symbols essential for semantic chunking and RAG.
        """
        # Collapse multiple whitespace characters into single space
        cleaned = re.sub(r'[\r\n\t]+', ' ', text)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        return cleaned.strip()

    def sanitize_filename(self, filename: str) -> str:
        """Strip path traversal components and unsafe characters from filenames."""
        base = Path(filename).name
        # Allow alphanumeric, hyphen, underscore, and dot
        sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', base)
        return sanitized or "document.pdf"

    async def validate_and_read_upload(self, file: UploadFile) -> Tuple[bytes, str, str]:
        """Validate file size, extension, and magic bytes."""
        filename = self.sanitize_filename(file.filename or "upload.pdf")
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF documents are supported."
            )

        contents = await file.read()
        file_size = len(contents)

        if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
            )

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        # Magic byte validation
        kind = filetype.guess(contents)
        if kind is None or kind.mime != "application/pdf":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File content does not match a valid PDF signature."
            )

        sha256_hash = hashlib.sha256(contents).hexdigest()
        return contents, filename, sha256_hash

    def extract_pdf_chunks(self, contents: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extract pages and split into chunks with preserved metadata."""
        try:
            reader = PdfReader(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse PDF document: malformed or corrupted file. Details: {str(e)}"
            )

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The provided PDF contains no readable pages."
            )

        if total_pages > settings.MAX_DOCUMENT_PAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PDF page count ({total_pages}) exceeds allowed limit of {settings.MAX_DOCUMENT_PAGES} pages."
            )

        documents = []
        for page_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = self.clean_text(raw_text)
            if cleaned:
                doc = Document(
                    text=cleaned,
                    metadata={
                        "file_name": filename,
                        "page_label": str(page_idx + 1)
                    }
                )
                documents.append(doc)

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract any text from the PDF. Scanned images without OCR are unsupported."
            )

        nodes = self.splitter.get_nodes_from_documents(documents)
        chunks = []
        for node in nodes:
            chunks.append({
                "text": node.text,
                "page_label": node.metadata.get("page_label", "1"),
                "file_name": filename
            })

        return chunks

    async def ingest_document(self, file: UploadFile, user: User, db: Session) -> DocumentRecord:
        """Full ingestion lifecycle with deduplication, chunking, embedding, and Qdrant storage."""
        contents, sanitized_filename, sha256_hash = await self.validate_and_read_upload(file)

        # Duplicate check per tenant
        existing = db.query(DocumentRecord).filter(
            DocumentRecord.user_id == user.id,
            DocumentRecord.sha256_hash == sha256_hash
        ).first()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document '{existing.original_filename}' has already been indexed."
            )

        # Extract and chunk
        chunks = self.extract_pdf_chunks(contents, sanitized_filename)
        texts = [c["text"] for c in chunks]

        # Generate embeddings
        dense_vectors = embedding_service.embed_dense(texts)
        sparse_vectors = embedding_service.embed_sparse(texts)

        # Save file to user-isolated directory
        user_upload_dir = settings.UPLOAD_DIR / user.id
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        doc_record = DocumentRecord(
            user_id=user.id,
            original_filename=sanitized_filename,
            stored_path="",
            sha256_hash=sha256_hash,
            file_size_bytes=len(contents),
            chunk_count=len(chunks)
        )
        db.add(doc_record)
        db.commit()
        db.refresh(doc_record)

        # Store path with doc_record id
        file_path = user_upload_dir / f"{doc_record.id}_{sanitized_filename}"
        with open(file_path, "wb") as f:
            f.write(contents)

        doc_record.stored_path = str(file_path)
        db.commit()

        # Upsert into Qdrant
        qdrant_service.upsert_chunks(
            user_id=user.id,
            doc_id=doc_record.id,
            file_name=sanitized_filename,
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors
        )

        return doc_record

ingestion_service = IngestionService()

