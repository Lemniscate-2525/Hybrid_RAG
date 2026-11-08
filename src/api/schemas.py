"""
Pydantic V2 Data Transfer Object (DTO) Schemas.
Implements strict input validation, type casting, and schema enforcement.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator

# ----------------- AUTH SCHEMAS -----------------

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128, description="Password must be at least 8 characters")

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user_id: str
    email: str

class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    is_active: bool
    created_at: datetime

# ----------------- DOCUMENT SCHEMAS -----------------

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    file_size_bytes: int
    chunk_count: int
    created_at: datetime

class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentResponse]

class DeleteDocumentResponse(BaseModel):
    id: str
    deleted: bool
    message: str

# ----------------- RAG QUERY SCHEMAS -----------------

class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="The user query text")
    filename: Optional[str] = Field(default=None, max_length=255, description="Optional document filter")
    top_k: int = Field(default=5, ge=1, le=20, description="Max candidate chunks retrieved before reranking")

    @field_validator("query")
    @classmethod
    def strip_and_clean_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or purely whitespace")
        return cleaned

class CitationItem(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    snippet: str
    relevance_score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationItem]
    is_abstention: bool = False
    cache_hit: bool = False
    duration_ms: int
    correlation_id: Optional[str] = None

# ----------------- HEALTH & SYSTEM SCHEMAS -----------------

class HealthLiveResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime

class ComponentStatus(BaseModel):
    status: str
    details: Optional[str] = None

class HealthReadyResponse(BaseModel):
    status: str
    timestamp: datetime
    components: Dict[str, ComponentStatus]

