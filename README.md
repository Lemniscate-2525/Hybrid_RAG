# Enterprise Hybrid RAG System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid%20Search%20(Dense%20%2B%20BM42)-dc2626?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Free%20%26%20Paid%20Models-6366f1)](https://openrouter.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passed%20(100%25)-brightgreen?logo=pytest&logoColor=white)](tests/)

An enterprise-grade, secure, multi-tenant **Hybrid Retrieval-Augmented Generation (RAG)** platform that combines **Dense Vector Search** (`sentence-transformers/all-MiniLM-L6-v2`), **Sparse Vector Search** (`Qdrant BM42`), **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`), and flexible **LLM Inference** (supporting OpenRouter free models, Mistral AI, and standard OpenAI-compatible APIs).

The system is delivered as a decoupled **FastAPI backend** and a reactive **Streamlit frontend**, fully hardened for multi-tenant isolation, prompt injection containment, distributed rate limiting, and fault tolerance.

---

## 💡 Core Idea & Why Hybrid RAG?

Traditional RAG systems that rely solely on dense vector semantic search often struggle with:
1. **Domain-Specific Vocabulary & Exact Keyword Matching:** Technical acronyms, serial numbers, code identifiers, or exact names get blurred in dense vector space.
2. **Keyword Mismatches:** Pure keyword search (BM25) fails to capture semantic synonyms and contextual nuance.
3. **Retrieval Noise:** Large top-K retrieval results dilute context windows and trigger hallucinations.

### The Hybrid Solution:
- **Dense Retrieval (Semantic Meaning):** Uses `all-MiniLM-L6-v2` embeddings (384 dimensions) to understand query intent and semantic proximity.
- **Sparse Retrieval (BM42 Keyword Attention):** Uses Qdrant's BM42 attention weights to retain precision on exact words, symbols, and terminology.
- **Reciprocal Rank Fusion (RRF):** Fuses the ranked lists of dense and sparse candidates mathematically inside Qdrant without arbitrary threshold tuning.
- **Cross-Encoder Reranking:** Computes joint cross-attention between query and candidate passages using `ms-marco-MiniLM-L-6-v2`, surfacing the top-2 most relevant passages with calibrated relevance scores.
- **Generative Synthesis:** Synthesizes a factual, grounded answer with page-level citations and strict boundary isolation.

---

## 🛠️ Complete Tech Stack & Tools

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **API Framework** | **FastAPI** | `0.116.1` | High-performance asynchronous REST API gateway and routing |
| **ASGI Server** | **Uvicorn** | `0.35.0` | Production ASGI web server |
| **Frontend UI** | **Streamlit** | `1.66.1` | Multi-tenant web chat interface, document manager, citation viewer |
| **Vector Database** | **Qdrant** | `1.20.0` | Vector database with dense, BM42 sparse, and RRF fusion |
| **Dense Embeddings**| **FastEmbed (MiniLM)** | `0.9.1` | High-throughput ONNX-optimized dense semantic embeddings (384d) |
| **Sparse Embeddings**| **FastEmbed (BM42)** | `0.9.1` | Qdrant BM42 sparse attention weights for exact keyword matching |
| **Reranking Model** | **SentenceTransformers** | `6.2.0` | Cross-Encoder `ms-marco-MiniLM-L-6-v2` for passage scoring |
| **LLM Inference** | **OpenRouter / Mistral** | HTTP / REST | Generative model inference (supports free models via OpenRouter) |
| **Chunking Engine** | **LlamaIndex Core** | `0.14.25` | Punctuation-preserving sentence splitter (chunk size: 1500, overlap: 200) |
| **Authentication** | **PyJWT + Bcrypt** | `2.15.1 / 5.0.0` | HS256 JWT tokens & salted bcrypt password hashing (12 rounds) |
| **Relational DB** | **SQLAlchemy** | `2.1.4` | Parameterized ORM for user records, documents, and query audit logs |
| **Rate Limiter** | **Redis (or FakeRedis)** | `8.1.0 / 2.40.0`| Distributed atomic Lua sliding-window rate limiting |
| **Cache Store** | **Redis** | `8.1.0` | Multi-tenant query response caching with auto-invalidation |
| **Validation** | **Pydantic V2 + FileType**| `2.14.3 / 1.2.0` | Strict schema validation, magic-byte inspection, upload limits |
| **Testing** | **Pytest + Starlette** | `9.1.1` | Automated test suite (unit, integration, security, real PDF e2e) |
| **Containers** | **Docker & Compose** | Multi-stage | Non-root containerization and orchestration |

---

## 🏛️ System Architecture Diagram

```mermaid
flowchart TD
    subgraph ClientTier["Client Tier"]
        Browser["User Web Browser"]
        StreamlitUI["Streamlit Frontend (Port 8501)"]
    end

    subgraph Gateway["FastAPI Gateway (Port 8000)"]
        CorrelationMW["Correlation ID Middleware (X-Correlation-ID)"]
        RateLimiter["Redis Sliding-Window Rate Limiter (Atomic Lua)"]
        AuthMiddleware["JWT Authentication & Tenant Context (Bcrypt + PyJWT)"]
        ValidationMW["Payload & Magic-Byte Ingestion Guardrails"]
    end

    subgraph SecurityShield["Security & Defenses"]
        PromptGuard["Prompt Guard (Direct & Indirect Injection Screening)"]
        ContextIsolator["Context Demarcation (<retrieved_evidence> tags)"]
        CitationVerifier["Grounding & Citation Verification"]
    end

    subgraph Persistence["Storage & State"]
        RelationalDB[("SQLite / PostgreSQL: Users & Metadata")]
        RedisStore[("Redis 8.x: Sliding-Window Rates & Isolated Cache")]
        FileSystem[("Private Storage: /data/uploads/{user_id}/")]
    end

    subgraph CoreEngine["Hardened Hybrid RAG Engine"]
        EmbedService["Singleton Embedding Service (Dense MiniLM + BM42 Sparse)"]
        QdrantEngine[("Qdrant Vector DB (Dense + Sparse + RRF + user_id Filter)")]
        RerankService["Singleton Reranker Service (MS-MARCO Cross-Encoder)"]
        ResilientLLM["Resilient LLM Client (Circuit Breaker & Exponential Backoff)"]
    end

    Browser --> StreamlitUI
    StreamlitUI --> CorrelationMW
    Browser --> CorrelationMW
    CorrelationMW --> RateLimiter
    RateLimiter <--> RedisStore
    RateLimiter --> AuthMiddleware
    AuthMiddleware <--> RelationalDB
    AuthMiddleware --> ValidationMW
    ValidationMW --> PromptGuard

    PromptGuard --> RedisStore
    RedisStore -->|Cache Hit| StreamlitUI
    RedisStore -->|Cache Miss| EmbedService

    EmbedService --> QdrantEngine
    QdrantEngine --> ContextIsolator
    ContextIsolator --> RerankService
    RerankService --> ResilientLLM
    ResilientLLM --> CitationVerifier
    CitationVerifier --> RedisStore
    CitationVerifier --> StreamlitUI
```

---

## 🔄 End-to-End Request-to-Response Traces

### 1. Document Ingestion Trace

```mermaid
sequenceDiagram
    autonumber
    actor User as Authenticated User
    participant UI as Streamlit UI
    participant API as FastAPI /documents/upload
    participant Guard as Ingestion Validator
    participant Embed as FastEmbed Engine
    participant Qdrant as Qdrant Vector DB
    participant DB as SQLite / Postgres

    User->>UI: Upload PDF file
    UI->>API: POST /api/v1/documents/upload (Bearer JWT)
    API->>Guard: Validate magic bytes (%PDF-1.4), file size (<=15MB), page count (<=150)
    Guard-->>API: Validated & Sanitized filename
    API->>Guard: Punctuation-preserving text cleaning & SentenceSplitter
    Guard-->>API: Document chunks with page metadata
    API->>Embed: Generate dense (MiniLM) & sparse (BM42) embeddings
    Embed-->>API: Dense vectors (384d) + BM42 sparse vectors
    API->>DB: Save DocumentRecord (user_id, sha256_hash, file_size)
    API->>Qdrant: Upsert PointStructs with UUIDv5 and user_id payload filter
    API-->>UI: 201 Created (document_id, chunk_count)
    UI-->>User: Display document in "Your Documents" list
```

### 2. Hybrid Search & Generative Query Trace

```mermaid
sequenceDiagram
    autonumber
    actor User as Authenticated User
    participant UI as Streamlit UI
    participant Gateway as FastAPI Gateway
    participant Rate as Redis Rate Limiter
    participant Guard as Prompt Guard
    participant Cache as Redis Cache
    participant Embed as FastEmbed Engine
    participant Qdrant as Qdrant Vector DB
    participant Rerank as Cross-Encoder Reranker
    participant LLM as Resilient LLM Client
    
    User->>UI: Submit question in chat bar
    UI->>Gateway: POST /api/v1/rag/query (Bearer JWT, query, scope)
    Gateway->>Rate: Check sliding-window limit (Lua script keyed by user_id)
    Rate-->>Gateway: Allowed (remaining quota, Retry-After)
    Gateway->>Guard: Scan query for direct prompt injection & jailbreak patterns
    Guard-->>Gateway: Query verified safe
    Gateway->>Cache: Check tenant cache (cache:v1:rag:{user_id}:{hash})
    alt Cache Hit
        Cache-->>Gateway: Cached response + citations
    else Cache Miss
        Gateway->>Embed: Embed query (Dense 384d + Sparse BM42)
        Embed-->>Gateway: Dense vector + Sparse vector
        Gateway->>Qdrant: Hybrid search (Prefetch Dense + Sparse, RRF Fusion, user_id == current_user.id)
        Qdrant-->>Gateway: Top candidate chunks
        Gateway->>Guard: Sanitize role spoofing & wrap in <retrieved_evidence> tags
        Gateway->>Rerank: CrossEncoder predict [(query, passage_1), (query, passage_2)...]
        Rerank-->>Gateway: Top-2 reranked passages with relevance scores
        Gateway->>LLM: Generate response (System prompt with evidence constraints + circuit breaker)
        LLM-->>Gateway: Grounded, factual response
        Gateway->>Cache: Store response in Redis (TTL: 1 hour)
    end
    Gateway-->>UI: 200 OK (answer, citations, duration_ms, correlation_id)
    UI-->>User: Render answer with collapsible citations drawer & latency badge
```

---

## 🛡️ Production Hardening & Security Features

### 1. Multi-Tenant Resource Isolation
- **Authentication:** Mandatory Bearer JWT tokens on all resource routes. Password hashing via `bcrypt` with 12 rounds.
- **Server-Derived Identity:** Identity (`user_id`) is strictly extracted from verified JWT claims (`sub`). Client-supplied user IDs in headers or bodies are completely ignored.
- **Vector Partitioning:** Every Qdrant hybrid query enforces `FieldCondition(key="user_id", match=MatchValue(value=current_user.id))`. Vector points use deterministic UUIDv5 identifiers: `uuid5(NAMESPACE_DNS, "{user_id}:{doc_id}:{chunk_idx}")`, eliminating ID collisions.
- **Cross-Tenant Blocking:** Users can only view, query, or delete their own documents. Cross-tenant deletion attempts return HTTP 404.

### 2. Prompt Injection Defenses
- **Direct Injection Screening:** Queries are scanned in `src/security/prompt_guard.py` using heuristic regexes to intercept instruction override attempts (`ignore previous instructions`), system prompt exfiltration, and privilege escalation. High-risk queries are rejected with HTTP 400.
- **Indirect Injection Containment:** Retrieved passages are sanitized to replace fake role markers (e.g. `[SYSTEM]` is replaced with `[quoted_text]`) and encased in `<retrieved_evidence id="..." source="..." page="...">` boundary tags.
- **System Prompt Hardening:** Explicit rules instruct the model that content inside `<retrieved_evidence>` is untrusted reference evidence and must never be executed as commands.
- **Grounded Citations & Abstention:** Structured chunk citations with file names, page numbers, and relevance scores. If retrieved evidence is empty, the system deterministically abstains without calling the LLM.

### 3. Distributed Redis Rate Limiting
- **Atomic Lua Sliding Window:** Concurrency-safe algorithm running over Redis sorted sets (`ZADD`, `ZREMRANGEBYSCORE`, `ZCARD`) preventing race conditions.
- **Isolated Quotas:** Rate limits are keyed per verified user ID for authenticated requests, and normalized client IP for unauthenticated routes.
- **Trusted Proxy Validation:** Only trusts `X-Forwarded-For` when requests originate from configured `TRUSTED_PROXIES`.
- **Fail-Closed Policy:** Sensitive routes (authentication, document uploads, LLM inference) fail-closed if Redis is down, preventing unmetered traffic storms. Emits HTTP 429 with standard `Retry-After` headers.

### 4. LLM Reliability & Fault Tolerance
- **Circuit Breaker:** Transitions from `CLOSED` to `OPEN` on 5 consecutive failures, fast-failing for 30 seconds to protect upstream API quotas. Probes recovery via a `HALF_OPEN` state.
- **Exponential Backoff:** Bounded retries (max 3) with full randomized jitter for transient 429 and 5xx errors.
- **Elimination of Double Inference:** Fixed the prototype bug where `generation.py` invoked both `llm.complete()` and `TreeSummarize`, cutting inference latency and token spend by ~50%.
- **Local Fallback Synthesis:** If no external LLM API key is configured, the system synthesizes extracted passages directly from your document so queries always succeed.

### 5. Multi-Tenant Caching
- **Isolated Cache Keys:** `cache:v1:rag:{user_id}:{query_hash}` ensures User A never receives cached answers from User B.
- **Automatic Invalidation:** Uploading or deleting a document automatically invalidates all cached query keys belonging to that user.

### 6. Observability & Health Probes
- **Correlation ID Middleware:** Injects `X-Correlation-ID` header into every request and structured JSON log.
- **Latency Breakdown:** Tracks end-to-end response time (`duration_ms`), retrieval time, and cache status.
- **Orchestrator Probes:** Kubernetes/Docker-compatible `/api/v1/health/live` and `/api/v1/health/ready` endpoints validating SQLite, Qdrant, and Redis.

---

## ⚡ Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Hybrid-RAG.git
cd Hybrid-RAG

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the template to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter free API key (or Mistral key):
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LLM_MODEL=meta-llama/llama-3.2-3b-instruct:free
```
*(Leave `Qdrant_URL` empty to use local embedded storage; leave `REDIS_URL` default to use local Redis or embedded FakeRedis)*.

### 3. One-Click Launch (Windows)

Simply run:
```powershell
.\run.bat
```
*(Or `.\run.ps1` in PowerShell)*. This automatically starts both the FastAPI backend and Streamlit UI and opens your browser to `http://localhost:8501`.

### 4. Manual Launch (Two Terminals)

**Terminal 1 — FastAPI Backend:**
```powershell
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```
*API docs available at:* `http://127.0.0.1:8000/docs`

**Terminal 2 — Streamlit Frontend:**
```powershell
python -m streamlit run src/app.py
```
*Web chat available at:* `http://localhost:8501`

---

## 🐳 Docker Deployment

Run the complete multi-container production stack (Qdrant + Redis + FastAPI Backend + Streamlit Frontend):

```bash
# Build and run with Docker Compose
docker compose up -d --build

# Verify container health
docker compose ps
curl http://localhost:8000/api/v1/health/ready
```

### Services Map:
- **FastAPI Backend & Swagger:** `http://localhost:8000/docs`
- **Streamlit Web Application:** `http://localhost:8501`
- **Qdrant Vector DB Console:** `http://localhost:6333/dashboard`
- **Redis Cache:** `localhost:6379`

---

## 🧪 Automated Test Suite

Run the full automated test suite:

```bash
pytest tests/ -v
```

All 20 test cases pass cleanly with 100% pass rate:

```
tests/test_auth.py::test_register_success PASSED                         [  5%]
tests/test_auth.py::test_register_weak_password PASSED                   [ 10%]
tests/test_auth.py::test_register_duplicate_email PASSED                 [ 15%]
tests/test_auth.py::test_login_success PASSED                            [ 20%]
tests/test_auth.py::test_login_invalid_password PASSED                   [ 25%]
tests/test_auth.py::test_protected_route_unauthenticated PASSED          [ 30%]
tests/test_auth.py::test_protected_route_authenticated PASSED            [ 35%]
tests/test_caching.py::test_cache_set_get_and_tenant_isolation PASSED    [ 40%]
tests/test_caching.py::test_cache_invalidation PASSED                    [ 45%]
tests/test_circuit_breaker.py::test_circuit_breaker_tripping_and_recovery PASSED [ 50%]
tests/test_e2e_real_pdf.py::test_e2e_real_pdf_lifecycle PASSED           [ 55%]
tests/test_health.py::test_liveness_probe PASSED                         [ 60%]
tests/test_health.py::test_readiness_probe PASSED                        [ 65%]
tests/test_multitenancy.py::test_document_upload_and_isolation PASSED    [ 70%]
tests/test_multitenancy.py::test_upload_magic_bytes_validation PASSED    [ 75%]
tests/test_rate_limiter.py::test_rate_limiter_exceeded PASSED            [ 80%]
tests/test_security.py::test_direct_prompt_injection_detection PASSED    [ 85%]
tests/test_security.py::test_context_isolation_sanitization PASSED       [ 90%]
tests/test_security.py::test_api_rejects_high_risk_query PASSED          [ 95%]
tests/test_security.py::test_grounded_abstention_when_no_evidence PASSED [100%]

============================= 20 passed in 10.08s =============================
```

---

## ⚙️ Environment Variables Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | Deployment mode: `development`, `testing`, `production` |
| `SECRET_KEY` | *(dev key)* | Minimum 32-character secret for signing JWT access tokens |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` (24h) | JWT session expiration in minutes |
| `OPENROUTER_API_KEY` | `None` | OpenRouter API key for LLM generation |
| `LLM_MODEL` | `meta-llama/llama-3.2-3b-instruct:free` | Model identifier on OpenRouter or OpenAI-compatible endpoint |
| `MistralAI` | `None` | Alternative Mistral AI API key |
| `Qdrant_URL` | `None` (embedded) | Qdrant Cloud or remote server URL |
| `Qdrant_API_KEY` | `None` | Qdrant Cloud API key |
| `Collection_Name` | `hybrid_rag_documents` | Target collection name in Qdrant |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URI for distributed rate limiting and caching |
| `REDIS_RATE_LIMIT_ENABLED`| `true` | Enables atomic sliding-window rate limiting |
| `REDIS_CACHE_ENABLED` | `true` | Enables tenant-isolated response caching |
| `MAX_UPLOAD_SIZE_BYTES` | `15728640` (15MB) | Maximum file size for PDF uploads |
| `MAX_DOCUMENT_PAGES` | `150` | Maximum page count per PDF upload |
| `RATE_LIMIT_QUERY_PER_MINUTE` | `30` | Max hybrid RAG queries per minute per user |
| `LLM_TIMEOUT_SECONDS` | `30.0` | Socket read timeout for inference calls |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive failures before circuit breaker trips to OPEN |

---

## 📡 REST API Reference

| Method | Endpoint | Auth | Description |
| :--- | :--- | :---: | :--- |
| `POST` | `/api/v1/auth/register` | None | Register new user account with bcrypt password hashing |
| `POST` | `/api/v1/auth/login` | None | Authenticate and obtain signed JWT access token |
| `GET` | `/api/v1/auth/me` | Bearer | Get authenticated user profile |
| `POST` | `/api/v1/documents/upload` | Bearer | Upload and index a PDF document into Qdrant |
| `GET` | `/api/v1/documents/` | Bearer | List all documents owned by the authenticated user |
| `DELETE` | `/api/v1/documents/{id}` | Bearer | Delete document from disk, DB, and Qdrant points |
| `POST` | `/api/v1/rag/query` | Bearer | Execute hybrid RAG query with citations and reranking |
| `GET` | `/api/v1/health/live` | None | Liveness probe (200 OK) |
| `GET` | `/api/v1/health/ready` | None | Readiness probe checking SQLite, Qdrant, and Redis |

---

## 📄 License & Attribution

This project is open-source under the MIT License.
Built with **Qdrant BM42**, **LlamaIndex**, **FastEmbed**, **Sentence-Transformers**, and **FastAPI**.
