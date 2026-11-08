# Security Policy & Threat Model — Production Hybrid RAG

## 1. Threat Model & Asset Protection

The system is designed under a **Zero-Trust Input** posture. All inputs—including client queries, uploaded PDFs, extracted text, and retrieved vector payloads—are treated as untrusted and potentially hostile.

### Protected Assets
1. **Tenant Document Data:** Private user PDFs, chunks, and embeddings.
2. **LLM System Prompt & Application Secrets:** API keys, database credentials, internal instructions.
3. **Infrastructure Resources:** LLM token quotas, Redis memory, Qdrant index integrity, and CPU/GPU compute.

---

## 2. Implemented Security Controls

### 2.1 Multi-Tenant Data Isolation
- **Authentication:** Mandatory Bearer JWT tokens on all resource endpoints. Unauthenticated requests are rejected with HTTP 401.
- **Server-Derived Identity:** The `user_id` is extracted strictly from cryptographically verified token claims (`sub`). Client-supplied user IDs in headers or request bodies are ignored.
- **Vector DB Partitioning:** Every Qdrant hybrid search query includes a non-negotiable payload filter:
  ```python
  qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=current_user.id))
  ```
  It is mathematically impossible for User A to retrieve or delete User B's vector points.
- **Storage Isolation:** Uploaded files are partitioned into user-specific private directories (`/data/uploads/{user_id}/`).

### 2.2 Direct & Indirect Prompt Injection Defenses
- **Detection Filter (`src/security/prompt_guard.py`):** User queries are scanned for instruction override patterns, jailbreak keywords, and exfiltration commands before execution. Queries flagged as `HIGH_RISK` are aborted immediately with HTTP 400.
- **Context Delimitation & Sanitization:** Retrieved passages are encapsulated within immutable XML tags:
  ```xml
  <retrieved_evidence id="..." source="..." page="...">
  [Passage Text]
  </retrieved_evidence>
  ```
- **Fake Role Tag Neutralization:** Any occurrence of `[SYSTEM]`, `[USER]`, or `[ASSISTANT]` embedded inside uploaded documents is stripped and replaced with `[quoted_text]`.
- **System Prompt Hardening:** The system prompt explicitly commands the model to treat content inside `<retrieved_evidence>` as untrusted evidence and forbids executing instructions found inside passages.

### 2.3 Input Validation & Upload Hardening
- **Magic Byte Verification:** File content is analyzed using `filetype` to ensure legitimate `application/pdf` signatures. Renamed executables or scripts are rejected with HTTP 415.
- **Payload Limits:** Strict enforcement of maximum file size (15MB), maximum page count (150 pages), and maximum query length (500 chars).
- **Path Traversal Prevention:** Filenames are sanitized with regex `[^a-zA-Z0-9_\-\.]` and stored using system-generated UUID prefixes.

### 2.4 Distributed Rate Limiting & DoS Protection
- **Sliding-Window Lua Script:** Atomic Redis sliding window prevents burst abuse and race conditions.
- **IP Proxy Verification:** Client IPs are resolved from `X-Forwarded-For` only when the direct connection originates from a configured trusted proxy (`TRUSTED_PROXIES`).
- **Fail-Closed Policy:** Sensitive operations (auth, upload, LLM generation) fail-closed if Redis becomes completely unreachable, preventing unmetered traffic storms.

---

## 3. Residual Risks & Operational Recommendations

1. **Novel Linguistic Jailbreaks:** Heuristic pattern matching catches known direct injection attempts. For nation-state threat profiles, integrate a fine-tuned binary classifier model (such as Llama Guard or NeMo Guardrails).
2. **Secret Rotation:** Regularly rotate `SECRET_KEY`, `MistralAI`, and `Qdrant_API_KEY` using an external secret manager (AWS Secrets Manager, HashiCorp Vault).
3. **TLS Enforcement:** In production deployments, ensure all traffic terminates behind a reverse proxy (Nginx, Traefik, Cloudflare) with HTTPS enforced and HSTS headers enabled.

