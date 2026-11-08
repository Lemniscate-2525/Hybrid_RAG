# Environment Variables Reference

| Variable | Type | Default | Required in Production | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Application Core** | | | | |
| `ENVIRONMENT` | string | `development` | Yes | Deployment environment: `development`, `testing`, `production`. |
| `DEBUG` | boolean | `false` | No | Enables detailed debug logging when `true`. |
| `HOST` | string | `0.0.0.0` | No | Network interface to bind the FastAPI server. |
| `PORT` | integer | `8000` | No | Port for the FastAPI server. |
| `SECRET_KEY` | string | *default dev key* | **YES** | Minimum 32-character key for signing JWT authentication tokens. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | integer | `1440` (24h) | No | Duration before JWT access tokens expire. |
| **Vector Database (Qdrant)** | | | | |
| `Qdrant_URL` | string | `None` (embedded) | Recommended | URL of the Qdrant cluster (e.g. `https://xyz.cloud.qdrant.io:6333`). |
| `Qdrant_API_KEY` | string | `None` | Recommended | API key for authenticating with Qdrant Cloud. |
| `Collection_Name` | string | `hybrid_rag_documents`| No | Target Qdrant collection name. |
| **LLM Inference (Mistral AI)** | | | | |
| `MistralAI` | string | `None` | **YES** | Mistral API key for LLM answer generation. |
| `MISTRAL_MODEL` | string | `open-mixtral-8x7b` | No | Mistral model identifier. |
| `LLM_TIMEOUT_SECONDS` | float | `30.0` | No | HTTP socket read timeout for LLM inference calls. |
| `LLM_MAX_RETRIES` | integer | `3` | No | Maximum retry attempts for transient 429/5xx errors. |
| **Redis (Cache & Rate Limiter)** | | | | |
| `REDIS_URL` | string | `redis://localhost:6379/0` | Recommended | Redis connection URI for distributed rate limiting and caching. |
| `REDIS_RATE_LIMIT_ENABLED` | boolean | `true` | No | Enables/disables Redis sliding-window rate limiting. |
| `REDIS_CACHE_ENABLED` | boolean | `true` | No | Enables/disables tenant-isolated query response caching. |
| `REDIS_CACHE_TTL_SECONDS` | integer | `3600` (1 hour) | No | Time-to-live for cached query answers. |
| **Rate Limit Thresholds** | | | | |
| `RATE_LIMIT_AUTH_PER_MINUTE` | integer | `10` | No | Max login/register attempts per minute per IP. |
| `RATE_LIMIT_QUERY_PER_MINUTE` | integer | `30` | No | Max hybrid RAG queries per minute per user. |
| `RATE_LIMIT_UPLOAD_PER_HOUR` | integer | `20` | No | Max PDF document uploads per hour per user. |
| **Storage & Guardrails** | | | | |
| `MAX_UPLOAD_SIZE_BYTES` | integer | `15728640` (15MB) | No | Maximum file size accepted for PDF uploads. |
| `MAX_DOCUMENT_PAGES` | integer | `150` | No | Maximum page count allowed in uploaded PDFs. |
| `MAX_QUERY_LENGTH` | integer | `500` | No | Maximum length of a single query string in characters. |

