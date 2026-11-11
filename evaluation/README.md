# Hybrid RAG evaluation

## Current baseline harness

The runner is intentionally honest about what is and is not measured:

- It calls the authenticated FastAPI RAG endpoint.
- It records end-to-end HTTP response latency (not TTFT).
- It can capture pre-rerank hybrid candidates and post-rerank passages through a protected evaluation-only response field.
- It calculates retrieval metrics only when a case has reviewed chunk-level qrels.
- Judge API errors are recorded as errors; no default or heuristic score is substituted.

## Configuration

Keep secrets in your local/backend `.env`; never commit that file.

Backend environment:
- `EVALUATION_DEBUG_TOKEN`: long random secret. This enables raw retrieval trace output only when the caller sends the matching header.

Evaluation runner environment:
- `RAG_API_BASE_URL`: API root, default `http://127.0.0.1:8000`.
- `EVAL_API_TOKEN`: existing JWT, or use `EVAL_EMAIL` and `EVAL_PASSWORD` for login.
- `EVALUATION_DEBUG_TOKEN`: same value configured in the backend.
- `LOCAL_EVAL_API` or `LLM_API_KEY`: FreeLLMAPI key (or another OpenAI-compatible judge key).
- `LLM_BASE_URL`: provider chat-completions URL.
- `LLM_MODEL`: exact provider model identifier.
- `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`: optional tracing.

The API key, evaluation token, JWT and login credentials must never be hardcoded or committed.

## Run

From the repository root, with the API, Qdrant, Redis and local corpus available:

```bash
python eval.py
```

Results are written to `evaluation/results/baseline_results.json` (the results directory is created automatically).

## Dataset and qrels

The initial `datasets/baseline.jsonl` contains three seed cases migrated from the previous script. They are **not a statistically meaningful benchmark**. Their `qrels` are intentionally empty and marked `pending`: do not interpret retrieval metrics for them.

For each reviewed case, populate `qrels` as a mapping from the actual Qdrant chunk point ID to a graded relevance score, for example:

```json
{
  "id": "case_001",
  "query": "Question grounded in the corpus",
  "reference_answer": "Human-verified answer",
  "qrels": {
    "actual-qdrant-point-id": 3,
    "another-actual-point-id": 1
  },
  "qrels_status": "reviewed"
}
```

Use grades consistently (recommended: 3 = directly answers, 2 = strong support, 1 = useful context, 0/omitted = irrelevant). Do not invent chunk IDs or mark LLM-proposed qrels as reviewed. Expand the benchmark only from the actual indexed corpus and human-review the reference answers and relevance labels.

## Important measurement limitation

The current RAG API returns a complete non-streaming response. Therefore the runner records `response_latency_ms` and explicitly leaves `ttft_ms` null. True time-to-first-token requires streaming support and timestamping the first generated token; response completion time is not TTFT.
