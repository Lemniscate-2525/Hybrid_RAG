"""Hybrid RAG evaluation runner for LangSmith.

Run from the repository root:
    python -m evaluation.eval

This runner:
- Calls the local FastAPI RAG endpoint for each LangSmith example.
- Uses one local Ollama call to score all four LLM-as-a-judge dimensions.
- Caches that combined judge response so the four LangSmith evaluators do not
  make four separate calls to the local model for the same run.
- Computes IR metrics from positive qrels only, with normalized chunk IDs.
- Checks that the authenticated tenant actually owns indexed documents before
  starting an expensive benchmark run.
"""

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from langsmith import Client

try:
    from .metrics import precision_at_k, recall_at_k, ndcg_at_k, reciprocal_rank
except ImportError:
    from metrics import precision_at_k, recall_at_k, ndcg_at_k, reciprocal_rank


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = (
    os.getenv("LANGSMITH_PROJECT")
    or os.getenv("LANGCHAIN_PROJECT")
    or "hybrid-rag-production-eval"
)
DATASET_NAME = os.getenv("LANGSMITH_DATASET", "hybrid-rag-pdf-benchmark-v1")

RAG_API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
AUTH_ENDPOINT = os.getenv("RAG_AUTH_ENDPOINT", "/api/v1/auth/login")
QUERY_ENDPOINT = os.getenv("RAG_QUERY_ENDPOINT", "/api/v1/rag/query")
DOCUMENTS_ENDPOINT = os.getenv("RAG_DOCUMENTS_ENDPOINT", "/api/v1/documents/")
EVAL_EMAIL = os.getenv("EVAL_EMAIL")
EVAL_PASSWORD = os.getenv("EVAL_PASSWORD")
TOP_K = int(os.getenv("EVAL_TOP_K", "5"))
RAG_TIMEOUT_SECONDS = float(os.getenv("EVAL_REQUEST_TIMEOUT", "240"))
JUDGE_TIMEOUT_SECONDS = float(os.getenv("LOCAL_EVAL_TIMEOUT", "180"))
JUDGE_MAX_TOKENS = int(os.getenv("LOCAL_EVAL_MAX_TOKENS", "600"))
EVAL_DEBUG = os.getenv("EVAL_DEBUG", "0").strip().lower() in {"1", "true", "yes"}

if not LANGSMITH_API_KEY:
    raise EnvironmentError(
        "Set LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) in the repository-root .env"
    )
if not EVAL_EMAIL or not EVAL_PASSWORD:
    raise EnvironmentError(
        "Set EVAL_EMAIL and EVAL_PASSWORD in the repository-root .env"
    )
if TOP_K < 1:
    raise ValueError("EVAL_TOP_K must be at least 1")

# LangSmith tracing/project configuration.
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

client = Client(api_key=LANGSMITH_API_KEY)


def _normalize_id(value: Any) -> Optional[str]:
    """Normalize UUID-like chunk IDs for stable qrels/retrieval comparisons."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


# ---------------------------------------------------------------------------
# Live RAG API client
# ---------------------------------------------------------------------------
class PDFRAGClient:
    def __init__(self) -> None:
        timeout = httpx.Timeout(
            timeout=RAG_TIMEOUT_SECONDS,
            connect=15.0,
            read=RAG_TIMEOUT_SECONDS,
            write=30.0,
            pool=15.0,
        )
        self.http = httpx.Client(timeout=timeout)
        self.token: Optional[str] = None
        self.user_id: Optional[str] = None
        self.documents_verified = False
        self.document_count = 0
        self.indexed_chunk_count = 0

    def authenticate(self) -> None:
        response = self.http.post(
            f"{RAG_API_BASE_URL}{AUTH_ENDPOINT}",
            json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD},
        )
        response.raise_for_status()
        payload = response.json()
        self.token = payload.get("access_token") or payload.get("token")
        self.user_id = str(payload.get("user_id") or "") or None

        if not self.token:
            raise RuntimeError("Auth response has no access_token or token field")


    def _verify_tenant_documents(self) -> None:
        """Fail before evaluation if this login has no indexed tenant corpus."""
        if self.documents_verified:
            return

        response = self.http.get(
            f"{RAG_API_BASE_URL}{DOCUMENTS_ENDPOINT}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        response.raise_for_status()
        payload = response.json()
        documents = payload.get("documents") or []
        self.document_count = int(payload.get("total", len(documents)))
        self.indexed_chunk_count = sum(
            int(document.get("chunk_count") or 0)
            for document in documents
            if isinstance(document, dict)
        )

        if self.document_count == 0 or not documents:
            raise RuntimeError(
                "The EVAL_EMAIL account has no documents in its tenant. "
                "Your RAG API filters Qdrant retrieval by the authenticated user_id, "
                "so a newly registered account cannot see chunks indexed under the "
                "original corpus owner's user_id. Use the original account that owns "
                "the benchmark corpus and its qrels. Re-uploading the same PDFs under "
                "a new account creates different chunk IDs; the existing qrels will "
                "then need to be regenerated. Evaluation stopped before spending time "
                "on 111 queries."
            )

        if self.indexed_chunk_count == 0:
            names = [
                str(document.get("original_filename", "unknown"))
                for document in documents[:5]
                if isinstance(document, dict)
            ]
            raise RuntimeError(
                f"The evaluation account has {self.document_count} document record(s), "
                "but their total chunk_count is 0. These documents are not indexed for "
                "retrieval. Check ingestion/Qdrant state before running evaluation. "
                f"Documents: {names}"
            )

        self.documents_verified = True
        print(
            f"Authenticated tenant: user_id={self.user_id or '[not returned]'} | "
            f"documents={self.document_count} | indexed_chunks={self.indexed_chunk_count}"
        )

        if EVAL_DEBUG:
            names = [
                str(document.get("original_filename", "unknown"))
                for document in documents[:10]
                if isinstance(document, dict)
            ]
            print(f"Tenant documents (first 10): {names}")

    def _post_query(self, request_payload: Dict[str, Any]) -> httpx.Response:
        """POST a query; refresh auth once and honor Retry-After for 429s."""
        url = f"{RAG_API_BASE_URL}{QUERY_ENDPOINT}"
        response = self.http.post(
            url,
            json=request_payload,
            headers={"Authorization": f"Bearer {self.token}"},
        )

        if response.status_code == 401:
            self.token = None
            self.authenticate()
            response = self.http.post(
                url,
                json=request_payload,
                headers={"Authorization": f"Bearer {self.token}"},
            )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "5")
            try:
                wait_seconds = max(1.0, min(float(retry_after), 60.0))
            except ValueError:
                wait_seconds = 5.0

            print(
                f"RAG API rate limited; retrying once after {wait_seconds:.0f}s.",
                flush=True,
            )
            time.sleep(wait_seconds)
            response = self.http.post(
                url,
                json=request_payload,
                headers={"Authorization": f"Bearer {self.token}"},
            )

        response.raise_for_status()
        return response

    def query(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query_text = inputs.get("query")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("Every LangSmith example must contain inputs.query")

        if not self.token:
            self.authenticate()

        request_payload: Dict[str, Any] = {
            "query": query_text,
            "top_k": TOP_K,
        }

        if inputs.get("filename"):
            request_payload["filename"] = inputs["filename"]

        started = time.perf_counter()
        response = self._post_query(request_payload)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        data = response.json()
        citations = data.get("citations") or []

        if not isinstance(citations, list):
            raise ValueError(
                "RAG API returned citations in an unexpected format: "
                f"{type(citations).__name__}"
            )

        context_parts: List[str] = []
        retrieved_chunk_ids: List[str] = []

        for citation in citations:
            if not isinstance(citation, dict):
                continue

            snippet = citation.get("snippet")
            if isinstance(snippet, str) and snippet.strip():
                context_parts.append(snippet.strip())

            chunk_id = _normalize_id(citation.get("chunk_id"))
            if chunk_id:
                retrieved_chunk_ids.append(chunk_id)

        # Preserve citation order and remove duplicate chunk IDs.
        retrieved_chunk_ids = list(dict.fromkeys(retrieved_chunk_ids))

        return {
            "answer": str(data.get("answer") or ""),
            "context": "\n\n".join(context_parts),
            "citations": citations,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "request_latency_ms": latency_ms,
            "ttft_ms": None,  # The current endpoint is non-streaming.
            "is_abstention": bool(data.get("is_abstention", False)),
            "cache_hit": bool(data.get("cache_hit", False)),
        }


rag_client = PDFRAGClient()


def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """LangSmith target: execute one query against the local RAG application."""
    return rag_client.query(inputs)


# ---------------------------------------------------------------------------
# One local LLM call for all four judge dimensions
# ---------------------------------------------------------------------------
class LLMJudge:
    """Local Ollama judge; one bundled call is cached for each LangSmith run."""

    REQUIRED_KEYS = (
        "retrieval_relevance",
        "groundedness",
        "answer_correctness",
        "answer_relevance",
    )

    def __init__(self) -> None:
        self.api_key = os.getenv("LOCAL_EVAL_API_KEY", "ollama")
        self.model = os.getenv("LOCAL_EVAL_MODEL", "qwen3:1.7b")

        base_url = os.getenv(
            "LOCAL_EVAL_BASE_URL",
            "http://localhost:11434/v1",
        ).rstrip("/")

        self.api_url = (
            base_url
            if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions"
        )

        self.timeout = httpx.Timeout(
            timeout=JUDGE_TIMEOUT_SECONDS,
            connect=10.0,
            read=JUDGE_TIMEOUT_SECONDS,
            write=30.0,
            pool=10.0,
        )
        self._cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = threading.RLock()

        print(f"Evaluation Judge: {self.model}")
        print(f"Evaluation Endpoint: {self.api_url}")
        print("Judge calls per example: 1 bundled call for all four scores")

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        content = content.strip()
        content = re.sub(
            r"^\s*```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"\s*```\s*$", "", content)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")

            if start < 0 or end <= start:
                raise ValueError(
                    f"Judge did not return a JSON object: {content[:1200]}"
                )

            try:
                parsed = json.loads(content[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Judge returned malformed JSON near line {exc.lineno}, "
                    f"column {exc.colno}: {content[:1200]}"
                ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("Judge response must be a JSON object")

        return parsed

    def _request_bundle(
        self,
        query: str,
        context: str,
        answer: str,
        reference_answer: str,
    ) -> Dict[str, Dict[str, Any]]:
        prompt = f"""Evaluate this RAG result on four independent dimensions.

QUESTION:
{query or '[Missing question]'}

RETRIEVED EVIDENCE:
{context or '[No context retrieved]'}

GENERATED ANSWER:
{answer or '[Empty answer]'}

REFERENCE ANSWER:
{reference_answer or '[Missing reference answer]'}

SCORING RUBRICS (each score must be between 0.0 and 1.0):

1. retrieval_relevance:
   Is the retrieved evidence useful for answering the question?
   1.0 = Directly supports answering the question.
   0.5 = Partially relevant or contains useful but incomplete evidence.
   0.0 = Completely irrelevant, or no evidence was retrieved.

2. groundedness:
   Are the generated answer's factual claims supported by the retrieved evidence?
   Do not use outside knowledge.
   1.0 = All material factual claims are supported.
   0.5 = Some claims are supported, but others are unsupported.
   0.0 = The answer is entirely unsupported or contradicts the evidence.

3. answer_correctness:
   Does the generated answer match the reference answer in facts and coverage?
   Accept equivalent wording.
   1.0 = Factually correct and sufficiently complete.
   0.5 = Partially correct or materially incomplete.
   0.0 = Incorrect or fundamentally contradictory.

4. answer_relevance:
   Does the generated answer directly and usefully address the question?
   1.0 = Directly and completely addresses the question.
   0.5 = Partially addresses the question.
   0.0 = Irrelevant, evasive, or fails to answer.

IMPORTANT SCORING CONSISTENCY RULES:
- Evaluate each dimension independently.
- Determine the reasoning first, then assign a score consistent with that reasoning.
- A score of 0.0 means the dimension genuinely failed, not merely that the answer could be improved.
- If evidence is described as directly useful, retrieval_relevance MUST NOT be 0.0.
- If all factual claims are described as supported, groundedness MUST NOT be 0.0.
- If no evidence was retrieved, retrieval_relevance MUST be 0.0.
- If the answer contains no factual claims, assess groundedness accordingly rather than automatically assigning 0.0.
- Do not change scores merely to make all four scores similar.
- Before returning JSON, verify that every score agrees with its reasoning.

Return ONLY this JSON object, with one concise reasoning sentence per dimension:

{{
    "retrieval_relevance": {{
        "score": 0.0,
        "reasoning": "..."
    }},
    "groundedness": {{
        "score": 0.0,
        "reasoning": "..."
    }},
    "answer_correctness": {{
        "score": 0.0,
        "reasoning": "..."
    }},
    "answer_relevance": {{
        "score": 0.0,
        "reasoning": "..."
    }}
}}
"""

        response = httpx.post(
            self.api_url,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict, impartial RAG evaluator. "
                            "Return one valid JSON object only. "
                            "Do not use Markdown fences. "
                            "Do not reward unsupported claims."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": JUDGE_MAX_TOKENS,
                "reasoning_effort": "none",
                "response_format": {"type": "json_object"},
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"LLM Judge HTTP {response.status_code}: {response.text}"
            )

        try:
            message = response.json()["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Unexpected Ollama response structure: {response.text[:1500]}"
            ) from exc

        content = message.get("content") if isinstance(message, dict) else None

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Ollama returned an empty message.content field. "
                f"Message payload: {json.dumps(message, ensure_ascii=False)[:1500]}"
            )

        parsed = self._extract_json(content)
        results: Dict[str, Dict[str, Any]] = {}

        for key in self.REQUIRED_KEYS:
            item = parsed.get(key)

            if not isinstance(item, dict) or "score" not in item:
                raise ValueError(
                    f"Judge JSON is missing a valid '{key}' object: {parsed}"
                )

            try:
                score = float(item["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Judge returned a non-numeric score for {key}: {item}"
                ) from exc

            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Judge score for {key} is outside [0, 1]: {score}"
                )

            reasoning = str(item.get("reasoning", "")).strip()

            if not reasoning:
                raise ValueError(
                    f"Judge returned empty reasoning for {key}: {item}"
                )

            reasoning_lower = reasoning.lower()

            if key == "retrieval_relevance" and score == 0.0:
                positive_indicators = (
                    "directly useful",
                    "directly relevant",
                    "directly supports",
                    "highly relevant",
                    "useful for answering",
                )
                if any(term in reasoning_lower for term in positive_indicators):
                    raise ValueError(
                        "Scoring inconsistency: retrieval is described as "
                        "useful but receives a 0.0 score."
                    )

            if key == "groundedness" and score == 0.0:
                positive_indicators = (
                    "all claims are supported",
                    "fully supported",
                    "entirely supported",
                    "fully grounded",
                    "all factual claims",
                )
                if any(term in reasoning_lower for term in positive_indicators):
                    raise ValueError(
                        "Scoring inconsistency: answer is described as "
                        "fully supported but receives a 0.0 groundedness score."
                    )

            results[key] = {
                "score": round(score, 4),
                "reasoning": reasoning,
            }

        if EVAL_DEBUG:
            print(
                "BUNDLED JUDGE SCORES:",
                json.dumps(results, ensure_ascii=False, indent=2),
                flush=True,
            )

        return results


    def evaluate_bundle(
        self,
        cache_key: str,
        query: str,
        context: str,
        answer: str,
        reference_answer: str,
    ) -> Dict[str, Dict[str, Any]]:
        # Evaluators for one LangSmith run may be scheduled on different worker
        # threads. Lock around the cache check + request to guarantee one call.
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

            result = self._request_bundle(
                query,
                context,
                answer,
                reference_answer,
            )
            self._cache[cache_key] = result
            return result


judge = LLMJudge()


def _run_outputs(run: Any) -> Dict[str, Any]:
    return getattr(run, "outputs", None) or {}


def _example_inputs(example: Any) -> Dict[str, Any]:
    return getattr(example, "inputs", None) or {}


def _example_outputs(example: Any) -> Dict[str, Any]:
    return getattr(example, "outputs", None) or {}


def _judge_bundle(run: Any, example: Any) -> Dict[str, Dict[str, Any]]:
    outputs = _run_outputs(run)
    inputs = _example_inputs(example)
    expected_outputs = _example_outputs(example)
    metadata = getattr(example, "metadata", None) or {}

    reference_answer = (
        expected_outputs.get("reference_answer")
        or metadata.get("reference_answer")
        or ""
    )

    run_id = getattr(run, "id", None)
    cache_key = (
        str(run_id)
        if run_id is not None
        else f"run-object-{id(run)}"
    )

    return judge.evaluate_bundle(
        cache_key=cache_key,
        query=str(inputs.get("query") or ""),
        context=str(outputs.get("context") or ""),
        answer=str(outputs.get("answer") or ""),
        reference_answer=str(reference_answer),
    )


def retrieval_relevance(run: Any, example: Any) -> Dict[str, Any]:
    result = _judge_bundle(run, example)["retrieval_relevance"]
    return {
        "key": "retrieval_relevance",
        "score": result["score"],
        "comment": result["reasoning"],
    }


def groundedness(run: Any, example: Any) -> Dict[str, Any]:
    result = _judge_bundle(run, example)["groundedness"]
    return {
        "key": "groundedness",
        "score": result["score"],
        "comment": result["reasoning"],
    }


def answer_correctness(run: Any, example: Any) -> Dict[str, Any]:
    result = _judge_bundle(run, example)["answer_correctness"]
    return {
        "key": "answer_correctness",
        "score": result["score"],
        "comment": result["reasoning"],
    }


def answer_relevance(run: Any, example: Any) -> Dict[str, Any]:
    result = _judge_bundle(run, example)["answer_relevance"]
    return {
        "key": "answer_relevance",
        "score": result["score"],
        "comment": result["reasoning"],
    }


# ---------------------------------------------------------------------------
# IR metrics and abstention handling
# ---------------------------------------------------------------------------
def _coerce_qrels(raw_qrels: Any) -> Dict[str, float]:
    """Accept the expected {chunk_id: grade} map and JSON-encoded maps."""
    if isinstance(raw_qrels, str):
        try:
            raw_qrels = json.loads(raw_qrels)
        except json.JSONDecodeError:
            return {}

    if not isinstance(raw_qrels, dict):
        return {}

    qrels: Dict[str, float] = {}

    for chunk_id, raw_grade in raw_qrels.items():
        normalized_id = _normalize_id(chunk_id)

        if not normalized_id:
            continue

        try:
            grade = float(raw_grade)
        except (TypeError, ValueError):
            continue

        if grade > 0:
            qrels[normalized_id] = grade

    return qrels


def retrieval_metrics(run: Any, example: Any):
    outputs = _run_outputs(run)
    metadata = getattr(example, "metadata", None) or {}
    inputs = _example_inputs(example)

    raw_retrieved = outputs.get("retrieved_chunk_ids") or []
    retrieved = list(
        dict.fromkeys(
            normalized
            for normalized in (
                _normalize_id(value) for value in raw_retrieved
            )
            if normalized
        )
    )

    raw_qrels = metadata.get("qrels")
    if raw_qrels is None:
        raw_qrels = inputs.get("qrels")

    qrels = _coerce_qrels(raw_qrels)

    expected_behavior = str(
        metadata.get("expected_behavior")
        or inputs.get("expected_behavior")
        or "answer"
    ).strip().lower()

    if expected_behavior == "abstain":
        retrieval_passed = len(retrieved) == 0
        answer = str(outputs.get("answer", "")).strip().lower()

        abstention_phrases = (
            "doesn't include enough",
            "does not include enough",
            "not enough information",
            "insufficient information",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "not found in the document",
            "not provided in the document",
            "do not have sufficient information",
            "don't have sufficient information",
        )

        answer_passed = bool(outputs.get("is_abstention")) or any(
            phrase in answer for phrase in abstention_phrases
        )

        return [
            {
                "key": "abstention_retrieval_success",
                "score": 1.0 if retrieval_passed else 0.0,
                "comment": (
                    "No chunks retrieved for a verified no-evidence case."
                    if retrieval_passed
                    else f"Retrieved {len(retrieved)} chunk(s) for a verified no-evidence case."
                ),
            },
            {
                "key": "abstention_answer_success",
                "score": 1.0 if answer_passed else 0.0,
                "comment": (
                    "The answer clearly abstains due to insufficient document evidence."
                    if answer_passed
                    else "The answer did not clearly abstain."
                ),
            },
        ]

    if not qrels:
        return {
            "key": "qrels_missing_for_answerable_case",
            "score": 0.0,
            "comment": (
                "This example is answerable but has no positive qrels. "
                "Check that LangSmith example metadata contains the reviewed qrels map."
            ),
        }

    relevant_ids = list(qrels.keys())  # grade-0 qrels are deliberately excluded
    retrieved_set = set(retrieved)
    overlap_count = len(retrieved_set.intersection(qrels))
    overlap_score = 1.0 if overlap_count else 0.0

    overlap_comment = (
        f"Matched {overlap_count} of {len(qrels)} positive qrel chunk ID(s); "
        f"retrieved {len(retrieved)} chunk(s)."
    )

    if not overlap_count:
        overlap_comment += (
            " No retrieved ID matches the qrels. Check tenant user_id and corpus snapshot: "
            "this application generates chunk UUIDs using user_id, doc_id and chunk index."
        )

    return [
        {
            "key": "precision_at_k",
            "score": precision_at_k(retrieved, relevant_ids, TOP_K),
        },
        {
            "key": "recall_at_k",
            "score": recall_at_k(retrieved, relevant_ids, TOP_K),
        },
        {
            "key": "mrr",
            "score": reciprocal_rank(retrieved, relevant_ids),
        },
        {
            "key": "ndcg_at_k",
            "score": ndcg_at_k(retrieved, qrels, TOP_K),
        },
        {
            "key": "qrels_id_overlap",
            "score": overlap_score,
            "comment": overlap_comment,
        },
    ]


# ---------------------------------------------------------------------------
# Run the LangSmith experiment
# ---------------------------------------------------------------------------
def run_evaluation() -> None:
    print("=" * 72)
    print("HYBRID RAG | LANGSMITH BENCHMARK EVALUATION")
    print("=" * 72)
    print(f"Project       : {LANGSMITH_PROJECT}")
    print(f"Dataset       : {DATASET_NAME}")
    print(f"RAG API       : {RAG_API_BASE_URL}")
    print(f"Top K         : {TOP_K}")
    print(f"RAG timeout   : {RAG_TIMEOUT_SECONDS:.0f}s")
    print(f"Judge timeout : {JUDGE_TIMEOUT_SECONDS:.0f}s")

    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    print(f"Dataset confirmed: {dataset.name} ({dataset.id})")

    # Authenticate and validate tenant corpus before creating an experiment.
    # This catches the common case where EVAL_EMAIL was newly registered and
    # therefore has a different user_id from the one that owns the qrels corpus.
    print("Checking evaluation account and indexed document inventory...")
    rag_client.authenticate()

    print("Starting experiment with max_concurrency=1 (laptop-friendly)...")
    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            retrieval_relevance,
            groundedness,
            answer_correctness,
            answer_relevance,
            retrieval_metrics,
        ],
        experiment_prefix="hybrid-rag-benchmark-v1",
        max_concurrency=1,
        metadata={
            "benchmark_version": "v1",
            "evaluation_type": "hybrid_rag_pdf_qa",
            "top_k": TOP_K,
            "judge_model": judge.model,
            "judge_calls_per_example": 1,
        },
    )

    print("\nExperiment submitted successfully.")
    print(results)


if __name__ == "__main__":
    run_evaluation()