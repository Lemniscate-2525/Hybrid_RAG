"""
Baseline RAG evaluation runner.

- Loads reviewed cases from evaluation/datasets/baseline.jsonl.
- Calls the authenticated RAG API once per case.
- Captures protected pre-rerank and post-rerank chunk traces when enabled.
- Computes chunk-level retrieval metrics only when qrels are reviewed.
- Uses real LLM judge calls; judge failures are recorded, never replaced by fake scores.
- Measures complete HTTP response latency. True TTFT requires streaming and is left null.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "evaluation" / "datasets" / "baseline.jsonl"
RESULTS_PATH = ROOT / "evaluation" / "results" / "baseline_results.json"
API_BASE = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.getenv("EVAL_API_TOKEN")
EVAL_EMAIL = os.getenv("EVAL_EMAIL")
EVAL_PASSWORD = os.getenv("EVAL_PASSWORD")
EVAL_DEBUG_TOKEN = os.getenv("EVALUATION_DEBUG_TOKEN") or os.getenv("EVAL_DEBUG_TOKEN")
TOP_K = int(os.getenv("EVAL_TOP_K", "10"))
METRIC_K = int(os.getenv("EVAL_METRIC_K", "5"))

LANGSMITH_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "hybrid-rag-evaluation"
if LANGSMITH_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_KEY
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = bool(LANGSMITH_KEY)
except Exception:
    LANGSMITH_AVAILABLE = False

    def traceable(name=None, run_type=None, **kwargs):
        def decorator(fn):
            return fn

from metrics import evaluate_retrieval


def load_dataset() -> list[Dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {DATASET_PATH}")
    cases = []
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            required = {"id", "query", "reference_answer", "qrels"}
            missing = required - case.keys()
            if missing:
                raise ValueError(f"Dataset line {line_number} missing fields: {sorted(missing)}")
            if not isinstance(case["qrels"], dict):
                raise ValueError(f"Dataset line {line_number}: qrels must be an object mapping chunk_id to grade")
            cases.append(case)
    if not cases:
        raise ValueError("Evaluation dataset is empty.")
    return cases


def get_access_token(client: httpx.Client) -> str:
    if API_TOKEN:
        return API_TOKEN
    if not EVAL_EMAIL or not EVAL_PASSWORD:
        raise RuntimeError(
            "Set EVAL_API_TOKEN, or set EVAL_EMAIL and EVAL_PASSWORD. "
            "Do not hardcode credentials in eval.py."
        )
    response = client.post(
        f"{API_BASE}/api/v1/auth/login",
        json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD},
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Login succeeded but the API did not return access_token.")
    return token


@traceable(name="hybrid_rag_evaluation_query", run_type="chain")
def execute_query(client: httpx.Client, token: str, case: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    if EVAL_DEBUG_TOKEN:
        headers["X-Evaluation-Token"] = EVAL_DEBUG_TOKEN
    started = time.perf_counter()
    response = client.post(
        f"{API_BASE}/api/v1/rag/query",
        json={
            "query": case["query"],
            "filename": case.get("filename"),
            "top_k": int(case.get("top_k", TOP_K)),
        },
        headers=headers,
    )
    response_latency_ms = round((time.perf_counter() - started) * 1000, 2)
    response.raise_for_status()
    payload = response.json()
    return {
        "answer": payload.get("answer", ""),
        "citations": payload.get("citations", []),
        "is_abstention": payload.get("is_abstention", False),
        "cache_hit": payload.get("cache_hit", False),
        "backend_duration_ms": payload.get("duration_ms"),
        "response_latency_ms": response_latency_ms,
        # The endpoint is non-streaming. This is deliberately NOT mislabeled TTFT.
        "ttft_ms": None,
        "evaluation_trace": payload.get("evaluation_trace"),
        "correlation_id": payload.get("correlation_id"),
    }


class LLMJudge:
    def __init__(self):
        self.api_key = (
            os.getenv("LOCAL_EVAL_API")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        self.api_url = os.getenv(
            "LLM_BASE_URL",
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.model = os.getenv("LLM_MODEL")
        if self.api_key:
            self.api_key = self.api_key.strip()

    def score(self, dimension: str, prompt: str) -> Dict[str, Any]:
        if not self.api_key or not self.model:
            raise RuntimeError("Judge credentials/model missing: configure LOCAL_EVAL_API or LLM_API_KEY, LLM_BASE_URL and LLM_MODEL.")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG evaluator. Return ONLY JSON with "
                        "score (number from 0 to 1) and reasoning (string). "
                        "Do not invent evidence. If evidence is insufficient, score accordingly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 350,
        }
        response = httpx.post(
            self.api_url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        choices = response.json().get("choices", [])
        if not choices:
            raise ValueError(f"{dimension}: judge API returned no choices.")
        content = choices[0].get("message", {}).get("content", "")
        content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content, flags=re.IGNORECASE)
        parsed = json.loads(content)
        if "score" not in parsed or "reasoning" not in parsed:
            raise ValueError(f"{dimension}: judge response must contain score and reasoning.")
        score = float(parsed["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{dimension}: judge score outside [0, 1].")
        return {"status": "ok", "score": round(score, 4), "reasoning": str(parsed["reasoning"])}


def run_judges(judge: LLMJudge, case: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    trace = output.get("evaluation_trace") or {}
    reranked = trace.get("reranked_chunks") or []
    context = "\n\n".join(item.get("text", "") for item in reranked)
    if not context:
        context = "\n\n".join(item.get("snippet", "") for item in output.get("citations", []))
    common = (
        f"Question:\n{case['query']}\n\n"
        f"Reference answer:\n{case['reference_answer']}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Generated answer:\n{output.get('answer', '')}\n\n"
    )
    prompts = {
        "faithfulness": common + "Score whether every substantive claim in the generated answer is supported by retrieved evidence. 1 means fully supported; 0 means unsupported or contradicted.",
        "correctness": common + "Compare the generated answer with the reference answer. Score factual correctness and coverage of the required facts.",
        "answer_relevance": common + "Score how directly and usefully the generated answer answers the question.",
        "citation_support": common + "The citations are: " + json.dumps(output.get("citations", []), ensure_ascii=False) + "\nScore whether cited chunks actually support the claims attributed to them. If there are no citations, score 0 unless the answer appropriately abstains.",
    }
    results = {}
    for dimension, prompt in prompts.items():
        try:
            results[dimension] = judge.score(dimension, prompt)
        except Exception as exc:
            # Failed judge calls are missing data, never fabricated numerical scores.
            results[dimension] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return results


def evaluate_case(case: Dict[str, Any], output: Dict[str, Any], judge: LLMJudge) -> Dict[str, Any]:
    trace = output.get("evaluation_trace") or {}
    retrieved = trace.get("retrieved_candidates") or []
    reranked = trace.get("reranked_chunks") or []
    qrels = {str(key): float(value) for key, value in case.get("qrels", {}).items()}

    retrieval = {"status": "pending_qrels", "candidate": None, "reranked": None}
    if qrels:
        retrieval = {
            "status": "scored",
            "candidate": evaluate_retrieval([str(x.get("chunk_id")) for x in retrieved], qrels, METRIC_K),
            "reranked": evaluate_retrieval([str(x.get("chunk_id")) for x in reranked], qrels, METRIC_K),
        }
    elif case.get("qrels_status") == "reviewed":
        retrieval["status"] = "invalid_reviewed_qrels_empty"

    return {
        "id": case["id"],
        "query": case["query"],
        "response_latency_ms": output["response_latency_ms"],
        "ttft_ms": None,
        "backend_duration_ms": output.get("backend_duration_ms"),
        "cache_hit": output.get("cache_hit"),
        "is_abstention": output.get("is_abstention"),
        "retrieval": retrieval,
        "judges": run_judges(judge, case, output),
        "answer": output.get("answer"),
        "citations": output.get("citations"),
        "retrieved_candidates": retrieved,
        "reranked_chunks": reranked,
        "correlation_id": output.get("correlation_id"),
    }


def main():
    cases = load_dataset()
    judge = LLMJudge()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with httpx.Client(timeout=90) as client:
        token = get_access_token(client)
        for index, case in enumerate(cases, start=1):
            print(f"\n[{index}/{len(cases)}] {case['id']}: {case['query']}")
            try:
                output = execute_query(client, token, case)
                result = evaluate_case(case, output, judge)
                print(f"  response latency: {result['response_latency_ms']} ms")
                print(f"  retrieval evaluation: {result['retrieval']['status']}")
                for name, value in result["judges"].items():
                    print(f"  {name}: {value.get('score', value.get('status'))}")
            except Exception as exc:
                result = {
                    "id": case["id"],
                    "query": case["query"],
                    "status": "query_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  ERROR: {result['error']}")
            results.append(result)

    scored_retrieval = [r for r in results if r.get("retrieval", {}).get("status") == "scored"]
    summary = {
        "created_at_unix": time.time(),
        "api_base": API_BASE,
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "case_count": len(cases),
        "successful_query_count": sum(r.get("status") != "query_error" for r in results),
        "retrieval_qrels_coverage": f"{len(scored_retrieval)}/{len(cases)}",
        "ttft_available": False,
        "ttft_note": "The current RAG endpoint is non-streaming; response latency is measured instead.",
        "runs": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results: {RESULTS_PATH}")
    print("Note: retrieval metrics remain pending until chunk-level qrels are human-reviewed.")


if __name__ == "__main__":
    main()
