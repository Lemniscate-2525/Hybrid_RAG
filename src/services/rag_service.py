"""
Unified Hybrid RAG Orchestration Service.
Executes the end-to-end request lifecycle:
Query Validation -> Prompt Guard -> Cache Check -> Dense & BM42 Hybrid Search
-> Context Isolation -> Cross-Encoder Rerank -> Resilient LLM Inference -> Citation Assembly -> Cache Write.
"""
import time
import logging
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status
from src.config import settings
from src.db.models import User
from src.api.schemas import QueryRequest, QueryResponse, CitationItem
from src.security.prompt_guard import prompt_guard, RiskLevel
from src.services.cache_service import cache_service
from src.services.embedding_service import embedding_service
from src.services.rerank_service import reranker_service
from src.services.qdrant_service import qdrant_service
from src.services.llm_client import llm_client

logger = logging.getLogger(__name__)

class RAGService:
    async def process_query(
        self,
        query_in: QueryRequest,
        user: User,
        correlation_id: Optional[str] = None
    ) -> QueryResponse:
        start_time = time.time()
        query = query_in.query
        filename = query_in.filename
        top_k = query_in.top_k

        # 1. Prompt Injection Defense on User Input
        security_check = prompt_guard.scan_query(query)
        if security_check.risk_level == RiskLevel.HIGH_RISK:
            logger.warning(f"Rejected high-risk query from user {user.id}: {security_check.reasons}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query rejected by security guardrails: {', '.join(security_check.reasons)}"
            )

        # 2. Check Cache
        cached_result = cache_service.get_cached_response(user.id, query, filename)
        if cached_result:
            duration_ms = int((time.time() - start_time) * 1000)
            cached_result["cache_hit"] = True
            cached_result["duration_ms"] = duration_ms
            cached_result["correlation_id"] = correlation_id
            return QueryResponse(**cached_result)

        # 3. Dense & Sparse BM42 Embedding Generation
        dense_query, sparse_query = embedding_service.embed_single_query(query)

        # 4. Multi-Tenant Hybrid Search in Qdrant (RRF Fusion)
        retrieved_chunks = qdrant_service.hybrid_search(
            user_id=user.id,
            dense_query=dense_query,
            sparse_query=sparse_query,
            filename=filename,
            limit=top_k
        )

        # 5. Abstention Guardrail: No evidence available
        if not retrieved_chunks:
            duration_ms = int((time.time() - start_time) * 1000)
            return QueryResponse(
                query=query,
                answer="I do not have sufficient information in your uploaded documents to answer this question. Please upload relevant PDF documents and try again.",
                citations=[],
                is_abstention=True,
                cache_hit=False,
                duration_ms=duration_ms,
                correlation_id=correlation_id
            )

        # 6. Context Isolation & Indirect Injection Sanitization
        _, safe_chunks = prompt_guard.sanitize_and_isolate_context(retrieved_chunks)

        # 7. Cross-Encoder Reranking
        reranked_chunks = reranker_service.rerank(query, safe_chunks, top_k=settings.TOP_K_RERANK)

        # Build isolated prompt with top reranked chunks
        context_blocks = []
        citations: List[CitationItem] = []

        for chunk in reranked_chunks:
            chunk_id = chunk.get("chunk_id", "")
            doc_id = chunk.get("doc_id", "")
            file_name = chunk.get("file_name", "unknown")
            page = chunk.get("page_label", "1")
            text = chunk.get("sanitized_text", chunk.get("text", ""))
            score = chunk.get("rerank_score", 0.0)

            context_blocks.append(
                f'<retrieved_evidence id="{chunk_id}" source="{file_name}" page="{page}">\n'
                f'{text}\n'
                f'</retrieved_evidence>'
            )

            snippet = (text[:180] + "...") if len(text) > 180 else text
            citations.append(
                CitationItem(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    file_name=file_name,
                    snippet=snippet,
                    relevance_score=round(score, 4)
                )
            )

        context_str = "\n\n".join(context_blocks)
        system_prompt = prompt_guard.construct_safe_system_prompt()
        user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question:\n{query}"

        # 8. Resilient LLM Completion
        answer = await llm_client.generate_response(system_prompt=system_prompt, user_prompt=user_prompt)

        duration_ms = int((time.time() - start_time) * 1000)

        response = QueryResponse(
            query=query,
            answer=answer,
            citations=citations,
            is_abstention=False,
            cache_hit=False,
            duration_ms=duration_ms,
            correlation_id=correlation_id
        )

        # 9. Store in Redis Cache
        cache_service.set_cached_response(
            user_id=user.id,
            query=query,
            response_data=response.model_dump(),
            filename=filename
        )

        return response

rag_service = RAGService()

