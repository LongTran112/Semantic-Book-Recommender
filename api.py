"""FastAPI endpoints for local RAG retrieval and answering."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
import threading
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from semantic_books.langchain_adapter import (
    RagServiceRetriever,
    build_context_from_documents,
    build_lc_answer_chain,
)
from semantic_books.rag_config import LlamaCppConfig, OllamaConfig, RetrievalConfig
from semantic_books.rag_service import RagFilters, RagService


_RATE_LIMIT_STATE: Dict[str, Dict[str, float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()


def _resolve_index_dir() -> Path:
    return Path(os.getenv("RAG_INDEX_DIR", "output/semantic_index_chunks"))


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService(_resolve_index_dir())


class RagFiltersPayload(BaseModel):
    categories: Optional[List[str]] = None
    learning_modes: Optional[List[str]] = None
    min_similarity: float = -1.0


class RetrievalConfigPayload(BaseModel):
    hybrid_enabled: bool = True
    dense_weight: float = 0.7
    lexical_weight: float = 0.3
    candidate_pool_size: int = 48
    final_top_k: int = 8
    reranker_enabled: bool = True
    reranker_model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 32


class LlamaCppConfigPayload(BaseModel):
    enabled: bool = False
    model_path: str = ""
    n_ctx: int = 2048
    max_tokens: int = 420
    temperature: float = 0.2
    top_p: float = 0.9
    n_threads: int = 6
    n_gpu_layers: int = 0
    seed: int = 42


class OllamaConfigPayload(BaseModel):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434"
    model: str = "deepseek-r1:14b"
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 8192
    timeout_sec: int = 180


class RagRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1)
    max_citations: int = Field(default=6, ge=1)
    allow_fallback: bool = True
    filters: RagFiltersPayload = Field(default_factory=RagFiltersPayload)
    retrieval: RetrievalConfigPayload = Field(default_factory=RetrievalConfigPayload)
    llm: LlamaCppConfigPayload = Field(default_factory=LlamaCppConfigPayload)
    ollama: OllamaConfigPayload = Field(default_factory=OllamaConfigPayload)


class HealthResponse(BaseModel):
    status: str
    index_dir: str
    chunks_indexed: int


class RetrieveResponse(BaseModel):
    chunks: List[Dict[str, Any]]


class AnswerResponse(BaseModel):
    answer: str
    summary: str
    follow_ups: List[str]
    citations: List[Dict[str, Any]]
    generation_mode: str
    fallback_reason: str
    metrics: Optional[Dict[str, Any]] = None


def _required_api_key() -> str:
    value = str(os.getenv("RAG_API_KEY", "") or "").strip()
    return value


def _rate_limit_window_sec() -> int:
    raw = str(os.getenv("RAG_RATE_LIMIT_WINDOW_SEC", "60") or "60").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


def _rate_limit_max_requests() -> int:
    raw = str(os.getenv("RAG_RATE_LIMIT_MAX_REQUESTS", "30") or "30").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 30


def _redact_path_value(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(payload)
    if "absolute_path" in clean:
        clean["absolute_path"] = ""
    return clean


def _redact_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_redact_path_value(item) for item in chunks]


def _redact_answer_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(payload)
    citations = clean.get("citations", [])
    if isinstance(citations, list):
        clean["citations"] = [_redact_path_value(dict(item)) for item in citations if isinstance(item, dict)]
    return clean


def _identity_from_request(request: Request, api_key: str) -> str:
    if api_key:
        return f"key:{api_key}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def require_guardrails(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    configured_key = _required_api_key()
    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="RAG_API_KEY is not configured. Set it before calling protected endpoints.",
        )
    provided_key = str(x_api_key or "").strip()
    if not provided_key or provided_key != configured_key:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing X-API-Key.")
    identity = _identity_from_request(request=request, api_key=provided_key)
    _apply_rate_limit(identity)
    return identity


def _apply_rate_limit(identity: str) -> None:
    import time

    now = time.time()
    window_sec = _rate_limit_window_sec()
    max_requests = _rate_limit_max_requests()
    with _RATE_LIMIT_LOCK:
        state = _RATE_LIMIT_STATE.get(identity)
        if state is None or (now - float(state.get("window_start", 0.0))) >= window_sec:
            _RATE_LIMIT_STATE[identity] = {"window_start": now, "count": 1.0}
            return
        count = float(state.get("count", 0.0)) + 1.0
        state["count"] = count
        if count > float(max_requests):
            retry_after = max(1, int(window_sec - (now - float(state.get("window_start", now)))))
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry in {retry_after}s.",
            )


def _reset_rate_limit_state() -> None:
    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_STATE.clear()


def _compact_sentence(text: str, max_len: int = 320) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def _source_label(item: Dict[str, Any]) -> str:
    source_type = str(item.get("source_type", "chunk"))
    source_index = int(item.get("source_index", item.get("chunk_order", 0)) or 0)
    return f"{source_type}:{source_index}"


def _build_citations_from_docs(documents: List[Any], max_citations: int) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents[: max(1, int(max_citations))], start=1):
        meta: Dict[str, Any] = dict(getattr(doc, "metadata", {}) or {})
        citations.append(
            {
                "citation_id": f"C{idx}",
                "title": str(meta.get("title", "Untitled")),
                "book_id": str(meta.get("book_id", "")),
                "absolute_path": "",
                "category": str(meta.get("category", "Other")),
                "learning_mode": str(meta.get("learning_mode", "unknown")),
                "source_label": _source_label(meta),
                "start_char": int(meta.get("start_char", 0) or 0),
                "end_char": int(meta.get("end_char", 0) or 0),
                "chunk_order": int(meta.get("chunk_order", meta.get("source_index", 0)) or 0),
                "chunk_len": int(meta.get("chunk_len", len(str(getattr(doc, "page_content", "")))) or 0),
                "section_label": str(meta.get("section_label", meta.get("source_type", "")) or ""),
                "similarity": float(meta.get("similarity", 0.0) or 0.0),
                "snippet": _compact_sentence(str(getattr(doc, "page_content", ""))),
            }
        )
    return citations


def _build_filters(payload: RagFiltersPayload) -> RagFilters:
    return RagFilters(
        categories=payload.categories,
        learning_modes=payload.learning_modes,
        min_similarity=float(payload.min_similarity),
    )


def _build_retrieval_config(payload: RetrievalConfigPayload, top_k: int) -> RetrievalConfig:
    return RetrievalConfig(
        hybrid_enabled=bool(payload.hybrid_enabled),
        dense_weight=float(payload.dense_weight),
        lexical_weight=float(payload.lexical_weight),
        candidate_pool_size=int(payload.candidate_pool_size),
        final_top_k=int(payload.final_top_k or top_k),
        reranker_enabled=bool(payload.reranker_enabled),
        reranker_model_name=(str(payload.reranker_model_name).strip() or None),
        reranker_top_n=int(payload.reranker_top_n),
    )


def _build_llm_config(payload: LlamaCppConfigPayload) -> LlamaCppConfig:
    return LlamaCppConfig(
        enabled=bool(payload.enabled),
        model_path=str(payload.model_path).strip(),
        n_ctx=int(payload.n_ctx),
        max_tokens=int(payload.max_tokens),
        temperature=float(payload.temperature),
        top_p=float(payload.top_p),
        n_threads=int(payload.n_threads),
        n_gpu_layers=int(payload.n_gpu_layers),
        seed=int(payload.seed),
    )


def _build_ollama_config(payload: OllamaConfigPayload) -> OllamaConfig:
    return OllamaConfig(
        enabled=bool(payload.enabled),
        base_url=str(payload.base_url).strip(),
        model=str(payload.model).strip(),
        temperature=float(payload.temperature),
        top_p=float(payload.top_p),
        num_ctx=int(payload.num_ctx),
        timeout_sec=int(payload.timeout_sec),
    )


app = FastAPI(title="EBooksSorter RAG API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        rag_service = get_rag_service()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {exc}") from exc
    return HealthResponse(
        status="ok",
        index_dir=str(_resolve_index_dir()),
        chunks_indexed=len(rag_service.metadata),
    )


@app.post("/rag/retrieve", response_model=RetrieveResponse)
def rag_retrieve(request: RagRequest, _identity: str = Depends(require_guardrails)) -> RetrieveResponse:
    _ = _identity
    try:
        rag_service = get_rag_service()
        filters = _build_filters(request.filters)
        retrieval = _build_retrieval_config(request.retrieval, request.top_k)
        chunks = rag_service.retrieve_chunks(
            query=request.query.strip(),
            filters=filters,
            top_k=int(request.top_k),
            retrieval_config=retrieval,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc
    return RetrieveResponse(chunks=_redact_chunks(chunks))


@app.post("/rag/answer", response_model=AnswerResponse)
def rag_answer(request: RagRequest, _identity: str = Depends(require_guardrails)) -> AnswerResponse:
    _ = _identity
    try:
        rag_service = get_rag_service()
        filters = _build_filters(request.filters)
        retrieval = _build_retrieval_config(request.retrieval, request.top_k)
        llm = _build_llm_config(request.llm)
        ollama = _build_ollama_config(request.ollama)
        response = rag_service.answer_question(
            query=request.query.strip(),
            filters=filters,
            top_k=int(request.top_k),
            max_citations=int(request.max_citations),
            retrieval_config=retrieval,
            llm_config=llm,
            ollama_config=ollama,
            allow_fallback=bool(request.allow_fallback),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc
    return AnswerResponse(**_redact_answer_payload(response))


@app.post("/rag/answer-lc", response_model=AnswerResponse)
def rag_answer_langchain(request: RagRequest, _identity: str = Depends(require_guardrails)) -> AnswerResponse:
    _ = _identity
    try:
        rag_service = get_rag_service()
        filters = _build_filters(request.filters)
        retrieval = _build_retrieval_config(request.retrieval, request.top_k)
        llm = _build_llm_config(request.llm)
        ollama = _build_ollama_config(request.ollama)

        retriever = RagServiceRetriever(
            rag_service=rag_service,
            filters=filters,
            retrieval_config=retrieval,
            top_k=int(request.top_k),
        )
        docs = retriever.invoke(request.query.strip())
        if not docs:
            fallback = rag_service.answer_question(
                query=request.query.strip(),
                filters=filters,
                top_k=int(request.top_k),
                max_citations=int(request.max_citations),
                retrieval_config=retrieval,
                llm_config=llm,
                ollama_config=ollama,
                allow_fallback=bool(request.allow_fallback),
            )
            fallback["fallback_reason"] = "LangChain retriever returned no documents."
            return AnswerResponse(**_redact_answer_payload(fallback))

        citations = _build_citations_from_docs(docs, max_citations=int(request.max_citations))
        chain = build_lc_answer_chain(llm)
        if chain is None:
            fallback = rag_service.answer_question(
                query=request.query.strip(),
                filters=filters,
                top_k=int(request.top_k),
                max_citations=int(request.max_citations),
                retrieval_config=retrieval,
                llm_config=llm,
                ollama_config=ollama,
                allow_fallback=bool(request.allow_fallback),
            )
            fallback["fallback_reason"] = "LangChain chain unavailable; used canonical RAG answer."
            return AnswerResponse(**_redact_answer_payload(fallback))

        context = build_context_from_documents(docs, max_docs=int(request.max_citations))
        generated = str(chain.invoke({"query": request.query.strip(), "context": context}) or "").strip()
        known = {str(item.get("citation_id", "")) for item in citations if item.get("citation_id")}
        if not rag_service._validate_generated_answer(generated, known):
            fallback = rag_service.answer_question(
                query=request.query.strip(),
                filters=filters,
                top_k=int(request.top_k),
                max_citations=int(request.max_citations),
                retrieval_config=retrieval,
                llm_config=llm,
                ollama_config=ollama,
                allow_fallback=bool(request.allow_fallback),
            )
            fallback["fallback_reason"] = "LangChain output missing valid citation markers."
            return AnswerResponse(**_redact_answer_payload(fallback))

        categories = sorted({str(item.get("category", "Other")) for item in citations})
        summary = (
            f"LangChain answer from {len(citations)} citations across {len(categories)} categories: "
            + ", ".join(categories[:4])
        )
        return AnswerResponse(
            answer=generated,
            summary=summary,
            follow_ups=rag_service._build_follow_ups(query=request.query.strip(), chunks=citations),
            citations=[_redact_path_value(item) for item in citations],
            generation_mode="langchain",
            fallback_reason="",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LangChain answer failed: {exc}") from exc


@app.post("/rag/answer-stream")
def rag_answer_stream(
    request: Request,
    payload: RagRequest,
    _identity: str = Depends(require_guardrails),
) -> StreamingResponse:
    _ = _identity
    rag_service = get_rag_service()
    filters = _build_filters(payload.filters)
    retrieval = _build_retrieval_config(payload.retrieval, payload.top_k)
    llm = _build_llm_config(payload.llm)
    ollama = _build_ollama_config(payload.ollama)
    events: "Queue[Optional[Dict[str, Any]]]" = Queue()
    cancel_event = threading.Event()

    def _on_token(token: str) -> None:
        if cancel_event.is_set():
            return
        events.put({"type": "token", "token": token})

    def _worker() -> None:
        try:
            response = rag_service.answer_question(
                query=payload.query.strip(),
                filters=filters,
                top_k=int(payload.top_k),
                max_citations=int(payload.max_citations),
                retrieval_config=retrieval,
                llm_config=llm,
                ollama_config=ollama,
                on_token=_on_token if bool(ollama.enabled or llm.enabled) else None,
                should_cancel=cancel_event.is_set,
                allow_fallback=bool(payload.allow_fallback),
            )
            if not cancel_event.is_set():
                events.put({"type": "final", "response": _redact_answer_payload(response)})
        except Exception as exc:
            if not cancel_event.is_set():
                events.put({"type": "error", "error": str(exc)})
        finally:
            events.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    async def _event_generator():
        while True:
            if await request.is_disconnected():
                cancel_event.set()
                break
            try:
                item = events.get(timeout=0.25)
            except Empty:
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")
