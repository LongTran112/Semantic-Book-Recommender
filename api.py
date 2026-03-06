"""FastAPI endpoints for local RAG retrieval and answering."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from semantic_books.langchain_adapter import (
    RagServiceRetriever,
    build_context_from_documents,
    build_lc_answer_chain,
)
from semantic_books.rag_config import LlamaCppConfig, OllamaConfig, RetrievalConfig
from semantic_books.rag_service import RagFilters, RagService


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
    reranker_enabled: bool = False
    reranker_model_name: Optional[str] = None
    reranker_top_n: int = 24


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
    model: str = "qwen3.5:9b"
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 8192
    timeout_sec: int = 180


class RagRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1)
    max_citations: int = Field(default=6, ge=1)
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
                "absolute_path": str(meta.get("absolute_path", "")),
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


def _valid_generated_citations(text: str, citations: List[Dict[str, Any]]) -> bool:
    if not text.strip():
        return False
    known = {str(item.get("citation_id", "")) for item in citations if item.get("citation_id")}
    found = set(re.findall(r"\[(C\d+)\]", text))
    if not found:
        return False
    return found.issubset(known)


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
def rag_retrieve(request: RagRequest) -> RetrieveResponse:
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
    return RetrieveResponse(chunks=chunks)


@app.post("/rag/answer", response_model=AnswerResponse)
def rag_answer(request: RagRequest) -> AnswerResponse:
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
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc
    return AnswerResponse(**response)


@app.post("/rag/answer-lc", response_model=AnswerResponse)
def rag_answer_langchain(request: RagRequest) -> AnswerResponse:
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
            )
            fallback["fallback_reason"] = "LangChain retriever returned no documents."
            return AnswerResponse(**fallback)

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
            )
            fallback["fallback_reason"] = "LangChain chain unavailable; used canonical RAG answer."
            return AnswerResponse(**fallback)

        context = build_context_from_documents(docs, max_docs=int(request.max_citations))
        generated = str(chain.invoke({"query": request.query.strip(), "context": context}) or "").strip()
        if not _valid_generated_citations(generated, citations):
            fallback = rag_service.answer_question(
                query=request.query.strip(),
                filters=filters,
                top_k=int(request.top_k),
                max_citations=int(request.max_citations),
                retrieval_config=retrieval,
                llm_config=llm,
                ollama_config=ollama,
            )
            fallback["fallback_reason"] = "LangChain output missing valid citation markers."
            return AnswerResponse(**fallback)

        categories = sorted({str(item.get("category", "Other")) for item in citations})
        summary = (
            f"LangChain answer from {len(citations)} citations across {len(categories)} categories: "
            + ", ".join(categories[:4])
        )
        return AnswerResponse(
            answer=generated,
            summary=summary,
            follow_ups=rag_service._build_follow_ups(),
            citations=citations,
            generation_mode="langchain",
            fallback_reason="",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LangChain answer failed: {exc}") from exc
