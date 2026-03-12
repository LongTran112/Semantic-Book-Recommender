"""Service helpers and payload mappers for Django RAG API."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict

from semantic_books.rag_config import LlamaCppConfig, OllamaConfig, RetrievalConfig
from semantic_books.rag_service import RagFilters, RagService


def resolve_index_dir() -> Path:
    return Path(os.getenv("RAG_INDEX_DIR", "output/semantic_index_chunks"))


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService(resolve_index_dir())


def reset_service_cache() -> None:
    get_rag_service.cache_clear()


def build_filters(payload: Dict[str, Any]) -> RagFilters:
    return RagFilters(
        categories=payload.get("categories"),
        learning_modes=payload.get("learning_modes"),
        min_similarity=float(payload.get("min_similarity", -1.0)),
    )


def build_retrieval(payload: Dict[str, Any], top_k: int) -> RetrievalConfig:
    name = payload.get("reranker_model_name")
    clean_name = str(name).strip() if name is not None else ""
    return RetrievalConfig(
        hybrid_enabled=bool(payload.get("hybrid_enabled", True)),
        dense_weight=float(payload.get("dense_weight", 0.7)),
        lexical_weight=float(payload.get("lexical_weight", 0.3)),
        candidate_pool_size=int(payload.get("candidate_pool_size", 48)),
        final_top_k=int(payload.get("final_top_k", top_k) or top_k),
        reranker_enabled=bool(payload.get("reranker_enabled", True)),
        reranker_model_name=(clean_name or None),
        reranker_top_n=int(payload.get("reranker_top_n", 32)),
    )


def build_llm(payload: Dict[str, Any]) -> LlamaCppConfig:
    return LlamaCppConfig(
        enabled=bool(payload.get("enabled", False)),
        model_path=str(payload.get("model_path", "")).strip(),
        n_ctx=int(payload.get("n_ctx", 2048)),
        max_tokens=int(payload.get("max_tokens", 420)),
        temperature=float(payload.get("temperature", 0.2)),
        top_p=float(payload.get("top_p", 0.9)),
        n_threads=int(payload.get("n_threads", 6)),
        n_gpu_layers=int(payload.get("n_gpu_layers", 0)),
        seed=int(payload.get("seed", 42)),
    )


def build_ollama(payload: Dict[str, Any]) -> OllamaConfig:
    return OllamaConfig(
        enabled=bool(payload.get("enabled", False)),
        base_url=str(payload.get("base_url", "http://127.0.0.1:11434")).strip(),
        model=str(payload.get("model", "granite3.3:8b")).strip(),
        temperature=float(payload.get("temperature", 0.2)),
        top_p=float(payload.get("top_p", 0.9)),
        num_ctx=int(payload.get("num_ctx", 8192)),
        timeout_sec=int(payload.get("timeout_sec", 180)),
    )
