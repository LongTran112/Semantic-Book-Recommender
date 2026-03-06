"""LangChain adapters for the local RAG service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from semantic_books.rag_config import LlamaCppConfig, RetrievalConfig
from semantic_books.rag_service import RagFilters, RagService

try:
    from langchain_community.llms import LlamaCpp as LangChainLlamaCpp
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import Runnable
except ImportError:  # pragma: no cover - optional dependency
    LangChainLlamaCpp = None  # type: ignore[assignment]
    CallbackManagerForRetrieverRun = Any  # type: ignore[misc, assignment]
    Document = Any  # type: ignore[misc, assignment]
    StrOutputParser = None  # type: ignore[assignment]
    ChatPromptTemplate = None  # type: ignore[assignment]
    BaseRetriever = object  # type: ignore[assignment]
    Runnable = Any  # type: ignore[misc, assignment]


class RagServiceRetriever(BaseRetriever):
    """LangChain retriever backed by ``RagService.retrieve_chunks``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rag_service: RagService
    filters: Optional[RagFilters] = None
    retrieval_config: Optional[RetrievalConfig] = None
    top_k: int = 8

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        chunks = self.rag_service.retrieve_chunks(
            query=query,
            filters=self.filters,
            top_k=int(self.top_k),
            retrieval_config=self.retrieval_config,
        )
        docs: List[Document] = []
        for chunk in chunks:
            metadata: Dict[str, Any] = dict(chunk)
            docs.append(
                Document(
                    page_content=str(chunk.get("chunk_text", "")),
                    metadata=metadata,
                )
            )
        return docs


def build_lc_answer_chain(llm_config: LlamaCppConfig) -> Optional[Runnable[Any, str]]:
    """Build an optional LCEL answer chain using llama.cpp."""

    if (
        LangChainLlamaCpp is None
        or ChatPromptTemplate is None
        or StrOutputParser is None
        or not llm_config.enabled
        or llm_config.resolved_model_path() is None
    ):
        return None

    llm = LangChainLlamaCpp(
        model_path=str(llm_config.resolved_model_path()),
        n_ctx=max(512, int(llm_config.n_ctx)),
        max_tokens=max(32, int(llm_config.max_tokens)),
        temperature=max(0.0, float(llm_config.temperature)),
        top_p=max(0.0, min(1.0, float(llm_config.top_p))),
        n_threads=max(1, int(llm_config.n_threads)),
        n_gpu_layers=int(llm_config.n_gpu_layers),
        seed=int(llm_config.seed),
        verbose=False,
    )
    prompt = ChatPromptTemplate.from_template(
        "You answer using only provided context.\n"
        "Cite claims with markers like [C1], [C2].\n"
        "Question: {query}\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    )
    return prompt | llm | StrOutputParser()


def build_context_from_documents(documents: List[Document], max_docs: int = 6) -> str:
    """Build compact context text with stable citation labels."""

    lines: List[str] = []
    for idx, doc in enumerate(documents[: max(1, int(max_docs))], start=1):
        meta = dict(getattr(doc, "metadata", {}) or {})
        title = str(meta.get("title", "Untitled"))
        category = str(meta.get("category", "Other"))
        source_type = str(meta.get("source_type", "chunk"))
        source_index = int(meta.get("source_index", meta.get("chunk_order", 0)) or 0)
        snippet = " ".join(str(getattr(doc, "page_content", "")).split())
        if len(snippet) > 320:
            snippet = snippet[:317].rstrip() + "..."
        lines.append(f"[C{idx}] {title} | {category} | {source_type}:{source_index} :: {snippet}")
    return "\n".join(lines)
