"""Local chunk-based RAG service for grounded book Q&A."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency in some test environments
    class SentenceTransformer:  # type: ignore[override]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "sentence_transformers is required for RAG retrieval. "
                "Install dependencies from requirements.txt."
            )


@dataclass
class RagFilters:
    categories: Optional[Sequence[str]] = None
    learning_modes: Optional[Sequence[str]] = None
    min_similarity: float = -1.0


class RagService:
    """Chunk retrieval plus deterministic grounded answer synthesis."""

    def __init__(self, index_dir: Path) -> None:
        vectors_path = index_dir / "vectors.npy"
        metadata_path = index_dir / "metadata.json"
        model_info_path = index_dir / "model_info.json"
        if not vectors_path.exists() or not metadata_path.exists() or not model_info_path.exists():
            raise FileNotFoundError(
                f"Missing chunk index artifacts in {index_dir}. "
                "Expected vectors.npy, metadata.json, and model_info.json."
            )

        self.vectors = np.load(vectors_path)
        self.vectors = self._normalize_rows(self.vectors)
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata: List[Dict[str, Any]] = json.load(handle)
        if self.metadata and "chunk_text" not in self.metadata[0]:
            raise ValueError(
                f"{metadata_path} is not a chunk index metadata file. "
                "Build chunks index from semantic_chunks.jsonl."
            )

        with model_info_path.open("r", encoding="utf-8") as handle:
            model_info = json.load(handle)
        model_name = str(model_info.get("model_name", "") or "")
        if not model_name:
            raise ValueError("Invalid model_info.json: model_name missing.")
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _filter_indices(self, filters: RagFilters) -> np.ndarray:
        idxs = np.arange(len(self.metadata))
        if filters.categories:
            categories = set(filters.categories)
            idxs = np.array([idx for idx in idxs if self.metadata[idx].get("category") in categories], dtype=int)
        if filters.learning_modes:
            modes = set(filters.learning_modes)
            idxs = np.array([idx for idx in idxs if self.metadata[idx].get("learning_mode") in modes], dtype=int)
        return idxs

    @staticmethod
    def _compact_sentence(text: str, max_len: int = 240) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _source_label(item: Dict[str, Any]) -> str:
        source_type = str(item.get("source_type", "chunk"))
        source_index = int(item.get("source_index", 0) or 0)
        return f"{source_type}:{source_index}"

    def retrieve_chunks(
        self,
        query: str,
        filters: Optional[RagFilters] = None,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        clean_query = query.strip()
        if not clean_query:
            return []
        filters = filters or RagFilters()
        filtered_idxs = self._filter_indices(filters)
        if filtered_idxs.size == 0:
            return []

        query_vec = self.model.encode([clean_query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.vectors[filtered_idxs] @ query_vec
        scored: List[Dict[str, Any]] = []
        for local_idx, score in enumerate(sims):
            if float(score) < float(filters.min_similarity):
                continue
            row_idx = int(filtered_idxs[local_idx])
            item = dict(self.metadata[row_idx])
            item["similarity"] = round(float(score), 4)
            item["score"] = round(float(score), 4)
            scored.append(item)
        scored.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return scored[: max(1, int(top_k))]

    def answer_question(
        self,
        query: str,
        filters: Optional[RagFilters] = None,
        top_k: int = 8,
        max_citations: int = 6,
    ) -> Dict[str, Any]:
        chunks = self.retrieve_chunks(query=query, filters=filters, top_k=top_k)
        if not chunks:
            return {
                "answer": "I could not find grounded passages for that question in your indexed books.",
                "summary": "Try broadening the question or lowering similarity filters.",
                "follow_ups": [
                    "Ask for a broader concept first.",
                    "Filter by a specific category and try again.",
                ],
                "citations": [],
            }

        top = chunks[: max(1, min(3, len(chunks)))]
        answer_lines: List[str] = []
        for item in top:
            snippet = self._compact_sentence(str(item.get("chunk_text", "")), max_len=260)
            title = str(item.get("title", "Untitled"))
            answer_lines.append(f"- {title}: {snippet}")

        categories = sorted({str(item.get("category", "Other")) for item in top})
        summary = (
            f"Grounded from {len(chunks)} retrieved chunks across {len(categories)} categories: "
            + ", ".join(categories[:4])
        )

        citations: List[Dict[str, Any]] = []
        for item in chunks[: max(1, int(max_citations))]:
            citations.append(
                {
                    "title": str(item.get("title", "Untitled")),
                    "book_id": str(item.get("book_id", "")),
                    "absolute_path": str(item.get("absolute_path", "")),
                    "category": str(item.get("category", "Other")),
                    "learning_mode": str(item.get("learning_mode", "unknown")),
                    "source_label": self._source_label(item),
                    "start_char": int(item.get("start_char", 0) or 0),
                    "end_char": int(item.get("end_char", 0) or 0),
                    "similarity": float(item.get("similarity", 0.0) or 0.0),
                    "snippet": self._compact_sentence(str(item.get("chunk_text", "")), max_len=320),
                }
            )

        return {
            "answer": "Here is a grounded answer from your library:\n" + "\n".join(answer_lines),
            "summary": summary,
            "follow_ups": [
                "Compare these books by depth and practical coverage.",
                "Ask for a step-by-step study path from these sources.",
                "Filter by theory/practical and ask a narrower question.",
            ],
            "citations": citations,
        }
