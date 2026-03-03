"""Semantic retrieval and related-book recommendations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency in test/runtime environments
    class SentenceTransformer:  # type: ignore[override]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "sentence_transformers is required for semantic search. "
                "Install dependencies from requirements.txt."
            )


@dataclass
class SearchFilters:
    categories: Optional[Sequence[str]] = None
    learning_modes: Optional[Sequence[str]] = None
    min_similarity: float = -1.0


class SemanticSearchService:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        vectors_path = index_dir / "vectors.npy"
        metadata_path = index_dir / "metadata.json"
        model_info_path = index_dir / "model_info.json"

        if not vectors_path.exists() or not metadata_path.exists() or not model_info_path.exists():
            raise FileNotFoundError(
                f"Missing semantic index artifacts in {index_dir}. "
                "Expected vectors.npy, metadata.json, and model_info.json."
            )

        self.vectors = np.load(vectors_path)
        self.vectors = self._normalize_rows(self.vectors)

        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata: List[Dict[str, Any]] = json.load(handle)

        with model_info_path.open("r", encoding="utf-8") as handle:
            model_info = json.load(handle)
        model_name = model_info.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Invalid model_info.json: model_name missing.")

        self.model = SentenceTransformer(model_name)
        self.book_id_to_idx = {
            item.get("book_id"): idx for idx, item in enumerate(self.metadata) if item.get("book_id")
        }

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _filter_indices(self, filters: SearchFilters) -> np.ndarray:
        idxs = np.arange(len(self.metadata))
        if filters.categories:
            categories = set(filters.categories)
            idxs = np.array([idx for idx in idxs if self.metadata[idx].get("category") in categories], dtype=int)
        if filters.learning_modes:
            modes = set(filters.learning_modes)
            idxs = np.array([idx for idx in idxs if self.metadata[idx].get("learning_mode") in modes], dtype=int)
        return idxs

    def search_books(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        filters = filters or SearchFilters()
        filtered_idxs = self._filter_indices(filters)
        if filtered_idxs.size == 0:
            return []

        query_vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.vectors[filtered_idxs] @ query_vector

        scored: List[Dict[str, Any]] = []
        for local_idx, score in enumerate(sims):
            if score < filters.min_similarity:
                continue
            row_idx = int(filtered_idxs[local_idx])
            item = dict(self.metadata[row_idx])
            confidence = float(item.get("confidence", 0.0) or 0.0)
            final_score = float(score) + 0.05 * confidence
            item["similarity"] = round(float(score), 4)
            item["score"] = round(final_score, 4)
            scored.append(item)

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, top_k)]

    def get_book(self, book_id: str) -> Optional[Dict[str, Any]]:
        idx = self.book_id_to_idx.get(book_id)
        if idx is None:
            return None
        return dict(self.metadata[idx])

    def recommend_related(
        self,
        book_id: str,
        top_k: int = 10,
        same_category_boost: bool = True,
    ) -> List[Dict[str, Any]]:
        src_idx = self.book_id_to_idx.get(book_id)
        if src_idx is None:
            return []

        src_vec = self.vectors[src_idx]
        sims = self.vectors @ src_vec
        src_category = self.metadata[src_idx].get("category")

        recs: List[Dict[str, Any]] = []
        for idx, score in enumerate(sims):
            if idx == src_idx:
                continue
            item = dict(self.metadata[idx])
            boosted = float(score)
            if same_category_boost and src_category and item.get("category") == src_category:
                boosted += 0.03
            item["similarity"] = round(float(score), 4)
            item["score"] = round(boosted, 4)
            recs.append(item)

        recs.sort(key=lambda item: item["score"], reverse=True)
        return recs[: max(1, top_k)]

