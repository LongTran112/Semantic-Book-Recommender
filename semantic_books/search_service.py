"""Semantic retrieval and related-book recommendations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

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

    @staticmethod
    def _as_graph_node(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(item.get("book_id", "")),
            "title": str(item.get("title", "")),
            "category": str(item.get("category", "Other")),
            "learning_mode": str(item.get("learning_mode", "unknown")),
            "absolute_path": str(item.get("absolute_path", "")),
            "filename": str(item.get("filename", "")),
            "confidence": float(item.get("confidence", 0.0) or 0.0),
        }

    def _build_edges_for_indices(
        self,
        selected_indices: np.ndarray,
        neighbors_per_node: int,
        min_similarity: float,
    ) -> List[Dict[str, Any]]:
        if selected_indices.size == 0:
            return []

        local_vectors = self.vectors[selected_indices]
        edges: List[Dict[str, Any]] = []
        seen_pairs: Set[tuple[int, int]] = set()
        neighbor_cap = max(1, int(neighbors_per_node))
        threshold = float(min_similarity)

        for local_i in range(len(selected_indices)):
            sims = local_vectors @ local_vectors[local_i]
            order = np.argsort(-sims)
            taken = 0
            for local_j in order:
                if local_j == local_i:
                    continue
                score = float(sims[local_j])
                if score < threshold:
                    continue
                src = int(selected_indices[local_i])
                dst = int(selected_indices[local_j])
                low, high = (src, dst) if src < dst else (dst, src)
                if low == high or (low, high) in seen_pairs:
                    continue
                seen_pairs.add((low, high))
                edges.append(
                    {
                        "source": str(self.metadata[low].get("book_id", "")),
                        "target": str(self.metadata[high].get("book_id", "")),
                        "weight": round(score, 4),
                    }
                )
                taken += 1
                if taken >= neighbor_cap:
                    break
        return edges

    def build_whole_relationship_graph(
        self,
        filters: Optional[SearchFilters] = None,
        max_nodes: int = 250,
        min_similarity: float = 0.25,
        neighbors_per_node: int = 6,
    ) -> Dict[str, Any]:
        filters = filters or SearchFilters()
        filtered_idxs = self._filter_indices(filters)
        if filtered_idxs.size == 0:
            return {"nodes": [], "edges": []}

        cap = max(1, int(max_nodes))
        if filtered_idxs.size > cap:
            ranked = sorted(
                (int(idx) for idx in filtered_idxs),
                key=lambda i: float(self.metadata[i].get("confidence", 0.0) or 0.0),
                reverse=True,
            )[:cap]
            selected = np.array(ranked, dtype=int)
        else:
            selected = filtered_idxs.astype(int)

        nodes = [self._as_graph_node(self.metadata[int(idx)]) for idx in selected]
        edges = self._build_edges_for_indices(
            selected_indices=selected,
            neighbors_per_node=neighbors_per_node,
            min_similarity=min_similarity,
        )
        return {"nodes": nodes, "edges": edges}

    def build_focused_relationship_graph(
        self,
        query: Optional[str] = None,
        seed_book_id: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
        seed_top_k: int = 2,
        neighbor_k: int = 12,
        max_nodes: int = 120,
        min_similarity: float = 0.2,
        neighbors_per_node: int = 8,
    ) -> Dict[str, Any]:
        filters = filters or SearchFilters()
        filtered_idxs = self._filter_indices(filters)
        if filtered_idxs.size == 0:
            return {"nodes": [], "edges": [], "seed_ids": []}
        allowed = {int(i) for i in filtered_idxs}

        seed_indices: List[int] = []
        if seed_book_id:
            src_idx = self.book_id_to_idx.get(seed_book_id)
            if src_idx is not None and int(src_idx) in allowed:
                seed_indices.append(int(src_idx))

        if query:
            query_hits = self.search_books(query=query, filters=filters, top_k=max(1, int(seed_top_k)))
            for hit in query_hits:
                hit_id = str(hit.get("book_id", ""))
                hit_idx = self.book_id_to_idx.get(hit_id)
                if hit_idx is not None and int(hit_idx) in allowed:
                    seed_indices.append(int(hit_idx))

        if not seed_indices:
            fallback = sorted(
                allowed,
                key=lambda i: float(self.metadata[i].get("confidence", 0.0) or 0.0),
                reverse=True,
            )
            if fallback:
                seed_indices.append(int(fallback[0]))

        unique_seeds = list(dict.fromkeys(seed_indices))
        selected_set: Set[int] = set(unique_seeds)
        neighbor_cap = max(1, int(neighbor_k))
        for seed_idx in unique_seeds:
            sims = self.vectors @ self.vectors[int(seed_idx)]
            candidates = np.argsort(-sims)
            taken = 0
            for idx in candidates:
                idx_int = int(idx)
                if idx_int == int(seed_idx) or idx_int not in allowed:
                    continue
                if float(sims[idx_int]) < float(min_similarity):
                    continue
                selected_set.add(idx_int)
                taken += 1
                if taken >= neighbor_cap:
                    break

        selected_sorted = sorted(selected_set)
        if len(selected_sorted) > int(max_nodes):
            keep = set(unique_seeds)
            remaining = [i for i in selected_sorted if i not in keep]
            remaining_sorted = sorted(
                remaining,
                key=lambda i: float(self.metadata[i].get("confidence", 0.0) or 0.0),
                reverse=True,
            )
            room = max(0, int(max_nodes) - len(keep))
            selected_sorted = sorted(list(keep) + remaining_sorted[:room])
        selected = np.array(selected_sorted, dtype=int)

        nodes = [self._as_graph_node(self.metadata[int(idx)]) for idx in selected]
        edges = self._build_edges_for_indices(
            selected_indices=selected,
            neighbors_per_node=neighbors_per_node,
            min_similarity=min_similarity,
        )
        seed_ids = [str(self.metadata[i].get("book_id", "")) for i in unique_seeds]
        return {"nodes": nodes, "edges": edges, "seed_ids": seed_ids}

