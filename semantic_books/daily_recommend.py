"""Daily 5-book recommendation engine with history and novelty."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from semantic_books.search_service import SemanticSearchService


@dataclass
class DailyRecommendationWeights:
    similarity: float = 0.55
    freshness: float = 0.2
    novelty: float = 0.15
    confidence: float = 0.1
    diversity_penalty: float = 0.1
    explore_bonus: float = 0.2


class DailyBookRecommender:
    def __init__(
        self,
        service: SemanticSearchService,
        reading_list_path: Path,
        history_path: Path,
        weights: Optional[DailyRecommendationWeights] = None,
        novelty_window_days: int = 14,
        freshness_window_days: int = 30,
    ) -> None:
        self.service = service
        self.reading_list_path = reading_list_path
        self.history_path = history_path
        self.weights = weights or DailyRecommendationWeights()
        self.novelty_window_days = novelty_window_days
        self.freshness_window_days = freshness_window_days
        self._metadata_by_id = {
            str(item.get("book_id")): item for item in self.service.metadata if item.get("book_id")
        }
        self._category_freq = self._build_category_frequency()

    def _build_category_frequency(self) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for item in self.service.metadata:
            category = str(item.get("category", "Other"))
            freq[category] = freq.get(category, 0) + 1
        return freq

    @staticmethod
    def _to_date(value: Optional[date]) -> date:
        if value is not None:
            return value
        return datetime.now().astimezone().date()

    @staticmethod
    def _date_key(value: date) -> str:
        return value.isoformat()

    def _load_reading_ids(self) -> set[str]:
        if not self.reading_list_path.exists():
            return set()
        try:
            with self.reading_list_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return set()
        if isinstance(payload, dict):
            return {str(book_id) for book_id in payload.keys()}
        return set()

    def _load_history(self) -> Dict[str, Any]:
        if not self.history_path.exists():
            return {"days": {}}
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {"days": {}}
        if not isinstance(payload, dict):
            return {"days": {}}
        if not isinstance(payload.get("days"), dict):
            payload["days"] = {}
        return payload

    def _save_history(self, payload: Dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _recent_history_ids(self, payload: Dict[str, Any], target_date: date) -> Tuple[set[str], set[str]]:
        days = payload.get("days", {})
        all_seen: set[str] = set()
        recent_seen: set[str] = set()
        for day_str, record in days.items():
            if not isinstance(record, dict):
                continue
            book_ids = record.get("book_ids", [])
            if not isinstance(book_ids, list):
                continue
            ids = {str(book_id) for book_id in book_ids}
            try:
                rec_date = datetime.fromisoformat(day_str).date()
            except ValueError:
                continue
            if rec_date < target_date:
                all_seen.update(ids)
            if rec_date >= (target_date - timedelta(days=self.novelty_window_days)) and rec_date < target_date:
                recent_seen.update(ids)
        return recent_seen, all_seen

    def _reading_indices(self, reading_ids: set[str]) -> np.ndarray:
        idxs: List[int] = []
        for book_id in reading_ids:
            idx = self.service.book_id_to_idx.get(book_id)
            if idx is not None:
                idxs.append(idx)
        if not idxs:
            return np.array([], dtype=int)
        return np.array(idxs, dtype=int)

    def _freshness_score(self, absolute_path: str, target_date: date) -> float:
        path = Path(absolute_path)
        if not path.exists():
            return 0.0
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
        except Exception:
            return 0.0
        days_old = max(0, (target_date - mtime).days)
        if days_old >= self.freshness_window_days:
            return 0.0
        return 1.0 - (days_old / float(self.freshness_window_days))

    def _similarity_to_current_reading(self, idx: int, reading_idxs: np.ndarray) -> float:
        if reading_idxs.size == 0:
            return 0.0
        sims = self.service.vectors[reading_idxs] @ self.service.vectors[idx]
        if sims.size == 0:
            return 0.0
        return float(np.max(sims))

    def _novelty_score(self, book_id: str, recent_seen: set[str], all_seen: set[str]) -> float:
        if book_id in recent_seen:
            return 0.0
        if book_id in all_seen:
            return 0.35
        return 1.0

    def _tie_break(self, target_date: date, book_id: str) -> float:
        seed = f"{target_date.isoformat()}::{book_id}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / float(0xFFFFFFFF)

    def _explore_boost(self, category: str, selected_categories: Sequence[str]) -> float:
        freq = self._category_freq.get(category, 1)
        rarity_boost = 1.0 / float(freq)
        unseen_category_bonus = self.weights.explore_bonus if category not in selected_categories else 0.0
        return rarity_boost + unseen_category_bonus

    def _score_candidates(
        self,
        target_date: date,
        reading_ids: set[str],
        recent_seen: set[str],
        all_seen: set[str],
    ) -> List[Dict[str, Any]]:
        reading_idxs = self._reading_indices(reading_ids)
        scored: List[Dict[str, Any]] = []
        for idx, item in enumerate(self.service.metadata):
            book_id = str(item.get("book_id", ""))
            if not book_id or book_id in reading_ids:
                continue

            confidence = float(item.get("confidence", 0.0) or 0.0)
            similarity = self._similarity_to_current_reading(idx, reading_idxs)
            freshness = self._freshness_score(str(item.get("absolute_path", "")), target_date)
            novelty = self._novelty_score(book_id, recent_seen, all_seen)
            tie_break = self._tie_break(target_date, book_id) * 1e-6
            base_score = (
                self.weights.similarity * similarity
                + self.weights.freshness * freshness
                + self.weights.novelty * novelty
                + self.weights.confidence * confidence
                + tie_break
            )
            scored.append(
                {
                    "idx": idx,
                    "book_id": book_id,
                    "category": str(item.get("category", "Other")),
                    "base_score": float(base_score),
                    "reasons": {
                        "similarity": round(similarity, 4),
                        "freshness": round(freshness, 4),
                        "novelty": round(novelty, 4),
                        "confidence": round(confidence, 4),
                    },
                }
            )
        scored.sort(key=lambda row: row["base_score"], reverse=True)
        return scored

    def _select_balanced(self, candidates: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        target_count = max(1, count)
        exploit_count = max(1, target_count - 2)
        selected: List[Dict[str, Any]] = []
        selected_ids: set[str] = set()
        selected_categories: List[str] = []

        # Exploit: top-scored candidates with diversity penalty.
        for candidate in candidates:
            if len(selected) >= exploit_count:
                break
            if candidate["book_id"] in selected_ids:
                continue
            category = candidate["category"]
            seen_cat_count = sum(1 for cat in selected_categories if cat == category)
            adjusted = candidate["base_score"] - self.weights.diversity_penalty * seen_cat_count
            row = dict(candidate)
            row["daily_score"] = adjusted
            row["strategy"] = "exploit"
            selected.append(row)
            selected_ids.add(candidate["book_id"])
            selected_categories.append(category)

        # Explore: category rarity/diversity to prevent same-cluster recommendations.
        remaining = [c for c in candidates if c["book_id"] not in selected_ids]
        remaining.sort(
            key=lambda c: (
                c["base_score"] + self._explore_boost(c["category"], selected_categories),
                -self._category_freq.get(c["category"], 1),
            ),
            reverse=True,
        )
        for candidate in remaining:
            if len(selected) >= target_count:
                break
            row = dict(candidate)
            row["daily_score"] = candidate["base_score"] + self._explore_boost(candidate["category"], selected_categories)
            row["strategy"] = "explore"
            selected.append(row)
            selected_ids.add(candidate["book_id"])
            selected_categories.append(candidate["category"])

        selected.sort(key=lambda row: row["daily_score"], reverse=True)
        return selected[:target_count]

    def _materialize(self, selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in selected:
            item = dict(self._metadata_by_id.get(row["book_id"], {}))
            item["daily_score"] = round(float(row.get("daily_score", row.get("base_score", 0.0))), 4)
            item["daily_reasons"] = row.get("reasons", {})
            item["daily_strategy"] = row.get("strategy", "exploit")
            out.append(item)
        return out

    def recommend_for_date(self, target_date: Optional[date] = None, count: int = 5) -> List[Dict[str, Any]]:
        target = self._to_date(target_date)
        history = self._load_history()
        reading_ids = self._load_reading_ids()
        recent_seen, all_seen = self._recent_history_ids(history, target)
        candidates = self._score_candidates(target, reading_ids, recent_seen, all_seen)
        selected = self._select_balanced(candidates, count=max(1, count))
        materialized = self._materialize(selected)

        day_key = self._date_key(target)
        history["days"][day_key] = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "book_ids": [row.get("book_id") for row in materialized],
            "books": [
                {
                    "book_id": row.get("book_id"),
                    "daily_score": row.get("daily_score"),
                    "strategy": row.get("daily_strategy"),
                    "reasons": row.get("daily_reasons"),
                }
                for row in materialized
            ],
        }
        self._save_history(history)
        return materialized

    def get_or_generate_for_date(
        self,
        target_date: Optional[date] = None,
        count: int = 5,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        target = self._to_date(target_date)
        day_key = self._date_key(target)
        history = self._load_history()
        if not force_refresh:
            day_data = history.get("days", {}).get(day_key)
            if isinstance(day_data, dict):
                book_ids = day_data.get("book_ids", [])
                if isinstance(book_ids, list):
                    hydrated: List[Dict[str, Any]] = []
                    books_meta = day_data.get("books", [])
                    reasons_by_id: Dict[str, Dict[str, Any]] = {}
                    if isinstance(books_meta, list):
                        for book_row in books_meta:
                            if isinstance(book_row, dict) and book_row.get("book_id"):
                                reasons_by_id[str(book_row["book_id"])] = book_row
                    for book_id in book_ids:
                        book_id_str = str(book_id)
                        item = self._metadata_by_id.get(book_id_str)
                        if not item:
                            continue
                        enriched = dict(item)
                        prev = reasons_by_id.get(book_id_str, {})
                        enriched["daily_score"] = float(prev.get("daily_score", 0.0) or 0.0)
                        enriched["daily_strategy"] = prev.get("strategy", "exploit")
                        enriched["daily_reasons"] = prev.get("reasons", {})
                        hydrated.append(enriched)
                    if hydrated:
                        return hydrated[: max(1, count)]
        return self.recommend_for_date(target_date=target, count=count)

