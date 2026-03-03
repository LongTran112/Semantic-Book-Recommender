import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from semantic_books.daily_recommend import DailyBookRecommender, DailyRecommendationWeights
from semantic_books.search_service import SemanticSearchService


class _FakeModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        _ = texts
        _ = normalize_embeddings
        _ = convert_to_numpy
        return np.array([[1.0, 0.0]], dtype=np.float32)


def _write_index(base_dir: Path) -> tuple[Path, Path, Path]:
    index_dir = base_dir / "semantic_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    vectors = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.0, 1.0],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )
    np.save(index_dir / "vectors.npy", vectors)

    books_dir = base_dir / "books"
    books_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    categories = [
        "DeepLearning",
        "AWS",
        "Generative-AI",
        "System-Design",
        "SQL",
        "Go",
        "Linux-SystemProgramming",
        "Parallel-Computing",
    ]
    for idx in range(8):
        path = books_dir / f"book_{idx}.pdf"
        path.write_text(f"book-{idx}", encoding="utf-8")
        metadata.append(
            {
                "book_id": f"b{idx}",
                "title": f"Book {idx}",
                "category": categories[idx],
                "learning_mode": "practical" if idx % 2 == 0 else "theory",
                "confidence": round(0.9 - 0.05 * idx, 3),
                "absolute_path": str(path),
                "matched_keywords": [f"title:k{idx}"],
            }
        )

    with (index_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    with (index_dir / "model_info.json").open("w", encoding="utf-8") as handle:
        json.dump({"model_name": "fake-model", "num_items": len(metadata)}, handle)

    reading_path = base_dir / "currently_reading.json"
    reading_path.write_text(json.dumps({"b0": {"book_id": "b0"}}), encoding="utf-8")
    history_path = base_dir / "daily_recommendations.json"
    return index_dir, reading_path, history_path


class DailyRecommendTests(unittest.TestCase):
    @patch("semantic_books.search_service.SentenceTransformer", return_value=_FakeModel())
    def test_daily_count_and_exclusion(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir, reading_path, history_path = _write_index(Path(tmp_dir))
            service = SemanticSearchService(index_dir)
            recommender = DailyBookRecommender(service, reading_path, history_path)

            recs = recommender.recommend_for_date(target_date=date(2026, 3, 3), count=5)
            self.assertEqual(len(recs), 5)
            self.assertNotIn("b0", {item.get("book_id") for item in recs})

    @patch("semantic_books.search_service.SentenceTransformer", return_value=_FakeModel())
    def test_deterministic_same_day(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir, reading_path, history_path = _write_index(Path(tmp_dir))
            service = SemanticSearchService(index_dir)
            recommender = DailyBookRecommender(
                service,
                reading_path,
                history_path,
                weights=DailyRecommendationWeights(),
            )
            d = date(2026, 3, 4)
            rec_a = recommender.recommend_for_date(target_date=d, count=5)
            rec_b = recommender.recommend_for_date(target_date=d, count=5)
            self.assertEqual(
                [item.get("book_id") for item in rec_a],
                [item.get("book_id") for item in rec_b],
            )

    @patch("semantic_books.search_service.SentenceTransformer", return_value=_FakeModel())
    def test_novelty_changes_next_day(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir, reading_path, history_path = _write_index(Path(tmp_dir))
            service = SemanticSearchService(index_dir)
            recommender = DailyBookRecommender(service, reading_path, history_path, novelty_window_days=14)
            day_one = recommender.recommend_for_date(target_date=date(2026, 3, 1), count=5)
            day_two = recommender.recommend_for_date(target_date=date(2026, 3, 1) + timedelta(days=1), count=5)
            overlap = {
                item.get("book_id") for item in day_one
            }.intersection({item.get("book_id") for item in day_two})
            self.assertLess(len(overlap), 5)


if __name__ == "__main__":
    unittest.main()

