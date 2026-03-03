import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from semantic_books.search_service import SearchFilters, SemanticSearchService


class _FakeModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        _ = normalize_embeddings
        _ = convert_to_numpy
        text = texts[0].lower()
        if "deep learning" in text:
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 1.0]], dtype=np.float32)


def _write_index(base_dir: Path) -> Path:
    index_dir = base_dir / "semantic_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(index_dir / "vectors.npy", vectors)

    metadata = [
        {
            "book_id": "a1",
            "title": "Deep Learning Theory",
            "category": "DeepLearning",
            "learning_mode": "theory",
            "confidence": 0.9,
            "absolute_path": "/tmp/a1.pdf",
            "matched_keywords": ["title:deep learning"],
        },
        {
            "book_id": "b1",
            "title": "Linux Practical Guide",
            "category": "Linux-SystemProgramming",
            "learning_mode": "practical",
            "confidence": 0.8,
            "absolute_path": "/tmp/b1.pdf",
            "matched_keywords": ["title:linux programming"],
        },
    ]
    with (index_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    with (index_dir / "model_info.json").open("w", encoding="utf-8") as handle:
        json.dump({"model_name": "fake-model", "num_items": 2}, handle)
    return index_dir


class SearchServiceTests(unittest.TestCase):
    @patch("semantic_books.search_service.SentenceTransformer", return_value=_FakeModel())
    def test_search_and_related(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_index(Path(tmp_dir))
            service = SemanticSearchService(index_dir)

            results = service.search_books(
                "give me book to learn about deep learning theory",
                filters=SearchFilters(categories=["DeepLearning"], learning_modes=["theory"], min_similarity=0.0),
                top_k=5,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["book_id"], "a1")

            related = service.recommend_related("a1", top_k=5)
            self.assertTrue(related)
            self.assertEqual(related[0]["book_id"], "b1")


if __name__ == "__main__":
    unittest.main()

