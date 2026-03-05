import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from semantic_books.rag_service import RagFilters, RagService


class _FakeModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        _ = normalize_embeddings
        _ = convert_to_numpy
        text = texts[0].lower()
        if "deep learning" in text:
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 1.0]], dtype=np.float32)


def _write_chunk_index(base_dir: Path) -> Path:
    index_dir = base_dir / "semantic_index_chunks"
    index_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(index_dir / "vectors.npy", vectors)
    metadata = [
        {
            "chunk_id": "c1",
            "book_id": "b1",
            "title": "Deep Learning Theory",
            "category": "DeepLearning",
            "learning_mode": "theory",
            "absolute_path": "/tmp/a.pdf",
            "source_type": "body_preview",
            "source_index": 0,
            "start_char": 0,
            "end_char": 120,
            "chunk_text": "Deep learning theory includes optimization and backpropagation foundations.",
        },
        {
            "chunk_id": "c2",
            "book_id": "b2",
            "title": "Linux Practical Guide",
            "category": "Linux-SystemProgramming",
            "learning_mode": "practical",
            "absolute_path": "/tmp/b.pdf",
            "source_type": "body_preview",
            "source_index": 0,
            "start_char": 0,
            "end_char": 120,
            "chunk_text": "Linux shell practical workflow and command-line scripting examples.",
        },
    ]
    with (index_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    with (index_dir / "model_info.json").open("w", encoding="utf-8") as handle:
        json.dump({"model_name": "fake-model", "num_items": 2}, handle)
    return index_dir


class RagServiceTests(unittest.TestCase):
    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    def test_answer_question_returns_grounded_citations(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            response = service.answer_question(
                query="Give me deep learning theory foundations",
                filters=RagFilters(categories=["DeepLearning"], learning_modes=["theory"]),
                top_k=4,
            )
            self.assertIn("grounded", response["answer"].lower())
            self.assertTrue(response["citations"])
            self.assertEqual(response["citations"][0]["book_id"], "b1")


if __name__ == "__main__":
    unittest.main()
