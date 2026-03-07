import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from semantic_books.generation_service import GenerationResult
from semantic_books.rag_config import LlamaCppConfig, OllamaConfig, RetrievalConfig
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
            "chunk_order": 0,
            "chunk_len": 120,
            "section_label": "body_preview",
            "chunk_text": "Deep learning is a subset of machine learning using multilayer neural networks.",
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
            "chunk_order": 0,
            "chunk_len": 120,
            "section_label": "body_preview",
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
            self.assertIn("answer", response["answer"].lower())
            self.assertTrue(response["citations"])
            self.assertEqual(response["citations"][0]["book_id"], "b1")
            self.assertIn("citation_id", response["citations"][0])
            self.assertTrue(response["follow_ups"])
            self.assertEqual(len(response["follow_ups"]), 3)
            self.assertTrue(any("deep learning" in item.lower() for item in response["follow_ups"]))
            citation_ids = [str(item.get("citation_id", "")) for item in response["citations"]]
            self.assertTrue(all(cid.startswith("C") for cid in citation_ids))
            self.assertIn("metrics", response)
            self.assertIn("total_ms", response["metrics"])

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    def test_answer_question_returns_cancelled_response_when_requested(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            response = service.answer_question(
                query="Give me deep learning theory foundations",
                should_cancel=lambda: True,
            )
            self.assertEqual(response["fallback_reason"], "request_cancelled")
            self.assertEqual(response["citations"], [])
            self.assertEqual(response["follow_ups"], [])

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    def test_definition_query_prefers_grounded_definition_when_similarity_is_high(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            response = service.answer_question(
                query="What is deep learning?",
                filters=RagFilters(categories=["DeepLearning"]),
                top_k=4,
            )
            answer = response["answer"]
            self.assertNotIn("Insufficient grounded definition in provided sources", answer)
            self.assertIn("[C1]", answer)

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    def test_hybrid_retrieval_includes_lexical_signals(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            chunks = service.retrieve_chunks(
                query="multilayer neural networks",
                filters=RagFilters(),
                top_k=2,
                retrieval_config=RetrievalConfig(
                    hybrid_enabled=True,
                    dense_weight=0.5,
                    lexical_weight=0.5,
                    candidate_pool_size=10,
                    final_top_k=2,
                ),
            )
            self.assertTrue(chunks)
            self.assertIn("lexical_score", chunks[0])
            self.assertEqual(chunks[0]["book_id"], "b1")
            self.assertIn("relevance_score", chunks[0])

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    def test_hybrid_filtering_uses_fused_relevance_not_raw_dense(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            with patch.object(service, "_dense_scores", return_value={0: -0.4, 1: -0.3}):
                chunks = service.retrieve_chunks(
                    query="multilayer neural networks",
                    filters=RagFilters(min_similarity=0.35),
                    top_k=2,
                    retrieval_config=RetrievalConfig(
                        hybrid_enabled=True,
                        dense_weight=0.2,
                        lexical_weight=0.8,
                        candidate_pool_size=10,
                        final_top_k=2,
                    ),
                )
            self.assertTrue(chunks)
            self.assertEqual(chunks[0]["book_id"], "b1")
            self.assertGreater(chunks[0]["relevance_score"], 0.35)

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    def test_reranker_toggle_falls_back_when_model_missing(self, _mock_model) -> None:
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            chunks = service.retrieve_chunks(
                query="deep learning theory",
                filters=RagFilters(),
                retrieval_config=RetrievalConfig(
                    hybrid_enabled=True,
                    reranker_enabled=True,
                    reranker_model_name="",
                    final_top_k=2,
                ),
            )
            self.assertTrue(chunks)
            self.assertIn("fused_score", chunks[0])

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    @patch("semantic_books.rag_service.create_generator")
    def test_generation_fallback_when_invalid_citation_markers(
        self,
        mock_create_generator,
        _mock_model,
    ) -> None:
        class _BadGenerator:
            def generate(self, _prompt: str) -> GenerationResult:
                return GenerationResult(text="Answer: This has no citations.", backend="llama.cpp")

        mock_create_generator.return_value = _BadGenerator()
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            response = service.answer_question(
                query="Give me deep learning theory foundations",
                filters=RagFilters(categories=["DeepLearning"]),
                retrieval_config=RetrievalConfig(final_top_k=4),
                llm_config=LlamaCppConfig(enabled=True, model_path="/tmp/model.gguf"),
            )
            self.assertEqual(response["generation_mode"], "deterministic")
            self.assertTrue(response["fallback_reason"])
            self.assertIn("[C1]", response["answer"])

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    @patch("semantic_books.rag_service.create_generator")
    def test_generation_rejects_reasoning_leak_even_with_valid_citations(
        self,
        mock_create_generator,
        _mock_model,
    ) -> None:
        class _LeakyGenerator:
            def generate(self, _prompt: str) -> GenerationResult:
                return GenerationResult(
                    text=(
                        "So, I need to answer this carefully.\n"
                        "Answer: Deep learning uses layered representations [C1].\n"
                        "SourcesUsed: C1"
                    ),
                    backend="ollama",
                )

        mock_create_generator.return_value = _LeakyGenerator()
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            response = service.answer_question(
                query="What is deep learning?",
                filters=RagFilters(categories=["DeepLearning"]),
                retrieval_config=RetrievalConfig(final_top_k=4),
                ollama_config=OllamaConfig(enabled=True, model="deepseek-r1-local:latest"),
            )
            self.assertEqual(response["generation_mode"], "deterministic")
            self.assertIn("citation markers", response["fallback_reason"])

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    @patch("semantic_books.rag_service.create_generator")
    def test_generation_uses_llama_cpp_when_valid(
        self,
        mock_create_generator,
        _mock_model,
    ) -> None:
        class _GoodGenerator:
            def generate(self, _prompt: str) -> GenerationResult:
                return GenerationResult(
                    text="Answer: Focus on optimization and generalization [C1].\nSourcesUsed: C1",
                    backend="llama.cpp",
                )

        mock_create_generator.return_value = _GoodGenerator()
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            response = service.answer_question(
                query="Give me deep learning theory foundations",
                filters=RagFilters(categories=["DeepLearning"]),
                retrieval_config=RetrievalConfig(final_top_k=4),
                llm_config=LlamaCppConfig(enabled=True, model_path="/tmp/model.gguf"),
            )
            self.assertEqual(response["generation_mode"], "llama.cpp")
            self.assertIn("[C1]", response["answer"])
            self.assertFalse(response["fallback_reason"])

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    @patch("semantic_books.rag_service.create_generator")
    def test_generation_uses_ollama_when_valid(
        self,
        mock_create_generator,
        _mock_model,
    ) -> None:
        class _OllamaGenerator:
            def generate(self, _prompt: str) -> GenerationResult:
                return GenerationResult(
                    text="Answer: Use regularization and validation checks [C1].\nSourcesUsed: C1",
                    backend="ollama",
                )

        mock_create_generator.return_value = _OllamaGenerator()
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            response = service.answer_question(
                query="Give me deep learning theory foundations",
                filters=RagFilters(categories=["DeepLearning"]),
                retrieval_config=RetrievalConfig(final_top_k=4),
                llm_config=LlamaCppConfig(enabled=False),
                ollama_config=OllamaConfig(enabled=True, model="deepseek-r1-local:latest"),
            )
            self.assertEqual(response["generation_mode"], "ollama")
            self.assertIn("[C1]", response["answer"])
            self.assertFalse(response["fallback_reason"])

    @patch("semantic_books.rag_service.SentenceTransformer", return_value=_FakeModel())
    @patch("semantic_books.rag_service.create_generator")
    def test_generation_rejects_sources_used_only_for_nontrivial_answers(
        self,
        mock_create_generator,
        _mock_model,
    ) -> None:
        class _SourcesUsedOnlyGenerator:
            def generate(self, _prompt: str) -> GenerationResult:
                return GenerationResult(
                    text=(
                        "Answer: Deep learning uses layered representations for pattern discovery, "
                        "optimization, and feature abstraction across many data modalities, and this "
                        "description intentionally stays long enough to require inline evidence markers; "
                        "it also references training stability, initialization strategy, regularization "
                        "trade-offs, and dataset shift considerations in practical model development.\n"
                        "SourcesUsed: C1"
                    ),
                    backend="ollama",
                )

        mock_create_generator.return_value = _SourcesUsedOnlyGenerator()
        with TemporaryDirectory() as tmp_dir:
            index_dir = _write_chunk_index(Path(tmp_dir))
            service = RagService(index_dir)
            response = service.answer_question(
                query="Give me deep learning theory foundations",
                filters=RagFilters(categories=["DeepLearning"]),
                retrieval_config=RetrievalConfig(final_top_k=4),
                llm_config=LlamaCppConfig(enabled=False),
                ollama_config=OllamaConfig(enabled=True, model="deepseek-r1-local:latest"),
            )
            self.assertEqual(response["generation_mode"], "deterministic")
            self.assertIn("citation markers", response["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
