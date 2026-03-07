from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api


class _FakeRagService:
    def __init__(self) -> None:
        self.metadata = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        self.last_answer_kwargs = {}

    def retrieve_chunks(self, *args, **kwargs):
        _ = args
        _ = kwargs
        return [
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
                "chunk_text": "Deep learning theory includes optimization and backpropagation foundations.",
                "similarity": 0.92,
            }
        ]

    def answer_question(self, **kwargs):
        self.last_answer_kwargs = kwargs
        return {
            "answer": "Answer: Use gradient-based optimization [C1].",
            "summary": "Grounded from 1 retrieved chunks across 1 categories: DeepLearning",
            "follow_ups": ["Compare these sources by theoretical depth and practical exercises."],
            "citations": [
                {
                    "citation_id": "C1",
                    "title": "Deep Learning Theory",
                    "book_id": "b1",
                    "absolute_path": "/tmp/a.pdf",
                    "category": "DeepLearning",
                    "learning_mode": "theory",
                    "source_label": "body_preview:0",
                    "start_char": 0,
                    "end_char": 120,
                    "chunk_order": 0,
                    "chunk_len": 120,
                    "section_label": "body_preview",
                    "similarity": 0.92,
                    "snippet": "Deep learning theory includes optimization.",
                }
            ],
            "generation_mode": "deterministic",
            "fallback_reason": "",
            "metrics": {
                "total_ms": 123.4,
                "retrieval_ms": 23.4,
                "generation_ms": 100.0,
                "retrieved_chunks": 1,
                "used_citations": 1,
                "prompt_chars": 512,
                "answer_chars": 47,
                "peak_rss_mb": 256.0,
            },
        }

    def _build_follow_ups(self):
        return ["Compare these sources by theoretical depth and practical exercises."]


class RagApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    @patch("api.get_rag_service")
    def test_health_ok(self, mock_get_service) -> None:
        mock_get_service.return_value = _FakeRagService()
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["chunks_indexed"], 2)

    @patch("api.get_rag_service", side_effect=RuntimeError("missing index"))
    def test_health_failure_returns_503(self, _mock_get_service) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertIn("RAG service unavailable", response.json()["detail"])

    @patch("api.get_rag_service")
    def test_retrieve_endpoint_returns_chunks(self, mock_get_service) -> None:
        mock_get_service.return_value = _FakeRagService()
        response = self.client.post(
            "/rag/retrieve",
            json={
                "query": "deep learning foundations",
                "top_k": 4,
                "filters": {"categories": ["DeepLearning"], "learning_modes": ["theory"], "min_similarity": 0.0},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["chunks"])
        self.assertEqual(payload["chunks"][0]["book_id"], "b1")

    @patch("api.get_rag_service")
    def test_answer_endpoint_returns_contract(self, mock_get_service) -> None:
        fake = _FakeRagService()
        mock_get_service.return_value = fake
        response = self.client.post(
            "/rag/answer",
            json={
                "query": "deep learning foundations",
                "top_k": 4,
                "max_citations": 3,
                "llm": {"enabled": False},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("answer", payload)
        self.assertIn("summary", payload)
        self.assertIn("citations", payload)
        self.assertIn("follow_ups", payload)
        self.assertTrue(payload["follow_ups"])
        self.assertEqual(payload["generation_mode"], "deterministic")
        self.assertIn("metrics", payload)
        self.assertFalse(fake.last_answer_kwargs["llm_config"].enabled)
        self.assertFalse(fake.last_answer_kwargs["ollama_config"].enabled)

    @patch("api.get_rag_service")
    def test_answer_endpoint_accepts_ollama_config(self, mock_get_service) -> None:
        fake = _FakeRagService()
        mock_get_service.return_value = fake
        response = self.client.post(
            "/rag/answer",
            json={
                "query": "deep learning foundations",
                "ollama": {
                    "enabled": True,
                    "base_url": "http://127.0.0.1:11434",
                    "model": "deepseek-r1-local:latest",
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "num_ctx": 4096,
                    "timeout_sec": 30,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(fake.last_answer_kwargs["ollama_config"].enabled)
        self.assertEqual(fake.last_answer_kwargs["ollama_config"].model, "deepseek-r1-local:latest")

    def test_invalid_payload_returns_422(self) -> None:
        response = self.client.post("/rag/answer", json={"query": "", "top_k": 0})
        self.assertEqual(response.status_code, 422)

    @patch("api.get_rag_service")
    def test_answer_stream_endpoint_returns_events(self, mock_get_service) -> None:
        mock_get_service.return_value = _FakeRagService()
        with self.client.stream(
            "POST",
            "/rag/answer-stream",
            json={"query": "deep learning foundations", "ollama": {"enabled": True}},
        ) as response:
            self.assertEqual(response.status_code, 200)
            text = response.read().decode("utf-8")
        self.assertIn('"type": "final"', text)


if __name__ == "__main__":
    unittest.main()
