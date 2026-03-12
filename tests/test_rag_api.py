from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
import django

django.setup()

from django.test import Client

from rag_api import guardrails


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

    def _build_follow_ups(self, *args, **kwargs):
        _ = args
        _ = kwargs
        return ["Compare these sources by theoretical depth and practical exercises."]

    def _validate_generated_answer(self, text, known_citations):
        inline = {"C1"} if "[C1]" in str(text) else set()
        return bool(inline) and inline.issubset(set(known_citations))


class RagApiTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RAG_API_KEY"] = "test-internal-key"
        os.environ["RAG_RATE_LIMIT_WINDOW_SEC"] = "60"
        os.environ["RAG_RATE_LIMIT_MAX_REQUESTS"] = "100"
        guardrails.reset_rate_limit_state()
        self.client = Client()
        self.headers = {"HTTP_X_API_KEY": "test-internal-key"}

    @patch("rag_api.views.get_rag_service")
    def test_health_ok(self, mock_get_service) -> None:
        mock_get_service.return_value = _FakeRagService()
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["chunks_indexed"], 2)

    @patch("rag_api.views.get_rag_service", side_effect=RuntimeError("missing index"))
    def test_health_failure_returns_503(self, _mock_get_service) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertIn("RAG service unavailable", response.json()["detail"])

    @patch("rag_api.views.get_rag_service")
    def test_retrieve_endpoint_returns_chunks(self, mock_get_service) -> None:
        mock_get_service.return_value = _FakeRagService()
        response = self.client.post(
            "/rag/retrieve",
            data=json.dumps(
                {
                    "query": "deep learning foundations",
                    "top_k": 4,
                    "filters": {"categories": ["DeepLearning"], "learning_modes": ["theory"], "min_similarity": 0.0},
                }
            ),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["chunks"])
        self.assertEqual(payload["chunks"][0]["book_id"], "b1")
        self.assertEqual(payload["chunks"][0]["absolute_path"], "")

    @patch("rag_api.views.get_rag_service")
    def test_answer_endpoint_returns_contract(self, mock_get_service) -> None:
        fake = _FakeRagService()
        mock_get_service.return_value = fake
        response = self.client.post(
            "/rag/answer",
            data=json.dumps(
                {
                    "query": "deep learning foundations",
                    "top_k": 4,
                    "max_citations": 3,
                    "llm": {"enabled": False},
                }
            ),
            content_type="application/json",
            **self.headers,
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
        self.assertEqual(payload["citations"][0]["absolute_path"], "")

    @patch("rag_api.views.get_rag_service")
    def test_answer_endpoint_accepts_ollama_config(self, mock_get_service) -> None:
        fake = _FakeRagService()
        mock_get_service.return_value = fake
        response = self.client.post(
            "/rag/answer",
            data=json.dumps(
                {
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
                }
            ),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(fake.last_answer_kwargs["ollama_config"].enabled)
        self.assertEqual(fake.last_answer_kwargs["ollama_config"].model, "deepseek-r1-local:latest")

    def test_invalid_payload_returns_422(self) -> None:
        response = self.client.post(
            "/rag/answer",
            data=json.dumps({"query": "", "top_k": 0}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(response.status_code, 422)

    def test_protected_endpoint_rejects_missing_api_key(self) -> None:
        response = self.client.post(
            "/rag/answer",
            data=json.dumps({"query": "test question"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 401)

    def test_protected_endpoint_rejects_invalid_api_key(self) -> None:
        response = self.client.post(
            "/rag/answer",
            data=json.dumps({"query": "test question"}),
            content_type="application/json",
            HTTP_X_API_KEY="wrong-key",
        )
        self.assertEqual(response.status_code, 401)

    def test_openapi_schema_endpoint_available(self) -> None:
        response = self.client.get("/openapi/schema/", HTTP_ACCEPT="application/json")
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertEqual(payload.get("openapi"), "3.0.3")
        security_schemes = payload.get("components", {}).get("securitySchemes", {})
        self.assertIn("ApiKeyAuth", security_schemes)
        self.assertEqual(security_schemes["ApiKeyAuth"].get("name"), "X-API-Key")
        rag_answer_security = payload.get("paths", {}).get("/rag/answer", {}).get("post", {}).get("security", [])
        self.assertIn("ApiKeyAuth", rag_answer_security)

    def test_openapi_swagger_ui_endpoint_available(self) -> None:
        response = self.client.get("/openapi/swagger/")
        self.assertEqual(response.status_code, 200)

    @patch("rag_api.views.get_rag_service")
    def test_rate_limit_returns_429(self, mock_get_service) -> None:
        mock_get_service.return_value = _FakeRagService()
        os.environ["RAG_RATE_LIMIT_MAX_REQUESTS"] = "2"
        guardrails.reset_rate_limit_state()
        first = self.client.post(
            "/rag/retrieve",
            data=json.dumps({"query": "a"}),
            content_type="application/json",
            **self.headers,
        )
        second = self.client.post(
            "/rag/retrieve",
            data=json.dumps({"query": "b"}),
            content_type="application/json",
            **self.headers,
        )
        third = self.client.post(
            "/rag/retrieve",
            data=json.dumps({"query": "c"}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 429)

    @patch("rag_api.views.get_rag_service")
    def test_answer_stream_endpoint_returns_events(self, mock_get_service) -> None:
        mock_get_service.return_value = _FakeRagService()
        response = self.client.post(
            "/rag/answer-stream",
            data=json.dumps({"query": "deep learning foundations", "ollama": {"enabled": True}}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(response.status_code, 200)
        text = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn('"type": "final"', text)
        self.assertNotIn("/tmp/a.pdf", text)

    @patch("rag_api.views.RagServiceRetriever")
    @patch("rag_api.views.build_lc_answer_chain", return_value=None)
    @patch("rag_api.views.get_rag_service")
    def test_answer_lc_falls_back_when_chain_unavailable(
        self,
        mock_get_service,
        _mock_chain,
        mock_retriever_cls,
    ) -> None:
        fake = _FakeRagService()
        mock_get_service.return_value = fake

        class _Doc:
            def __init__(self):
                self.page_content = "Deep learning uses optimization and backpropagation."
                self.metadata = {
                    "title": "Deep Learning Theory",
                    "book_id": "b1",
                    "category": "DeepLearning",
                    "learning_mode": "theory",
                    "source_type": "body_preview",
                    "source_index": 0,
                    "start_char": 0,
                    "end_char": 120,
                    "chunk_order": 0,
                    "chunk_len": 120,
                    "similarity": 0.92,
                }

        class _Retriever:
            def invoke(self, _query):
                return [_Doc()]

        mock_retriever_cls.return_value = _Retriever()
        response = self.client.post(
            "/rag/answer-lc",
            data=json.dumps({"query": "deep learning foundations", "top_k": 4, "max_citations": 3}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("fallback_reason", payload)
        self.assertIn("LangChain chain unavailable", payload["fallback_reason"])
        self.assertEqual(payload["citations"][0]["absolute_path"], "")

    @patch("rag_api.views.RagServiceRetriever")
    @patch("rag_api.views.build_lc_answer_chain")
    @patch("rag_api.views.get_rag_service")
    def test_answer_lc_falls_back_on_invalid_citations(
        self,
        mock_get_service,
        mock_build_chain,
        mock_retriever_cls,
    ) -> None:
        fake = _FakeRagService()
        mock_get_service.return_value = fake

        class _Doc:
            def __init__(self):
                self.page_content = "Deep learning uses optimization and backpropagation."
                self.metadata = {
                    "title": "Deep Learning Theory",
                    "book_id": "b1",
                    "category": "DeepLearning",
                    "learning_mode": "theory",
                    "source_type": "body_preview",
                    "source_index": 0,
                    "start_char": 0,
                    "end_char": 120,
                    "chunk_order": 0,
                    "chunk_len": 120,
                    "similarity": 0.92,
                }

        class _Retriever:
            def invoke(self, _query):
                return [_Doc()]

        mock_retriever_cls.return_value = _Retriever()

        class _FakeChain:
            def invoke(self, _payload):
                return "Answer: This omits citation ids.\nSourcesUsed: X9"

        mock_build_chain.return_value = _FakeChain()
        response = self.client.post(
            "/rag/answer-lc",
            data=json.dumps({"query": "deep learning foundations", "top_k": 4, "max_citations": 3}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("LangChain output missing valid citation markers.", payload["fallback_reason"])
        self.assertEqual(payload["citations"][0]["absolute_path"], "")


if __name__ == "__main__":
    unittest.main()
