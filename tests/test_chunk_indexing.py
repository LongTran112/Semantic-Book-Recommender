import unittest

from index_books import (
    BookRecord,
    build_semantic_chunks,
    parse_category_depth_overrides,
    resolve_extraction_settings,
    summarize_chunk_coverage,
)


class ChunkIndexingTests(unittest.TestCase):
    def test_build_semantic_chunks_contains_required_fields(self) -> None:
        records = [
            BookRecord(
                category="DeepLearning",
                confidence=0.9,
                title="Deep Learning Theory",
                filename="deep-learning.pdf",
                absolute_path="/tmp/deep-learning.pdf",
                matched_keywords=["title:deep learning"],
                book_id="b1",
                metadata_text="Foundations and theory for neural networks.",
                body_preview=" ".join(["gradient descent and backpropagation"] * 80),
                learning_mode="theory",
            )
        ]
        chunks = build_semantic_chunks(records, chunk_size=400, chunk_overlap=50)
        self.assertGreater(len(chunks), 1)
        sample = chunks[0]
        self.assertTrue(sample.chunk_id)
        self.assertEqual(sample.book_id, "b1")
        self.assertIn(sample.source_type, {"metadata", "body_preview"})
        self.assertGreater(sample.end_char, sample.start_char)
        self.assertGreaterEqual(sample.chunk_order, 0)
        self.assertGreater(sample.chunk_len, 0)
        self.assertTrue(sample.section_label)
        self.assertTrue(sample.chunk_text.strip())

    def test_resolve_extraction_settings_custom_preserves_legacy_defaults(self) -> None:
        resolved = resolve_extraction_settings(
            profile_name="custom",
            cli_max_pages=None,
            cli_extract_timeout=None,
            cli_chunk_size=None,
            cli_chunk_overlap=None,
        )
        self.assertEqual(resolved.max_pages, 8)
        self.assertEqual(resolved.extract_timeout, 12)
        self.assertEqual(resolved.chunk_size, 1200)
        self.assertEqual(resolved.chunk_overlap, 200)

    def test_resolve_extraction_settings_profile_then_cli_override(self) -> None:
        resolved = resolve_extraction_settings(
            profile_name="deep",
            cli_max_pages=18,
            cli_extract_timeout=None,
            cli_chunk_size=1300,
            cli_chunk_overlap=240,
        )
        self.assertEqual(resolved.max_pages, 18)
        self.assertEqual(resolved.extract_timeout, 28)
        self.assertEqual(resolved.chunk_size, 1300)
        self.assertEqual(resolved.chunk_overlap, 240)

    def test_parse_category_depth_overrides(self) -> None:
        parsed = parse_category_depth_overrides(["Arduino=24:35", "SQL=16"])
        self.assertEqual(parsed["Arduino"].max_pages, 24)
        self.assertEqual(parsed["Arduino"].extract_timeout, 35)
        self.assertEqual(parsed["SQL"].max_pages, 16)
        self.assertIsNone(parsed["SQL"].extract_timeout)

    def test_summarize_chunk_coverage_counts_sources(self) -> None:
        records = [
            BookRecord(
                category="Embedded-Systems",
                confidence=0.9,
                title="Arduino Projects",
                filename="arduino.pdf",
                absolute_path="/tmp/arduino.pdf",
                matched_keywords=["forced_name:arduino"],
                book_id="b-arduino",
                metadata_text=(
                    "Arduino IDE, boards, sensors, microcontroller pin mapping, "
                    "breadboard setup, and serial monitor workflows."
                ),
                body_preview=" ".join(["Arduino helps build interactive systems"] * 60),
                learning_mode="practical",
            )
        ]
        chunks = build_semantic_chunks(records, chunk_size=500, chunk_overlap=50)
        summary = summarize_chunk_coverage(chunks)
        self.assertGreater(summary["total_chunks"], 0)
        self.assertGreater(summary["avg_chunk_len"], 0)
        self.assertIn("metadata", summary["source_type_counts"])
        self.assertIn("body_preview", summary["source_type_counts"])

    def test_build_semantic_chunks_is_deterministic_for_same_input(self) -> None:
        records = [
            BookRecord(
                category="Databases",
                confidence=0.88,
                title="SQL Internals",
                filename="sql.pdf",
                absolute_path="/tmp/sql.pdf",
                matched_keywords=["title:sql"],
                book_id="b-sql",
                metadata_text="SQL indexing, joins, and optimization plans.",
                body_preview=" ".join(["query planner cardinality and indexing strategy"] * 80),
                learning_mode="theory",
            )
        ]
        first = build_semantic_chunks(records, chunk_size=420, chunk_overlap=80)
        second = build_semantic_chunks(records, chunk_size=420, chunk_overlap=80)
        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])
        self.assertEqual([(c.start_char, c.end_char) for c in first], [(c.start_char, c.end_char) for c in second])

    def test_chunk_offsets_are_monotonic_per_source(self) -> None:
        records = [
            BookRecord(
                category="DeepLearning",
                confidence=0.95,
                title="Neural Nets",
                filename="nn.pdf",
                absolute_path="/tmp/nn.pdf",
                matched_keywords=["title:neural network"],
                book_id="b-nn",
                metadata_text="Backpropagation and optimization.",
                body_preview=" ".join(["gradient descent batch normalization regularization"] * 120),
                learning_mode="theory",
            )
        ]
        chunks = build_semantic_chunks(records, chunk_size=500, chunk_overlap=120)
        self.assertTrue(chunks)
        for source_type in {"metadata", "body_preview"}:
            subset = [c for c in chunks if c.source_type == source_type]
            for idx in range(1, len(subset)):
                self.assertGreaterEqual(subset[idx].start_char, subset[idx - 1].start_char)
                self.assertGreater(subset[idx].end_char, subset[idx].start_char)


if __name__ == "__main__":
    unittest.main()
