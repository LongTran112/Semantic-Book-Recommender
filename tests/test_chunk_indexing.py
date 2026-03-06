import unittest

from index_books import BookRecord, build_semantic_chunks


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


if __name__ == "__main__":
    unittest.main()
