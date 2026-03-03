import unittest

from semantic_books.learning_mode import (
    MODE_BALANCED,
    MODE_PRACTICAL,
    MODE_THEORY,
    MODE_UNKNOWN,
    infer_learning_mode,
)


class LearningModeTests(unittest.TestCase):
    def test_infer_learning_mode_theory(self) -> None:
        text = "A theoretical foundation with proofs and mathematical principles."
        self.assertEqual(infer_learning_mode(text), MODE_THEORY)

    def test_infer_learning_mode_practical(self) -> None:
        text = "Hands-on tutorial with projects and implementation examples."
        self.assertEqual(infer_learning_mode(text), MODE_PRACTICAL)

    def test_infer_learning_mode_balanced(self) -> None:
        text = "A practical guide with theory and principles."
        self.assertEqual(infer_learning_mode(text), MODE_BALANCED)

    def test_infer_learning_mode_unknown(self) -> None:
        self.assertEqual(infer_learning_mode(""), MODE_UNKNOWN)


if __name__ == "__main__":
    unittest.main()

