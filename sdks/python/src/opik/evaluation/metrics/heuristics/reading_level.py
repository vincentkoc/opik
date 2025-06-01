from typing import Any, Optional
import re

from opik.exceptions import MetricComputationError
from .. import base_metric, score_result

try:
    import nltk
except ImportError:  # pragma: no cover - dependency missing
    nltk = None


class ReadingLevel(base_metric.BaseMetric):
    """Compute the Flesch–Kincaid grade level of a text."""

    def __init__(self, name: str = "reading_level_metric", track: bool = True, project_name: Optional[str] = None) -> None:
        super().__init__(name=name, track=track, project_name=project_name)
        if nltk is None:
            raise ImportError(
                "`nltk` library is required for ReadingLevel calculation. Install via `pip install nltk`."
            )

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e"):
            count = max(1, count - 1)
        return count or 1

    def score(self, output: str, **ignored_kwargs: Any) -> score_result.ScoreResult:
        if not output.strip():
            raise MetricComputationError("Output is empty.")

        sentences = [s for s in re.split(r"[.!?]+", output) if s.strip()]
        words = [w for w in nltk.tokenize.wordpunct_tokenize(output) if re.match(r"[A-Za-z]", w)]

        if not sentences or not words:
            raise MetricComputationError("Insufficient text to compute reading level.")

        syllables = sum(self._count_syllables(w) for w in words)
        num_words = len(words)
        num_sentences = len(sentences)
        grade = 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59
        return score_result.ScoreResult(value=grade, name=self.name)
