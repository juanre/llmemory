"""Multi-language support for document processing.

This module provides language detection, normalization, and language-specific
text processing for improved search quality across multiple languages.
"""

import logging
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    from fast_langdetect import detect, detect_multilingual

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("fast_langdetect not available - language detection disabled")

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported languages for processing."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HEBREW = "he"
    HINDI = "hi"
    DUTCH = "nl"
    POLISH = "pl"
    UNKNOWN = "unknown"


@dataclass
class LanguageSegment:
    """A segment of text with detected language."""

    text: str
    language: str
    confidence: float
    start: int
    end: int


@dataclass
class LanguageConfig:
    """Configuration for language-specific processing."""

    stemmer: Optional[str] = None
    stop_words: Optional[str] = None
    tokenizer: Optional[str] = None
    direction: str = "ltr"  # left-to-right or rtl
    text_search_config: str = "simple"  # PostgreSQL text search config


class MultilingualProcessor:
    """Handles multi-language text processing and normalization."""

    # Language configurations for PostgreSQL and processing
    LANGUAGE_CONFIGS = {
        Language.ENGLISH: LanguageConfig(
            stemmer="english", stop_words="english", text_search_config="english"
        ),
        Language.SPANISH: LanguageConfig(
            stemmer="spanish", stop_words="spanish", text_search_config="spanish"
        ),
        Language.FRENCH: LanguageConfig(
            stemmer="french", stop_words="french", text_search_config="french"
        ),
        Language.GERMAN: LanguageConfig(
            stemmer="german", stop_words="german", text_search_config="german"
        ),
        Language.ITALIAN: LanguageConfig(
            stemmer="italian", stop_words="italian", text_search_config="italian"
        ),
        Language.PORTUGUESE: LanguageConfig(
            stemmer="portuguese",
            stop_words="portuguese",
            text_search_config="portuguese",
        ),
        Language.RUSSIAN: LanguageConfig(
            stemmer="russian", stop_words="russian", text_search_config="russian"
        ),
        Language.CHINESE: LanguageConfig(
            tokenizer="jieba", text_search_config="simple"
        ),
        Language.JAPANESE: LanguageConfig(
            tokenizer="mecab", text_search_config="simple"
        ),
        Language.KOREAN: LanguageConfig(
            tokenizer="konlpy", text_search_config="simple"
        ),
        Language.ARABIC: LanguageConfig(direction="rtl", text_search_config="arabic"),
        Language.HEBREW: LanguageConfig(direction="rtl", text_search_config="simple"),
        Language.DUTCH: LanguageConfig(
            stemmer="dutch", stop_words="dutch", text_search_config="dutch"
        ),
        Language.POLISH: LanguageConfig(text_search_config="simple"),
    }

    def __init__(self):
        """Initialize the multilingual processor."""
        self.configs = self.LANGUAGE_CONFIGS

    def detect_language(
        self, text: str, min_confidence: float = 0.8
    ) -> Tuple[str, float]:
        """
        Detect the primary language of a text.

        Args:
            text: Text to analyze
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (language_code, confidence)
        """
        if not LANGDETECT_AVAILABLE:
            return Language.UNKNOWN, 0.0

        try:
            # Use first 1000 chars for speed
            sample = text[:1000] if len(text) > 1000 else text
            result = detect(sample)

            if result["score"] >= min_confidence:
                # Map to our Language enum
                lang_code = result["lang"]
                if lang_code in [lang.value for lang in Language]:
                    return lang_code, result["score"]
                else:
                    logger.info(f"Detected unsupported language: {lang_code}")
                    return Language.UNKNOWN, result["score"]
            else:
                return Language.UNKNOWN, result["score"]

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return Language.UNKNOWN, 0.0

    def detect_language_segments(
        self, text: str, min_segment_length: int = 50
    ) -> List[LanguageSegment]:
        """
        Detect language boundaries in mixed-language text.

        Args:
            text: Text to analyze
            min_segment_length: Minimum characters per segment

        Returns:
            List of LanguageSegment objects
        """
        if not LANGDETECT_AVAILABLE:
            return [
                LanguageSegment(
                    text=text,
                    language=Language.UNKNOWN,
                    confidence=0.0,
                    start=0,
                    end=len(text),
                )
            ]

        segments = []

        try:
            # Split by common boundaries
            sentences = self._split_sentences(text)

            current_segment = []
            current_lang = None
            current_start = 0

            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:
                    # Too short to detect
                    if current_segment:
                        current_segment.append(sentence)
                    continue

                # Detect language
                lang, confidence = self.detect_language(sentence)

                if current_lang is None:
                    current_lang = lang
                    current_segment = [sentence]
                elif lang == current_lang or confidence < 0.7:
                    # Same language or low confidence - continue segment
                    current_segment.append(sentence)
                else:
                    # Language switch detected
                    if current_segment:
                        segment_text = " ".join(current_segment)
                        segments.append(
                            LanguageSegment(
                                text=segment_text,
                                language=current_lang,
                                confidence=confidence,
                                start=current_start,
                                end=current_start + len(segment_text),
                            )
                        )
                        current_start += len(segment_text) + 1

                    current_segment = [sentence]
                    current_lang = lang

            # Add final segment
            if current_segment:
                segment_text = " ".join(current_segment)
                segments.append(
                    LanguageSegment(
                        text=segment_text,
                        language=current_lang or Language.UNKNOWN,
                        confidence=0.8,
                        start=current_start,
                        end=current_start + len(segment_text),
                    )
                )

        except Exception as e:
            logger.error(f"Language segmentation failed: {e}")
            segments = [
                LanguageSegment(
                    text=text,
                    language=Language.UNKNOWN,
                    confidence=0.0,
                    start=0,
                    end=len(text),
                )
            ]

        return segments

    def normalize_text(self, text: str, language: str) -> str:
        """
        Apply language-specific text normalization.

        Args:
            text: Text to normalize
            language: Language code

        Returns:
            Normalized text
        """
        # Unicode normalization
        if language in [Language.ARABIC, Language.HEBREW]:
            # NFKD for Arabic/Hebrew
            text = unicodedata.normalize("NFKD", text)
        else:
            # NFKC for most languages
            text = unicodedata.normalize("NFKC", text)

        # Language-specific normalization
        if language == Language.ARABIC:
            text = self._normalize_arabic(text)
        elif language in [Language.CHINESE, Language.JAPANESE, Language.KOREAN]:
            text = self._normalize_cjk(text)
        elif language == Language.HEBREW:
            text = self._normalize_hebrew(text)

        return text.strip()

    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text."""
        # Remove Arabic diacritics (tashkeel)
        arabic_diacritics = [
            "\u064b",  # Fathatan
            "\u064c",  # Dammatan
            "\u064d",  # Kasratan
            "\u064e",  # Fatha
            "\u064f",  # Damma
            "\u0650",  # Kasra
            "\u0651",  # Shadda
            "\u0652",  # Sukun
        ]

        for diacritic in arabic_diacritics:
            text = text.replace(diacritic, "")

        # Normalize Alef variations
        text = text.replace("\u0622", "\u0627")  # Alef with madda
        text = text.replace("\u0623", "\u0627")  # Alef with hamza above
        text = text.replace("\u0625", "\u0627")  # Alef with hamza below

        # Normalize Yaa
        text = text.replace("\u064a", "\u0649")  # Yaa to Alef maksura

        return text

    def _normalize_cjk(self, text: str) -> str:
        """Normalize Chinese, Japanese, Korean text."""
        # Convert full-width characters to half-width
        text = unicodedata.normalize("NFKC", text)

        # Remove zero-width spaces
        text = text.replace("\u200b", "")
        text = text.replace("\u200c", "")
        text = text.replace("\u200d", "")

        return text

    def _normalize_hebrew(self, text: str) -> str:
        """Normalize Hebrew text."""
        # Remove Hebrew points (nikud)
        hebrew_points = range(0x0591, 0x05C8)
        for point in hebrew_points:
            text = text.replace(chr(point), "")

        return text

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for language detection.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be enhanced with NLTK if needed
        import re

        # Split on common sentence endings
        sentences = re.split(r"[.!?]+\s+", text)

        # Also split on newlines
        result = []
        for sentence in sentences:
            lines = sentence.split("\n")
            result.extend([line.strip() for line in lines if line.strip()])

        return result

    def get_text_search_config(self, language: str) -> str:
        """
        Get PostgreSQL text search configuration for a language.

        Args:
            language: Language code

        Returns:
            PostgreSQL text search config name
        """
        if language in self.configs:
            return self.configs[language].text_search_config
        return "simple"  # Default fallback

    def process_multilingual_document(self, text: str) -> Dict:
        """
        Process a potentially multilingual document.

        Args:
            text: Document text

        Returns:
            Dictionary with processing results
        """
        # Detect primary language
        primary_lang, confidence = self.detect_language(text)

        # Check for multilingual content
        segments = self.detect_language_segments(text)
        is_multilingual = len(set(seg.language for seg in segments)) > 1

        # Process each segment
        processed_segments = []
        for segment in segments:
            processed_text = self.normalize_text(segment.text, segment.language)
            processed_segments.append(
                {
                    "text": processed_text,
                    "language": segment.language,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": segment.confidence,
                }
            )

        return {
            "primary_language": primary_lang,
            "primary_confidence": confidence,
            "is_multilingual": is_multilingual,
            "segments": processed_segments,
            "languages": list(set(seg.language for seg in segments)),
            "text_search_config": self.get_text_search_config(primary_lang),
        }


# Create a global instance
multilingual_processor = MultilingualProcessor()


def detect_and_process_language(text: str) -> Dict:
    """
    Convenience function to detect and process language.

    Args:
        text: Text to process

    Returns:
        Language processing results
    """
    return multilingual_processor.process_multilingual_document(text)
