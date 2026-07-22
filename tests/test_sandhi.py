"""Unit tests for Malayalam Sandhi matching & lookup module."""
from __future__ import annotations

import os
import sqlite3
import pytest
from unittest.mock import patch, MagicMock

import definitions
from backend.malayalam_sandhi import (
    normalize_text,
    tokenize_words,
    string_similarity,
    align_reading_sandhi,
    split_known_suffixes,
    lookup_sandhi_compound,
)


def test_normalize_and_tokenize() -> None:
    raw = "മേശപ്പുറത്ത്\u200C ഉണ്ട്!"
    norm = normalize_text(raw)
    assert "\u200C" not in norm
    tokens = tokenize_words(raw)
    assert tokens == ["മേശപ്പുറത്ത്", "ഉണ്ട്"]


def test_tier1_sandhi_speech_alignment_combined_spoken() -> None:
    # Expected: 2 separate words ("മേശപ്പുറത്ത്", "ഉണ്ട്")
    # Spoken: 1 sandhi joined compound ("മേശപ്പുറത്തുണ്ട്")
    expected = "മേശപ്പുറത്ത് ഉണ്ട്"
    spoken = "മേശപ്പുറത്തുണ്ട്"

    res = align_reading_sandhi(expected, spoken, similarity_threshold=0.80)
    assert res["accuracy"] == 100
    assert res["expected_status"] == ["correct", "correct"]
    assert res["spoken_status"] == ["correct"]
    assert res["correct_count"] == 2
    assert res["missing_count"] == 0


def test_tier1_sandhi_speech_alignment_separated_spoken() -> None:
    # Expected: 1 compound word ("മേശപ്പുറത്തുണ്ട്")
    # Spoken: 2 separate words ("മേശപ്പുറത്ത്", "ഉണ്ട്")
    expected = "മേശപ്പുറത്തുണ്ട്"
    spoken = "മേശപ്പുറത്ത് ഉണ്ട്"

    res = align_reading_sandhi(expected, spoken, similarity_threshold=0.80)
    assert res["accuracy"] == 100
    assert res["expected_status"] == ["correct"]
    assert res["spoken_status"] == ["correct", "correct"]


def test_tier1_mispronunciation_flagged() -> None:
    # Expected: "മേശപ്പുറത്ത് ഉണ്ട്"
    # Spoken: "ആന" (completely different word)
    expected = "മേശപ്പുറത്ത് ഉണ്ട്"
    spoken = "ആന"

    res = align_reading_sandhi(expected, spoken, similarity_threshold=0.80)
    assert res["accuracy"] < 50
    assert "wrong" in res["expected_status"] or "missing" in res["expected_status"]


def test_tier2_suffix_splitting() -> None:
    word = "മേശപ്പുറത്തുണ്ട്"
    candidates = split_known_suffixes(word)
    stems = [c[0] for c in candidates]
    suffixes = [c[1] for c in candidates]
    assert "മേശപ്പുറത്ത്" in stems
    assert "ഉണ്ട്" in suffixes


def test_sandhi_dictionary_lookup_integration(tmp_path) -> None:
    db_file = tmp_path / "test_sandhi_cache.db"
    with patch("definitions.DB_PATH", str(db_file)):
        definitions.init_db()

        # Cache definitions for constituent base words
        definitions.cache_definition("മേശപ്പുറത്ത്", True, "on the table", "postposition", "http://wiktionary")
        definitions.cache_definition("ഉണ്ട്", True, "is / exists", "verb", "http://wiktionary")

        # Looking up the compound word "മേശപ്പുറത്തുണ്ട്" should trigger Sandhi fallback & return partial match
        res = definitions.lookup_word("മേശപ്പുറത്തുണ്ട്")
        assert res["found"] is True
        assert res.get("partial_match") is True
        assert res.get("tier") == 2
        assert "Part of this word:" in res["definition"]
        assert len(res.get("components", [])) == 2


def test_tier1_alignment_recovery_after_mismatch() -> None:
    # Expected: "മേശപ്പുറത്ത് ഉണ്ട് ആന അവിടെ ഉണ്ട്"
    # Spoken: "മേശപ്പുറത്ത് ഉണ്ട് പൂച്ച അവിടെ ഉണ്ട്"
    expected = "മേശപ്പുറത്ത് ഉണ്ട് ആന അവിടെ ഉണ്ട്"
    spoken = "മേശപ്പുറത്ത് ഉണ്ട് പൂച്ച അവിടെ ഉണ്ട്"

    res = align_reading_sandhi(expected, spoken, similarity_threshold=0.80)
    assert res["expected_status"] == ["correct", "correct", "wrong", "correct", "correct"]
    assert res["spoken_status"] == ["correct", "correct", "wrong", "correct", "correct"]
    assert res["accuracy"] == 80  # 4 out of 5 correct

