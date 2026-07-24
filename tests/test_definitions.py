from __future__ import annotations

import os
import sqlite3
import pytest
import unicodedata
from unittest.mock import patch, MagicMock

import definitions

def test_nfc_normalization() -> None:
    # Decomposed character (e + acute) should normalize to NFC
    decomposed = "e\u0301"
    normalized = unicodedata.normalize('NFC', decomposed)
    assert normalized == "é"

def test_init_db(tmp_path) -> None:
    db_file = tmp_path / "test_cache.db"
    with patch("definitions.DB_PATH", str(db_file)):
        definitions.init_db()
        assert db_file.exists()
        
        # Verify schema
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='word_definitions'")
        assert cursor.fetchone() is not None
        conn.close()

def test_cache_and_lookup_found(tmp_path) -> None:
    db_file = tmp_path / "test_cache.db"
    with patch("definitions.DB_PATH", str(db_file)):
        definitions.init_db()
        
        # Manually cache a word
        definitions.cache_definition("മരം", True, "tree", "noun", "http://source")
        
        # Lookup should hit cache
        res = definitions.lookup_word("മരം")
        assert res["found"] is True
        assert res["definition"] == "tree"
        assert res["part_of_speech"] == "noun"
        assert res["source_url"] == "http://source"

def test_cache_and_lookup_not_found(tmp_path) -> None:
    db_file = tmp_path / "test_cache.db"
    with patch("definitions.DB_PATH", str(db_file)):
        definitions.init_db()
        
        # Cache as not found
        definitions.cache_definition("invalid_word", False)
        
        # Mock fetch_from_api to verify it's not called (proving negative cache hit)
        with patch("definitions.fetch_from_api") as mock_fetch:
            res = definitions.lookup_word("invalid_word")
            assert res["found"] is False
            mock_fetch.assert_not_called()
