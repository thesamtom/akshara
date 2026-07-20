from __future__ import annotations

import os
import json
import logging
import sqlite3
import unicodedata
import urllib.request
import urllib.parse
from datetime import datetime
from urllib.error import HTTPError, URLError

logger = logging.getLogger("akshara.definitions")

DB_PATH = os.path.join(os.path.dirname(__file__), "cache.db")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def init_db() -> None:
    """Initializes the local SQLite caching database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS word_definitions (
                word TEXT PRIMARY KEY,
                found BOOLEAN NOT NULL,
                definition TEXT,
                part_of_speech TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Local SQLite cache database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite cache database: {e}")

def get_cached_definition(word: str) -> dict | None:
    """Queries cache (Supabase if available, fallback to SQLite)."""
    # 1. Try Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            url = f"{SUPABASE_URL}/rest/v1/word_definitions?word=eq.{urllib.parse.quote(word)}"
            req = urllib.request.Request(
                url,
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                if data:
                    row = data[0]
                    return {
                        "word": row["word"],
                        "found": bool(row["found"]),
                        "definition": row.get("definition"),
                        "part_of_speech": row.get("part_of_speech"),
                        "source_url": row.get("source_url"),
                        "attribution": "Definitions provided by Wiktionary via FreeDictionaryAPI.com"
                    }
        except Exception as e:
            logger.error(f"Supabase cache read failed for '{word}': {e}")
            # Fall back to SQLite even if Supabase is configured but fails

    # 2. SQLite fallback
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM word_definitions WHERE word = ?", (word,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "word": row["word"],
                "found": bool(row["found"]),
                "definition": row["definition"],
                "part_of_speech": row["part_of_speech"],
                "source_url": row["source_url"],
                "attribution": "Definitions provided by Wiktionary via FreeDictionaryAPI.com"
            }
    except Exception as e:
        logger.error(f"SQLite cache read failed for '{word}': {e}")

    return None

def cache_definition(
    word: str,
    found: bool,
    definition: str | None = None,
    part_of_speech: str | None = None,
    source_url: str | None = None
) -> None:
    """Saves lookup result to cache (Supabase + local SQLite fallback)."""
    # 1. Save to Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            url = f"{SUPABASE_URL}/rest/v1/word_definitions"
            payload = {
                "word": word,
                "found": found,
                "definition": definition,
                "part_of_speech": part_of_speech,
                "source_url": source_url
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
            logger.info(f"Successfully cached '{word}' in Supabase.")
        except Exception as e:
            logger.error(f"Failed to save '{word}' to Supabase cache: {e}")

    # 2. Save to SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO word_definitions (word, found, definition, part_of_speech, source_url)
            VALUES (?, ?, ?, ?, ?)
        """, (word, found, definition, part_of_speech, source_url))
        conn.commit()
        conn.close()
        logger.info(f"Successfully cached '{word}' in SQLite.")
    except Exception as e:
        logger.error(f"Failed to save '{word}' to SQLite cache: {e}")

def fetch_from_api(word: str) -> dict:
    """Queries the external Dictionary API and caches the result."""
    quoted_word = urllib.parse.quote(word)
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{quoted_word}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Akshara/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode('utf-8'))

            # The API returns a JSON array of word entry objects
            if not isinstance(payload, list) or not payload:
                cache_definition(word=word, found=False)
                return {"word": word, "found": False}

            entry = payload[0]
            source_urls = entry.get("sourceUrls", [])
            source_url = source_urls[0] if source_urls else None

            definition = None
            part_of_speech = None

            meanings = entry.get("meanings", [])
            for meaning in meanings:
                pos = meaning.get("partOfSpeech")
                defs = meaning.get("definitions", [])
                if defs:
                    defn = defs[0].get("definition")
                    if defn:
                        definition = defn
                        part_of_speech = pos
                        break

            if definition:
                cache_definition(
                    word=word,
                    found=True,
                    definition=definition,
                    part_of_speech=part_of_speech,
                    source_url=source_url
                )
                return {
                    "word": word,
                    "found": True,
                    "definition": definition,
                    "part_of_speech": part_of_speech,
                    "source_url": source_url,
                    "attribution": "Definitions via dictionaryapi.dev (Wiktionary)"
                }
            else:
                cache_definition(word=word, found=False)
                return {"word": word, "found": False}

    except HTTPError as e:
        if e.code == 404:
            logger.info(f"Word '{word}' not found in Dictionary API (404).")
            cache_definition(word=word, found=False)
            return {"word": word, "found": False}
        else:
            logger.error(f"Dictionary API HTTP error {e.code} for '{word}'")
            raise e
    except URLError as e:
        logger.error(f"Dictionary API network unreachable for '{word}': {e.reason}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected Dictionary API exception for '{word}': {e}")
        raise e

def lookup_word(word: str) -> dict:
    """Public wrapper to look up a word with NFC normalization and caching."""
    if not word or not word.strip():
        return {"word": "", "found": False}
        
    normalized = unicodedata.normalize('NFC', word.strip())
    
    # Check cache first
    cached = get_cached_definition(normalized)
    if cached is not None:
        logger.info(f"Cache hit for word '{normalized}'")
        return cached

    # Fetch from API on cache miss
    logger.info(f"Cache miss for word '{normalized}'. Fetching from API.")
    return fetch_from_api(normalized)
