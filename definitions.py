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

def clean_html_text(raw_html: str) -> str:
    """Strips HTML tags from Wiktionary definition strings."""
    if not raw_html:
        return ""
    clean = re.sub(r'<[^>]+>', '', raw_html).strip()
    return re.sub(r'\s+', ' ', clean)

def fetch_from_api(word: str) -> dict:
    """Queries Wiktionary REST API and FreeDictionaryAPI for Malayalam and English word meanings."""
    quoted_word = urllib.parse.quote(word)
    source_url = f"https://en.wiktionary.org/wiki/{quoted_word}"

    # 1. Primary Lookup: Official Wiktionary REST API (Supports Malayalam & English)
    wiktionary_url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{quoted_word}"
    req = urllib.request.Request(
        wiktionary_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AksharaApp/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=6) as response:
            payload = json.loads(response.read().decode('utf-8'))

            # Extract language entries (prefer 'ml' for Malayalam, then 'en', then any language entry)
            ml_entries = payload.get("ml", []) or payload.get("en", [])
            if not ml_entries and isinstance(payload, dict):
                for lang_key, entries in payload.items():
                    if entries and isinstance(entries, list):
                        ml_entries = entries
                        break

            if ml_entries:
                entry = ml_entries[0]
                pos = entry.get("partOfSpeech", "")
                raw_defs = [clean_html_text(d.get("definition", "")) for d in entry.get("definitions", [])]
                clean_defs = [d for d in raw_defs if d and len(d) > 1]

                if clean_defs:
                    definition = ", ".join(clean_defs[:2])
                    cache_definition(
                        word=word,
                        found=True,
                        definition=definition,
                        part_of_speech=pos,
                        source_url=source_url
                    )
                    return {
                        "word": word,
                        "found": True,
                        "definition": definition,
                        "part_of_speech": pos,
                        "source_url": source_url,
                        "attribution": "Definitions provided by Wiktionary"
                    }
    except HTTPError as e:
        if e.code != 404:
            logger.error(f"Wiktionary API HTTP error {e.code} for '{word}'")
    except Exception as e:
        logger.error(f"Wiktionary REST API lookup failed for '{word}': {e}")

    # 2. Fallback Lookup: FreeDictionaryAPI Malayalam endpoint
    freedict_url = f"https://freedictionaryapi.com/api/v1/entries/ml/{quoted_word}"
    req_fallback = urllib.request.Request(
        freedict_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        with urllib.request.urlopen(req_fallback, timeout=6) as response:
            payload = json.loads(response.read().decode('utf-8'))
            entries = payload.get("entries", [])
            for entry in entries:
                senses = entry.get("senses", [])
                for s in senses:
                    defn = clean_html_text(s.get("definition", ""))
                    if defn:
                        pos = entry.get("partOfSpeech", "")
                        alt_source = payload.get("source", {}).get("url") or source_url
                        cache_definition(
                            word=word,
                            found=True,
                            definition=defn,
                            part_of_speech=pos,
                            source_url=alt_source
                        )
                        return {
                            "word": word,
                            "found": True,
                            "definition": defn,
                            "part_of_speech": pos,
                            "source_url": alt_source,
                            "attribution": "Definitions provided by Wiktionary via FreeDictionaryAPI"
                        }
    except HTTPError as e:
        if e.code == 404:
            logger.info(f"Word '{word}' not found in dictionary APIs (404).")
    except Exception as e:
        logger.error(f"FreeDictionaryAPI fallback failed for '{word}': {e}")

    # Not found in any API -> cache as not found
    cache_definition(word=word, found=False)
    return {"word": word, "found": False}

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
