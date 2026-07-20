from __future__ import annotations

import os
import re
import json
import logging
import sqlite3
import unicodedata
import urllib.request
import urllib.parse
from datetime import datetime
from urllib.error import HTTPError, URLError

import backend.malayalam_sandhi

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
                raw_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Ensure column raw_json exists for upgrades
        cursor.execute("PRAGMA table_info(word_definitions)")
        cols = [col[1] for col in cursor.fetchall()]
        if "raw_json" not in cols:
            cursor.execute("ALTER TABLE word_definitions ADD COLUMN raw_json TEXT")

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
                if data and data[0].get("found"):
                    row = data[0]
                    if row.get("raw_json"):
                        try:
                            return json.loads(row["raw_json"])
                        except Exception:
                            pass
                    return {
                        "word": row["word"],
                        "found": True,
                        "definition": row.get("definition"),
                        "part_of_speech": row.get("part_of_speech"),
                        "source_url": row.get("source_url"),
                        "attribution": "Definitions provided by Wiktionary"
                    }
        except Exception as e:
            logger.error(f"Supabase cache read failed for '{word}': {e}")

    # 2. SQLite fallback
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM word_definitions WHERE word = ? AND found = 1", (word,))
        row = cursor.fetchone()
        conn.close()
        if row:
            if row["raw_json"]:
                try:
                    return json.loads(row["raw_json"])
                except Exception:
                    pass
            return {
                "word": row["word"],
                "found": True,
                "definition": row["definition"],
                "part_of_speech": row["part_of_speech"],
                "source_url": row["source_url"],
                "attribution": "Definitions provided by Wiktionary"
            }
    except Exception as e:
        logger.error(f"SQLite cache read failed for '{word}': {e}")

    return None

def cache_definition(
    word: str,
    found: bool,
    definition: str | None = None,
    part_of_speech: str | None = None,
    source_url: str | None = None,
    raw_json: str | None = None
) -> None:
    """Saves successful lookup result to cache (Supabase + local SQLite fallback)."""
    # Do not write negative (unfound) entries to cache
    if not found or not definition:
        return
    # 1. Save to Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            url = f"{SUPABASE_URL}/rest/v1/word_definitions"
            payload = {
                "word": word,
                "found": found,
                "definition": definition,
                "part_of_speech": part_of_speech,
                "source_url": source_url,
                "raw_json": raw_json
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
            INSERT OR REPLACE INTO word_definitions (word, found, definition, part_of_speech, source_url, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (word, found, definition, part_of_speech, source_url, raw_json))
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
    """Queries FreeDictionaryAPI & Wiktionary REST API for Malayalam and English word meanings matching exact JSON schema."""
    quoted_word = urllib.parse.quote(word)
    source_url = f"https://en.wiktionary.org/wiki/{quoted_word}"

    # 1. Primary Lookup: FreeDictionaryAPI (returns exact structured JSON schema with pronunciations, forms, senses)
    for lang_code in ["ml", "en"]:
        freedict_url = f"https://freedictionaryapi.com/api/v1/entries/{lang_code}/{quoted_word}"
        req = urllib.request.Request(
            freedict_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AksharaApp/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as response:
                payload = json.loads(response.read().decode('utf-8'))
                entries = payload.get("entries", [])
                if entries:
                    definition = ""
                    pos = entries[0].get("partOfSpeech", "")
                    for e in entries:
                        for s in e.get("senses", []):
                            d = clean_html_text(s.get("definition", ""))
                            if d:
                                definition = d
                                if not pos: pos = e.get("partOfSpeech", "")
                                break
                        if definition: break

                    ipa = ""
                    rom = ""
                    for e in entries:
                        for p in e.get("pronunciations", []):
                            if p.get("text"):
                                ipa = p["text"]
                                break
                        for f in e.get("forms", []):
                            if f.get("word") and "romanization" in f.get("tags", []):
                                rom = f["word"]
                                break

                    source_info = payload.get("source") or {
                        "url": source_url,
                        "license": {"name": "CC BY-SA 4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"}
                    }
                    src_link = source_info.get("url") or source_url

                    result_obj = {
                        "word": payload.get("word", word),
                        "found": True,
                        "definition": definition,
                        "part_of_speech": pos,
                        "ipa": ipa,
                        "romanization": rom,
                        "source_url": src_link,
                        "entries": entries,
                        "source": source_info,
                        "attribution": "Definitions via FreeDictionaryAPI (Wiktionary)"
                    }
                    cache_definition(
                        word=word,
                        found=True,
                        definition=definition,
                        part_of_speech=pos,
                        source_url=src_link,
                        raw_json=json.dumps(result_obj, ensure_ascii=False)
                    )
                    return result_obj
        except HTTPError as e:
            if e.code != 404:
                logger.error(f"FreeDictionaryAPI HTTP error {e.code} for '{word}'")
        except Exception as e:
            logger.error(f"FreeDictionaryAPI lookup failed for '{word}': {e}")

    # 2. Fallback Lookup: Official Wiktionary REST API
    wiktionary_url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{quoted_word}"
    req_wik = urllib.request.Request(
        wiktionary_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AksharaApp/1.0"}
    )
    try:
        with urllib.request.urlopen(req_wik, timeout=6) as response:
            payload = json.loads(response.read().decode('utf-8'))
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
                    source_obj = {
                        "url": source_url,
                        "license": {"name": "CC BY-SA 4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"}
                    }
                    constructed_entries = [{
                        "language": {"code": "ml", "name": "Malayalam"},
                        "partOfSpeech": pos.lower(),
                        "pronunciations": [],
                        "forms": [],
                        "senses": [{"definition": definition, "tags": [], "examples": [], "quotes": [], "synonyms": [], "antonyms": [], "subsenses": []}],
                        "synonyms": [],
                        "antonyms": []
                    }]
                    result_obj = {
                        "word": word,
                        "found": True,
                        "definition": definition,
                        "part_of_speech": pos,
                        "ipa": "",
                        "romanization": "",
                        "source_url": source_url,
                        "entries": constructed_entries,
                        "source": source_obj,
                        "attribution": "Definitions provided by Wiktionary"
                    }
                    cache_definition(
                        word=word,
                        found=True,
                        definition=definition,
                        part_of_speech=pos,
                        source_url=source_url,
                        raw_json=json.dumps(result_obj, ensure_ascii=False)
                    )
                    return result_obj
    except HTTPError as e:
        if e.code != 404:
            logger.error(f"Wiktionary REST HTTP error {e.code} for '{word}'")
    except Exception as e:
        logger.error(f"Wiktionary REST API lookup failed for '{word}': {e}")

    # Not found in any API -> return unfound response
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
    res = fetch_from_api(normalized)
    if res.get("found"):
        return res

    # Sandhi Fallback (Tier 2 known suffixes & Tier 3 LLM fallback)
    logger.info(f"Direct lookup unfound for '{normalized}'. Attempting Sandhi fallback.")
    sandhi_res = backend.malayalam_sandhi.lookup_sandhi_compound(normalized, fetch_from_api)
    if sandhi_res and sandhi_res.get("found"):
        cache_definition(
            word=normalized,
            found=True,
            definition=sandhi_res.get("definition"),
            part_of_speech=sandhi_res.get("part_of_speech"),
            source_url=sandhi_res.get("source_url"),
            raw_json=json.dumps(sandhi_res, ensure_ascii=False)
        )
        return sandhi_res

    return res
