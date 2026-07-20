"""Malayalam Sandhi (Word-Joining) Matching & Lookup Engine.

Provides:
- Tier 1: Sandhi-tolerant live-reading speech comparison using RapidFuzz sliding windows.
- Tier 2: Known-suffix rule-based splitter and mlmorph morphological analyzer integration.
- Tier 3: LLM fallback splitter for complex sandhi compounds.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import urllib.request
import urllib.parse
from typing import Callable

try:
    from rapidfuzz import fuzz, distance
except ImportError:
    fuzz = None
    distance = None

try:
    from mlmorph import Analyser
    _MLMORPH_ANALYSER = Analyser()
except Exception:
    _MLMORPH_ANALYSER = None

logger = logging.getLogger("akshara.sandhi")

# Tier 2 Curated Sandhi Suffix Rules
# Format: (suffix, list of transformation rules mapping target suffix to stem mutation)
# Transformations: (replacement_suffix, stem_ending_fix, second_part_default)
COMMON_SANDHI_SUFFIX_RULES = [
    # 1. ഉണ്ട് (is/exists)
    ("ഉണ്ട്", [
        ("ത്തുണ്ട്", "ത്ത്", "ഉണ്ട്"),   # മേശപ്പുറത്തുണ്ട് -> മേശപ്പുറത്ത് + ഉണ്ട്
        ("ന്തുണ്ട്", "ന്ത്", "ഉണ്ട്"),   # വന്നിട്ടുണ്ട് -> വന്നിട്ട് / വന്ന് + ഉണ്ട്
        ("ന്യുണ്ട്", "ന്", "ഉണ്ട്"),
        ("യുണ്ട്", "ി", "ഉണ്ട്"),     # അവിടെയുണ്ട് -> അവിടെ + ഉണ്ട് / വഴിയിമുണ്ട്
        ("യുണ്ട്", "െ", "ഉണ്ട്"),
        ("വുണ്ട്", "വ്", "ഉണ്ട്"),     # വു -> വ് + ഉണ്ട്
        ("റുണ്ട്", "റ്", "ഉണ്ട്"),
        ("കൊണ്ട്", "കൊണ്ട്", "ഉണ്ട്"),
        ("മുണ്ട്", "ം", "ഉണ്ട്"),     # കാര്യമുണ്ട് -> കാര്യം + ഉണ്ട്
        ("ഉണ്ട്", "", "ഉണ്ട്"),      # fallback direct strip
    ]),
    # 2. ആണ് (is)
    ("ആണ്", [
        ("ത്താണ്", "ത്ത്", "ആണ്"),    # അതാണ് -> അത് / അത്താണ്
        ("താണ്", "ത്", "ആണ്"),       # ഇതാണ് -> ഇത് + ആണ്
        ("വാണ്", "വ്", "ആണ്"),       # ജീവനാണ് -> ജീവൻ + ആണ്
        ("ഡാണ്", "ഡ്", "ആണ്"),
        ("രാണ്", "ർ", "ആണ്"),
        ("മാണാണ്", "മം", "ആണ്"),
        ("മാണ്", "ം", "ആണ്"),       # സത്യമാണ് -> സത്യം + ആണ്
        ("യാണാണ്", "യം", "ആണ്"),
        ("യാണ്", "", "ആണ്"),        # വലിയതാണ് / കുഞ്ഞാണ് -> കുട്ടി + ആണ്
        ("ആണ്", "", "ആണ്"),
    ]),
    # 3. എന്ന് (that / saying)
    ("എന്ന്", [
        ("മെന്ന്", "ം", "എന്ന്"),     # വരുമെന്ന് -> വരും + എന്ന്
        ("ത്തെയെന്ന്", "ത്തെ", "എന്ന്"),
        ("തെന്ന", "ത്", "എന്ന്"),
        ("യെന്ന്", "", "എന്ന്"),      # പോയെന്ന് -> പോയി + എന്ന്
        ("എന്ന്", "", "എന്ന്"),
    ]),
    # 4. ഉള്ള (having / which is)
    ("ഉള്ള", [
        ("വുള്ള", "വ്", "ഉള്ള"),     # കഴിവുള്ള -> കഴിവ് + ഉള്ള
        ("മുള്ള", "ം", "ഉള്ള"),      # ഗുണമുള്ള -> ഗുണം + ഉള്ള
        ("യുള്ള", "", "ഉള്ള"),       # ഭംഗിയുള്ള -> ഭംഗി + ഉള്ള
        ("ത്തുള്ള", "ത്ത്", "ഉള്ള"),
        ("ഉള്ള", "", "ഉള്ള"),
    ]),
    # 5. ഇല്ല (is not)
    ("ഇല്ല", [
        ("ട്ടില്ല", "ട്ട്", "ഇല്ല"),   # വന്നിട്ടില്ല -> വന്നിട്ട് + ഇല്ല
        ("ണ്ടില്ല", "ണ്ട്", "ഇല്ല"),   # കണ്ടില്ല -> കണ്ട് + ഇല്ല
        ("വില്ല", "വ്", "ഇല്ല"),     # വരില്ല -> വരി / വന്ന് + ഇല്ല
        ("യില്ല", "", "ഇല്ല"),       # അറിയില്ല -> അറിയുക + ഇല്ല
        ("ഇല്ല", "", "ഇല്ല"),
    ]),
    # 6. ഓട് (with / towards)
    ("ഓട്", [
        ("നോട്", "ൻ", "ഓട്"),       # അവനോട് -> അവൻ + ഓട്
        ("നോട്", "ന്", "ഓട്"),
        ("ളോട്", "ൾ", "ഓട്"),       # അവളോട് -> അവൾ + ഓട്
        ("രോട്", "ർ", "ഓട്"),       # അവരോട് -> അവർ + ഓട്
        ("യോട്", "", "ഓട്"),        # കുട്ടിയോട് -> കുട്ടി + ഓട്
        ("ഓട്", "", "ഓട്"),
    ]),
    # 7. ആയി (became)
    ("ആയി", [
        ("തായി", "ത്", "ആയി"),       # വലിയതായി -> വലിയത് + ആയി
        ("മായി", "ം", "ആയി"),       # ശുദ്ധമായി -> ശുദ്ധം + ആയി
        ("യായി", "", "ആയി"),
        ("ആയി", "", "ആയി"),
    ]),
    # 8. ആക്കി (made)
    ("ആക്കി", [
        ("മാക്കി", "ം", "ആക്കി"),    # ശുദ്ധമാക്കി -> ശുദ്ധം + ആക്കി
        ("താക്കി", "ത്", "ആക്കി"),
        ("യാക്കി", "", "ആക്കി"),
        ("ആക്കി", "", "ആക്കി"),
    ]),
    # 9. ഓടെ (with)
    ("ഓടെ", [
        ("ത്തോടെ", "ത്ത", "ഓടെ"),     # സന്തോഷത്തോടെ -> സന്തോഷം + ഓടെ
        ("ത്തോടെ", "ത്ത്", "ഓടെ"),
        ("മോടെ", "ം", "ഓടെ"),
        ("ഓടെ", "", "ഓടെ"),
    ]),
    # 10. ഉം (and / also)
    ("ഉം", [
        ("വും", "വ്", "ഉം"),        # കാറ്റും വെളിച്ചവും -> വെളിച്ചം + ഉം
        ("യും", "", "ഉം"),         # തീയും -> തീ + ഉം
        ("തും", "ത്", "ഉം"),
        ("ഉം", "", "ഉം"),
    ]),
]


def normalize_text(text: str) -> str:
    """Normalize Malayalam unicode characters and remove zero-width joiners/spaces."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text.strip())
    # Remove zero-width non-joiner (U+200C), zero-width joiner (U+200D), BOM
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", normalized)
    return cleaned


def tokenize_words(text: str) -> list[str]:
    """Tokenize Malayalam text into sanitized word tokens."""
    clean = normalize_text(text)
    # Split on whitespace or non-Malayalam/non-word characters
    words = re.split(r"[^\w\u0D00-\u0D7F]+", clean)
    tokens = []
    for w in words:
        w_sanitized = re.sub(r"[^\w\u0D00-\u0D7F]", "", w)
        if w_sanitized:
            tokens.append(w_sanitized)
    return tokens


# ============================================================================
# Tier 1 — Sandhi-Tolerant Live Speech Alignment Engine
# ============================================================================

def string_similarity(s1: str, s2: str) -> float:
    """Calculates similarity score (0.0 to 1.0) between two Malayalam strings."""
    s1_norm = normalize_text(s1)
    s2_norm = normalize_text(s2)
    if not s1_norm or not s2_norm:
        return 1.0 if s1_norm == s2_norm else 0.0
    if s1_norm == s2_norm:
        return 1.0

    if fuzz is not None:
        # RapidFuzz similarity ratio (0 to 100) converted to float
        return fuzz.ratio(s1_norm, s2_norm) / 100.0

    # Custom Levenshtein fallback if RapidFuzz is missing
    m, n = len(s1_norm), len(s2_norm)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1_norm[i - 1] == s2_norm[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def create_sandhi_variants(w1: str, w2: str) -> list[str]:
    """Generates common sandhi phonetic compound variants for 2 adjacent words."""
    w1_clean = normalize_text(w1)
    w2_clean = normalize_text(w2)
    variants = [
        f"{w1_clean}{w2_clean}",
        f"{w1_clean} {w2_clean}",
    ]
    # Handle euphonic vowel joins: e.g. മേശപ്പുറത്ത് + ഉണ്ട് -> മേശപ്പുറത്തുണ്ട്
    if w1_clean.endswith("ത്ത്") and w2_clean.startswith("ഉ"):
        variants.append(w1_clean[:-2] + "ത്തു" + w2_clean[1:])
    elif w1_clean.endswith("്") and w2_clean.startswith("ഉ"):
        variants.append(w1_clean[:-1] + "ു" + w2_clean[1:])
    elif w1_clean.endswith("്") and w2_clean.startswith("ആ"):
        variants.append(w1_clean[:-1] + "ാ" + w2_clean[1:])
    elif w1_clean.endswith("്") and w2_clean.startswith("ഇ"):
        variants.append(w1_clean[:-1] + "ി" + w2_clean[1:])
    elif w1_clean.endswith("ം") and w2_clean.startswith("ഉ"):
        variants.append(w1_clean[:-1] + "മു" + w2_clean[1:])
    return variants


def align_reading_sandhi(
    expected_text: str,
    spoken_text: str,
    similarity_threshold: float = 0.80
) -> dict:
    """Performs sequence alignment of spoken transcript against expected textbook text,

    handling 1-2 word sliding windows with Sandhi tolerance.
    """
    expected_tokens = tokenize_words(expected_text)
    spoken_tokens = tokenize_words(spoken_text)

    m = len(expected_tokens)
    n = len(spoken_tokens)

    if m == 0:
        return {
            "expected_tokens": [],
            "spoken_tokens": spoken_tokens,
            "expected_status": [],
            "spoken_status": ["extra"] * n,
            "accuracy": 0,
            "correct_count": 0,
            "wrong_count": n,
            "missing_count": 0
        }

    if n == 0:
        return {
            "expected_tokens": expected_tokens,
            "spoken_tokens": [],
            "expected_status": ["missing"] * m,
            "spoken_status": [],
            "accuracy": 0,
            "correct_count": 0,
            "wrong_count": 0,
            "missing_count": m
        }

    expected_status = ["missing"] * m
    spoken_status = ["extra"] * n

    i = 0  # Index into expected_tokens
    j = 0  # Index into spoken_tokens

    while i < m and j < n:
        exp_w = expected_tokens[i]
        spk_w = spoken_tokens[j]

        sim_1to1 = string_similarity(exp_w, spk_w)

        sim_2exp = 0.0
        if i + 1 < m:
            w1, w2 = expected_tokens[i], expected_tokens[i + 1]
            variants = create_sandhi_variants(w1, w2)
            sim_2exp = max(string_similarity(var, spk_w) for var in variants)

        sim_2spk = 0.0
        if j + 1 < n:
            s1, s2 = spoken_tokens[j], spoken_tokens[j + 1]
            variants = create_sandhi_variants(s1, s2)
            sim_2spk = max(string_similarity(exp_w, var) for var in variants)

        # Exact 1-to-1 match
        if sim_1to1 == 1.0:
            expected_status[i] = "correct"
            spoken_status[j] = "correct"
            i += 1
            j += 1
            continue

        # 2 Expected -> 1 Spoken (Sandhi compound in spoken transcript)
        if sim_2exp >= similarity_threshold and sim_2exp > sim_1to1:
            expected_status[i] = "correct"
            expected_status[i + 1] = "correct"
            spoken_status[j] = "correct"
            i += 2
            j += 1
            continue

        # 1 Expected -> 2 Spoken (Spoken split sandhi compound)
        if sim_2spk >= similarity_threshold and sim_2spk > sim_1to1:
            expected_status[i] = "correct"
            spoken_status[j] = "correct"
            spoken_status[j + 1] = "correct"
            i += 1
            j += 2
            continue

        # 1-to-1 Fuzzy match
        if sim_1to1 >= similarity_threshold:
            expected_status[i] = "correct"
            spoken_status[j] = "correct"
            i += 1
            j += 1
            continue

        # Flag mismatch & advance greedy
        expected_status[i] = "wrong"
        spoken_status[j] = "wrong"
        i += 1
        j += 1

    # Calculate statistics
    correct_count = expected_status.count("correct")
    wrong_count = expected_status.count("wrong") + spoken_status.count("extra")
    missing_count = expected_status.count("missing")
    accuracy = round((correct_count / m) * 100) if m > 0 else 0

    return {
        "expected_tokens": expected_tokens,
        "spoken_tokens": spoken_tokens,
        "expected_status": expected_status,
        "spoken_status": spoken_status,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "missing_count": missing_count,
    }


# ============================================================================
# Tier 2 — Known-Suffix Rule-Based Splitter & mlmorph Morphological Analyzer
# ============================================================================

def split_known_suffixes(word: str) -> list[tuple[str, str]]:
    """Generates candidate (headword_stem, suffix_word) split pairs for a given Malayalam word.

    Uses Tier 2 curated suffix rules and mlmorph FST transducer if available.
    """
    word_norm = normalize_text(word)
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    # 1. Curated Suffix Matching
    for suffix_name, transformations in COMMON_SANDHI_SUFFIX_RULES:
        for end_pat, stem_fix, second_part in transformations:
            if end_pat and word_norm.endswith(end_pat):
                stem = word_norm[:-len(end_pat)] + stem_fix
                stem = normalize_text(stem)
                second = normalize_text(second_part)
                if stem and second and (stem, second) not in seen and len(stem) >= 2:
                    seen.add((stem, second))
                    candidates.append((stem, second))

    # 2. mlmorph FST Transducer Evaluation
    if _MLMORPH_ANALYSER is not None:
        try:
            analyses = _MLMORPH_ANALYSER.analyse(word_norm)
            for parse_str, score in analyses:
                # Example parse_str: 'മേശ<n>പുറത്ത്<postp>ഉണ്ട്<aff>'
                tokens = re.split(r"(<[^>]+>)", parse_str)
                stems = [t for t in tokens if t and not t.startswith("<")]
                if len(stems) >= 2:
                    stem1, stem2 = normalize_text(stems[0]), normalize_text(stems[1])
                    if len(stem1) >= 2 and len(stem2) >= 2 and (stem1, stem2) not in seen:
                        seen.add((stem1, stem2))
                        candidates.append((stem1, stem2))
        except Exception as err:
            logger.debug(f"mlmorph analysis failed for '{word_norm}': {err}")

    return candidates


# ============================================================================
# Tier 3 — LLM Fallback Sandhi Splitter
# ============================================================================

def llm_sandhi_split(word: str) -> list[str]:
    """Tier 3 LLM fallback call to decompose complex Malayalam sandhi compound into words."""
    key = os.environ.get("SARVAM_STT_API_KEY") or os.environ.get("SARVAM_TTS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        logger.info("No API key configured for Tier 3 LLM sandhi splitter.")
        return [word]

    prompt = (
        f"The following is a single Malayalam string that may be a sandhi-joined compound of two or more independent words. "
        f"If it is, return the individual words separated by spaces. If it is already a single independent word, return it unchanged. "
        f"Respond with only the split/unchanged word(s), nothing else.\n\n"
        f"Word: {word}"
    )

    try:
        # 1. Try Sarvam Chat / Text Model if key is available
        sarvam_key = os.environ.get("SARVAM_STT_API_KEY") or os.environ.get("SARVAM_TTS_API_KEY")
        if sarvam_key:
            req_data = json.dumps({
                "model": "sarvam-2b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.1
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.sarvam.ai/v1/chat/completions",
                data=req_data,
                headers={
                    "api-subscription-key": sarvam_key,
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"].strip()
                split_words = tokenize_words(content)
                if split_words:
                    return split_words
    except Exception as e:
        logger.warning(f"Tier 3 LLM sandhi split failed for '{word}': {e}")

    return [word]


# ============================================================================
# Sandhi Dictionary Lookup Orchestrator
# ============================================================================

def lookup_sandhi_compound(word: str, lookup_fn: Callable[[str], dict]) -> dict | None:
    """Orchestrates Tier 2 and Tier 3 sandhi fallbacks for dictionary lookups."""
    word_norm = normalize_text(word)

    # --- Tier 2: Known Suffix Splitter ---
    candidates = split_known_suffixes(word_norm)
    for w1, w2 in candidates:
        res1 = lookup_fn(w1)
        res2 = lookup_fn(w2)

        # If at least one part is a valid dictionary entry
        if (res1 and res1.get("found")) or (res2 and res2.get("found")):
            components = []
            def_parts = []

            if res1 and res1.get("found"):
                components.append(res1)
                def_parts.append(f"[{res1['word']}] {res1.get('definition', '')}")
            else:
                components.append({"word": w1, "found": False})

            if res2 and res2.get("found"):
                components.append(res2)
                def_parts.append(f"[{res2['word']}] {res2.get('definition', '')}")
            else:
                components.append({"word": w2, "found": False})

            combined_def = "Part of this word: " + "; ".join(def_parts)
            return {
                "word": word_norm,
                "found": True,
                "partial_match": True,
                "tier": 2,
                "definition": combined_def,
                "part_of_speech": "compound",
                "components": components,
                "attribution": "Component definitions via Wiktionary (Sandhi fallback)"
            }

    # --- Tier 3: LLM Fallback ---
    split_tokens = llm_sandhi_split(word_norm)
    if len(split_tokens) > 1 and split_tokens != [word_norm]:
        components = []
        def_parts = []
        found_any = False

        for token in split_tokens:
            res = lookup_fn(token)
            if res and res.get("found"):
                found_any = True
                components.append(res)
                def_parts.append(f"[{res['word']}] {res.get('definition', '')}")
            else:
                components.append({"word": token, "found": False})

        if found_any:
            combined_def = "Part of this word: " + "; ".join(def_parts)
            return {
                "word": word_norm,
                "found": True,
                "partial_match": True,
                "tier": 3,
                "definition": combined_def,
                "part_of_speech": "compound",
                "components": components,
                "attribution": "Component definitions via LLM Sandhi decomposition & Wiktionary"
            }

    return None
