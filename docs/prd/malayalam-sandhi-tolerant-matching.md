# Product Requirements Document (PRD): Malayalam Sandhi-Tolerant Matching & Lookup

## 1. Problem Statement

In Malayalam, two or more words can phoneticize and join together to form a single compound string — a linguistic process known as **sandhi** (സന്ധി). 

Example:
```
മേശപ്പുറത്ത് (on the table) + ഉണ്ട് (is/exists) → മേശപ്പുറത്തുണ്ട് (is on the table)
```

Although both independent words and compound forms carry identical underlying semantic meanings, they manifest as distinct Unicode string representations (different lengths, word boundaries, and phonetic mutations at the junction point).

In **Akshara**, sandhi compound formation causes two critical failures:
1. **Live Reading Speech Comparison (Sarvam Speech-to-Text)**: A child reading text aloud may pronounce sandhi compounds either as a single combined unit (`മേശപ്പുറത്തുണ്ട്`) or pause to pronounce constituent words individually (`മേശപ്പുറത്ത്` `ഉണ്ട്`). Strict word-by-word alignment flags linguistically correct readings as mismatches.
2. **Tap-to-Define Dictionary Lookups**: FreeDictionaryAPI and Wiktionary index headwords in unjoined base forms. Tapping a sandhi-joined word in OCR'd text (e.g., `മേശപ്പുറത്തുണ്ട്`) yields "word not found", even though individual components (`മേശപ്പുറത്ത്` and `ഉണ്ട്`) possess valid dictionary entries.

---

## 2. Solution Strategy: Tiered Heuristic Approach

Building a full probabilistic/neural sandhi splitter from scratch is beyond MVP scope and yields diminishing returns. Instead, Akshara implements a pragmatic, **three-tiered engineering fallback chain**, ordering by computational efficiency and latency.

```
+-------------------------------------------------------------------------+
|                              Word Input                                 |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| Tier 1: Sandhi-Tolerant Live Speech Comparison                          |
| - Sliding window comparison over 1-2 expected tokens                    |
| - RapidFuzz normalized edit distance string matching                    |
+-------------------------------------------------------------------------+
                                     | (For Tap-to-Define Lookup Fallback)
                                     v
+-------------------------------------------------------------------------+
| Direct API / Cache Lookup                                                |
+-------------------------------------------------------------------------+
                                     | (If Not Found)
                                     v
+-------------------------------------------------------------------------+
| Tier 2: Known-Suffix Rule-Based Fallback                                |
| - Target common sandhi suffixes: ഉണ്ട്, ആണ്, എന്ന്, ഉള്ള, ഇല്ല, ഓട്, etc.   |
| - Decompose & query FreeDictionaryAPI for constituent base words        |
| - Optional: mlmorph FST morphological analyzer assistance                |
+-------------------------------------------------------------------------+
                                     | (If Tier 2 Fails)
                                     v
+-------------------------------------------------------------------------+
| Tier 3: LLM Sandhi Splitter Fallback (Last Resort)                     |
| - Tightly-scoped prompt via LLM to decompose compounds                  |
| - Asynchronous API call with cache storage (Supabase / SQLite)          |
+-------------------------------------------------------------------------+
```

---

## 3. Tier Specifications

### Tier 1 — Sandhi-Tolerant Speech Comparison (Live Reading)
- **Scope**: Live STT transcript vs. textbook expected text matching.
- **Design**:
  - Maintains a sliding window over 1 to 2 expected tokens.
  - Concatenates adjacent expected words (e.g., `W1 + W2` -> `W1W2` and phonetic variants) when comparing against spoken transcript tokens.
  - Uses `rapidfuzz` (Levenshtein ratio / normalized edit distance) with a similarity threshold of **0.80 (80%)**.
  - If spoken token matches the 2-word concatenated window better than single tokens, the reading pointer advances past **both** expected words simultaneously.
- **Guardrail**: Strict score threshold prevents false positive matches on actual mispronunciations (e.g., reading an entirely different word).

### Tier 2 — Known-Suffix Rule-Based Fallback (Tap-to-Define)
- **Scope**: Tap-to-Define word definitions when exact lookup returns `found: false`.
- **Design**:
  - Curated high-frequency Malayalam joining suffixes based on textbook material:
    - `ഉണ്ട്` (is/exists)
    - `ആണ്` (is)
    - `എന്ന്` (that/saying)
    - `ഉള്ള` (having/which is)
    - `ഇല്ല` (is not)
    - `ഓട്` (with/towards)
    - `ആയി` (became)
    - `ആക്കി` (made)
    - `ഓടെ` (with)
    - `ഉം` (and/also)
  - Applies phonetic junction rules (e.g., removing joining geminated consonants or euphonic vowels like `ത്തുണ്ട്` -> `ത്ത്` + `ഉണ്ട്`).
  - Queries `FreeDictionaryAPI` / Wiktionary for each constituent stem.
  - Displays component matches with visual distinction in the UI ("Part of this word means...").

### Tier 3 — LLM Fallback (Tap-to-Define Last Resort)
- **Scope**: Used only when Tier 2 suffix matching fails to resolve the word.
- **Contract**:
  - Prompt: *"The following is a single Malayalam string that may be a sandhi-joined compound of two or more independent words. If it is, return the individual words separated by spaces. If it is already a single independent word, return it unchanged. Respond with only the split/unchanged word(s), nothing else."*
  - Timeout: 5.0 seconds. On timeout or error, falls back gracefully to "definition unavailable".
  - **Caching Strategy**: Results (both successful splits and un-splittable tokens) are saved to Supabase & local SQLite cache under key `sandhi:<word>` or merged into `word_definitions`. Tier 3 will run **at most once per unique word across all users**.

---

## 4. Evaluation of `mlmorph`

`mlmorph` (built by SMC / Santhosh Thottingal) was evaluated via its Python FST analyzer (`mlmorph.Analyser`).
- **Findings**: `mlmorph` excels at inflectional analysis (e.g. `കേരളത്തിന്റെ` -> `കേരളം<n><genitive>`). For standard external sandhi compounds present in its transducer dictionary (e.g., `മേശപ്പുറത്തുണ്ട്`), it successfully outputs tag streams (`മേശ<n>പുറത്ത്<postp>ഉണ്ട്<aff>`).
- **Integration**: Integrated as a Tier-2-adjacent helper to complement rule-based suffix splitting before falling through to Tier 3 LLM calls.

---

## 5. Acceptance Criteria

1. **Sandhi Speech Alignment**: `മേശപ്പുറത്ത്` + `ഉണ്ട്` read as `മേശപ്പുറത്തുണ്ട്` (or vice-versa) is correctly recognized as 100% correct reading and advances both words.
2. **Mispronunciation Detection**: Genuinely incorrect speech (e.g., saying `ആന` instead of `മേശപ്പുറത്തുണ്ട്`) continues to be flagged as wrong/missing.
3. **Tap-to-Define Component UI**: Tapping a compound word like `മേശപ്പുറത്തുണ്ട്` shows partial/component definitions with visual badge `"Part of this word"`.
4. **Caching & Tier 3 Efficiency**: Tier 3 LLM calls are invoked only after Tier 2 fails, and cache hits avoid repeated LLM calls.
5. **Robust Offline/Fallback Operation**: System operates gracefully without errors if external network or LLM APIs are unreachable.

---

## 6. Open Risks & Accepted Limitations

- **Complex Sandhi Coverage**: Multi-word compounds with rare archaic transformations may not be captured by Tier 2 and rely on Tier 3 LLM availability.
- **LLM Latency**: First-time Tier 3 lookups incur LLM API roundtrip latency (~500ms - 1.5s), mitigated entirely by caching on subsequent lookups.
