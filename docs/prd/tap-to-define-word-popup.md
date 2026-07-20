# PRD: Tap-to-Define — Word Meaning Popup for the Reading Screen

This document outlines the product requirements and technical design for the **Tap-to-Define** feature in the Akshara app. This feature extends the existing Touch-and-Say Phonics functionality to show a small popup showing the meaning of tapped words in Malayalam.

---

## 1. Overview & Goal

Currently, tapping a word on the reading screen triggers Touch-and-Say Phonics to read the word's pronunciation aloud. 
The goal of this feature is to add an additive helper layer:
- When a reader taps a word, the existing pronunciation audio continues to play as normal.
- Simultaneously, the app fetches the word's definition and displays it in a clean, non-disruptive card popup anchored near the tapped word.
- The feature is fully offline-aware: definitions are cached, and network failures or missing dictionary entries are handled gracefully without disrupting the user flow.

---

## 2. API Contract

### Backend REST Proxy Route
To prevent client-side CORS issues and manage caching centrally, the client will query the FastAPI backend:
`GET /api/define?word={malayalam_word}`

### Backend Response Schema
- **Found State (200 OK)**:
  ```json
  {
    "word": "മരം",
    "found": true,
    "definition": "വൃക്ഷം; ഭൂമിയിൽ വളരുന്ന വലിയ സസ്യം.",
    "part_of_speech": "noun",
    "source_url": "https://ml.wiktionary.org/wiki/%E0%B4%AE%E0%B4%B0%E0%B4%82",
    "attribution": "Definitions provided by Wiktionary via FreeDictionaryAPI.com"
  }
  ```
- **Not Found State (200 OK)**:
  ```json
  {
    "word": "സംയുക്തപദങ്ങൾ",
    "found": false
  }
  ```

---

## 3. Word Matching & Normalization

- **Normalization**: All words queried to the backend will be normalized to Unicode Normalization Form C (NFC) before database cache lookup or API request.
- **MVP Suffix Constraint**: Malayalam words in text often contain suffixes and inflections (e.g. വിദ്യാലയത്തിൽ). Tapping the word will search for the exact inflected form as it appears. If not found in the dictionary, it will display the graceful "Not Available" message. Stemming/lemmatization is out of scope for the initial MVP.

---

## 4. Database Caching Strategy

To minimize API traffic, prevent rate limits, and enable fast response times, results will be cached:
1. **Supabase Cache (Primary)**: If `SUPABASE_URL` and `SUPABASE_KEY` are configured in `.env`, the backend will check and populate a Supabase table.
2. **SQLite Cache (Local Fallback)**: If Supabase is unconfigured, the backend will fall back to a local SQLite database file (`cache.db`) in the workspace.

### Database Table Schema (`word_definitions`)
- `word` (TEXT, Primary Key, NFC normalized)
- `found` (BOOLEAN)
- `definition` (TEXT, Nullable)
- `part_of_speech` (TEXT, Nullable)
- `source_url` (TEXT, Nullable)
- `created_at` (TIMESTAMP/DATETIME, defaults to current time)

---

## 5. Frontend UI States

The definition popup will be dynamically positioned near the tapped word.

| State | Visual Layout / UI Elements |
|---|---|
| **Loading** | Shows a tiny, calm circular spinner inside a small anchored card popup. |
| **Found** | Displays the definition in a clean font. Shows a small, low-contrast attribution footer: *"Definitions provided by Wiktionary via FreeDictionaryAPI.com"* with a clickable link to the source Wiktionary page. |
| **Not-Found** | Displays a gentle message: *"No definition available for this word yet."* (not styled as an error). |
| **Error (Network/Server)** | Displays: *"Couldn't load definition right now."* |

---

## 6. Verification Plan

### Automated Checks
- Unit test mock requests for `/api/define` verifying cache hit and cache miss scenarios.
- Verify MIME-types and HTTP status codes for the definition endpoint.

### Manual Checks
- Tap a word to verify the popup appears near the word while audio plays.
- Verify tapping outside the popup closes it.
- Verify switching network off causes the popup to display the offline error state on uncached words, but displays cached definitions instantly.
