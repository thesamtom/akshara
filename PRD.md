# PWA PRD: Converting Akshara into a Complete PWA

This document outlines the product requirements, offline capability architecture, asset caching strategies, and validation criteria for converting the Akshara app into a Progressive Web App (PWA).

---

## 1. Offline Capabilities Confirmation

Based on the actual codebase structure (single-page frontend `index.html` + FastAPI `server.py` backend), the application features are classified as follows for offline operation:

| Feature | Sub-system / Route | Offline Capable? | Technical Reason / Dependency |
|---|---|---|---|
| **App Shell Loading** | `/`, `/index.html` | **Yes** | Static web assets can be pre-cached in Cache Storage. |
| **Image Upload / UI Interaction** | Client-side JS | **Yes** | Standard browser event handlers do not require network. |
| **Malayalam OCR Extraction** | Tesseract.js (Client-side) | **Yes** | *Crucial*: Only if `tesseract.min.js`, the worker script, core WASM binaries, and the Malayalam trained data are cached locally. |
| **Text-to-Speech Playback** | `/api/tts` (REST Proxy) | **No** | Requires forwarding request to Sarvam's AI endpoint via FastAPI backend over the internet. |
| **Live Reading STT Captions** | `/ws/reading` (WebSocket Proxy) | **No** | WebSockets require an active socket connection over TCP/IP and cannot be intercepted or served from Cache Storage. |

---

## 2. Pre-cached Offline OCR Assets

To enable full offline OCR capability, Tesseract.js v5 must run entirely locally. The following assets must be pre-downloaded, stored on the server, and cached inside the PWA Service Worker:

1.  **Main Interface Script**: `/static/tesseract/tesseract.min.js`
    - Entry point loaded by the client.
2.  **Worker Thread Script**: `/static/tesseract/worker.min.js`
    - Spawns the web worker handling image text reading processes.
3.  **WASM Core Wrappers**:
    - `/static/tesseract/tesseract-core.wasm.js` (JavaScript wrapper for WebAssembly core)
    - `/static/tesseract/tesseract-core-simd.wasm.js` (SIMD-enabled JavaScript wrapper for WebAssembly core)
    - `/static/tesseract/tesseract-core-lstm.wasm.js` (LSTM neural-net core wrapper)
    - `/static/tesseract/tesseract-core-simd-lstm.wasm.js` (SIMD-enabled LSTM neural-net core wrapper)
4.  **WASM Binary Assets**:
    - `/static/tesseract/tesseract-core.wasm` (Compiled ASR/OCR core binaries)
    - `/static/tesseract/tesseract-core-simd.wasm` (SIMD optimized binary compilation)
    - `/static/tesseract/tesseract-core-lstm.wasm` (Compiled LSTM neural-net core binaries)
    - `/static/tesseract/tesseract-core-simd-lstm.wasm` (SIMD optimized LSTM neural-net core binaries)
5.  **Gzipped Malayalam Trained Data**: `/static/tesseract/tessdata/mal.traineddata.gz`
    - Malayalam character recognition database (~11.6 MB compressed). Must be in `.traineddata.gz` format as expected by Tesseract.js.

---

## 3. Service Worker Caching Strategy

The service worker (`sw.js`) will implement the following caching policies:

| Asset Group | Caching Strategy | Justification |
|---|---|---|
| **App Shell** (`index.html`, inline CSS, custom web fonts) | **Cache-First with Network Fallback** | Minimizes load times, guarantees instant display when offline, and falls back to network only when cache is missing. |
| **Offline OCR Assets** (Tesseract files, `mal.traineddata.gz`) | **Cache-First** | These are large, immutable library resources. Caching them permanently prevents repeated high-bandwidth downloads. |
| **PWA Icons & Manifest** (`manifest.json`, `icons/` folder) | **Cache-First** | Immutable brand assets that are loaded once by the OS/browser shell. |
| **FastAPI REST API endpoints** (`/api/tts`, `/api/health`) | **Network-Only** | Dynamically processed requests that contact cloud systems. Must fail immediately offline without displaying stale audio data. |
| **FastAPI WebSockets** (`/ws/reading`) | **Bypass Cache** | Standard behavior (handled natively by the browser since WebSockets are outside the fetch lifecycle), ensuring clean real-time proxy connections. |

---

## 4. iOS-Specific Requirements

To satisfy Apple's strict PWA framework parameters, we must implement:
1.  **Touch Icons**: An `apple-touch-icon.png` (180x180 px) linked inside the HTML `<head>`.
2.  **iOS Safari Web App Capability**:
    - `<meta name="apple-mobile-web-app-capable" content="yes">`
    - `<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">`
3.  **Persistence Safety**: Check that `mal.traineddata.gz` remains cached. iOS has a history of evicting caches under low-storage alerts; we will catch load errors client-side and display a descriptive warning if assets need re-downloading.

---

## 5. Acceptance Criteria (Verification Checklist)

*   **PWA Installability**: Passes the Lighthouse PWA audit (returns green PWA indicators, valid manifest, active service worker).
*   **Offline App Shell**: Reloading `http://localhost:8000` under DevTools "Offline" state loads the UI without a browser network error.
*   **Offline OCR Conversion**: Under "Offline" state, uploading an image and clicking "Convert to Text" finishes successfully and outputs extracted Malayalam text.
*   **Graceful Offline UI**: When offline, the "Play Text" and "Start Reading" buttons are disabled and grayed out, displaying a clear status text saying "Internet connection required".
*   **Cache Invalidation**: Changing the cache version variable in `sw.js` activates the new cache version and cleans up the old cache storage on page reload.
