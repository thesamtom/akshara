const CACHE_VERSION = 'v6';
const CACHE_NAME = `akshara-cache-${CACHE_VERSION}`;

// Assets to pre-cache on install
const PRECACHE_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/maskable-512.png',
  '/icons/apple-touch-icon.png',
  '/icons/favicon.ico',
  '/static/tesseract/tesseract.min.js',
  '/static/tesseract/worker.min.js',
  '/static/tesseract/tesseract-core.wasm.js',
  '/static/tesseract/tesseract-core-simd.wasm.js',
  '/static/tesseract/tesseract-core.wasm',
  '/static/tesseract/tesseract-core-simd.wasm',
  '/static/tesseract/tesseract-core-lstm.wasm.js',
  '/static/tesseract/tesseract-core-simd-lstm.wasm.js',
  '/static/tesseract/tesseract-core-lstm.wasm',
  '/static/tesseract/tesseract-core-simd-lstm.wasm',
  '/static/tesseract/tessdata/mal.traineddata.gz'
];

// Install Event
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Pre-caching offline assets...');
        return cache.addAll(PRECACHE_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate Event
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('[Service Worker] Cleaning up old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Event
self.addEventListener('fetch', event => {
  // Only handle GET requests
  if (event.request.method !== 'GET') {
    event.respondWith(fetch(event.request));
    return;
  }

  const url = new URL(event.request.url);

  // Direct bypass for REST API and WebSocket connection requests
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) {
    event.respondWith(fetch(event.request));
    return;
  }

  // Cache-First with Network Fallback strategy for static assets
  event.respondWith(
    caches.match(event.request)
      .then(cachedResponse => {
        if (cachedResponse) {
          return cachedResponse;
        }

        // Fallback to network
        return fetch(event.request).then(networkResponse => {
          // If valid response, cache it dynamically for later
          if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
            const responseToCache = networkResponse.clone();
            caches.open(CACHE_NAME).then(cache => {
              cache.put(event.request, responseToCache);
            });
          }
          return networkResponse;
        });
      }).catch(err => {
        console.error('[Service Worker] Fetch failed:', err);
        // Return a generic offline fallback page for navigation requests if both cache and network fail
        if (event.request.mode === 'navigate') {
          return caches.match('/').then(res => {
            return res || Response.error();
          });
        }
        throw err;
      })
  );
});
