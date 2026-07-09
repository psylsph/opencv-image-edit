// OpenCV Image Editor — Service Worker
// - Cache-first for the app shell (HTML/CSS/JS/manifest/icons)
// - Network-first (no cache) for /api/* (always fresh results)
// - Bumping CACHE_VERSION invalidates old caches.

const CACHE_VERSION = "v13";
const CACHE_NAME = `opencv-image-edit-${CACHE_VERSION}`;

const APP_SHELL = [
  "/",
  "/index.html",
  "/style.css",
  "/app.js",
  "/manifest.webmanifest",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // API requests: always go to the network, never serve a stale response.
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(fetch(event.request));
    return;
  }

  // Everything else (app shell, static assets): cache-first, fall back to network.
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        // Cache successful GET responses so the app works offline.
        if (response.ok && event.request.method === "GET") {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      });
    }).catch(() => {
      // Offline + not in cache: serve the SPA shell for navigation requests.
      if (event.request.mode === "navigate") {
        return caches.match("/index.html");
      }
    })
  );
});
