# AGENTS.md

Guidance for AI coding agents (and humans) working in this repo.

## Git workflow

- **Do all work directly on `master`.** Do not create feature branches or open
  PRs unless explicitly asked. Commit to `master` and push `master`.
- If a stray branch/PR exists, merge its content into `master` rather than
  leaving work on the branch.

## Running the app

- The app runs in Docker; code is baked into the image, so changes need a
  rebuild: `docker compose up -d --build` (add `--force-recreate` if the
  container is not replaced).
- Bump `app_version` in `app/config.py`, the `image:` tag in
  `docker-compose.yml` and the assertion in `tests/test_config.py` together.
- When `web/` assets change, bump `CACHE_VERSION` in `web/sw.js` (and the
  `?v=` query strings in `web/index.html`) so the PWA drops its cached copy.

## Tests

- `pytest` (install `requirements.txt` + `pytest httpx` in a venv).
- Format with `ruff format` before committing.

## Frontend

- `web/gl.js` is the on-device live preview (WebGL2). It mirrors `app/pipeline/grain.py` + `filters.py`; after
  changing either side run `python tools/parity/run.py` (needs numpy, opencv, Pillow and Chromium; it needs no GPU)
  and add a case for every new effect. The server is the reference. `apply_sepia()` applies its kernel to BGR, so its
  red/blue inputs are swapped versus the textbook matrix; the shader copies that on purpose.
- Shaders use `highp` only (iPhone GPUs run `mediump` as 16-bit). Target device is an iPhone 14 Pro (see `ROADMAP.md`).
- Mobile-first vanilla JS/CSS in `web/`, no build step. Keep tap targets
  ≥ 44px and check layout at phone width.
