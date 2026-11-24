# ROADMAP

Where the app goes after 1.9.x. Three themes, in the order they should land:

1. **Move image processing onto the phone** (WebGL first, then in-browser ML).
2. **Replace the background with another image.**
3. **Borrow what works from `vid-ed-mobile`**: techniques and the editor UI.

Themes 1 and 3 overlap: the live-preview shader and the matte-caching idea are the same piece of work, so
Phase 1 delivers both. Theme 2 sits on top of them.

## Target device: iPhone 14 Pro (decided)

Everything is designed for, and checked on, an **iPhone 14 Pro** (A16, 6 GB RAM, 120 Hz ProMotion screen at 3× DPR,
Dynamic Island, Safari/installed PWA on iOS 16 or later). Other phones are best-effort; desktop is for development.
What that means in practice:

- **WebGL2 is the baseline** (Safari 15+). No WebGPU dependency: Safari only has it from iOS 26, so it's an optional
  accelerator that's probed at runtime, never required.
- **Big photos.** Default camera shots are 12 MP HEIC; 48 MP ProRAW/HEIF can appear. Safari caps canvas area (assume
  ~16.7 MP) and kills tabs under memory pressure, so decode once, keep one GPU texture, and downscale for the preview.
  Uploads to the server are already capped at 1536 px, so the client should downscale before upload (smaller, faster).
- **HEIC is decoded natively by Safari** (`createImageBitmap`), so on-device processing doesn't need a decoder library.
  Desktop Chrome can't decode HEIC, which is fine: the app falls back to the server path there.
- **PWA on iOS:** installed web apps have their own storage and no background execution; keep models/backgrounds in
  Cache Storage/IndexedDB, expect eviction after weeks of disuse, and save work often (iOS may reload the page).
- **UI:** safe-area insets (Dynamic Island on top, home indicator at the bottom), 16 px inputs so Safari doesn't zoom,
  ≥ 44 px targets, `touch-action` set so paint/tap doesn't fight page scroll, `100dvh` not `100vh`.
- **Real-device checks** go in `IPHONE-CHECKLIST.md` (as vid-ed-mobile does); simulators and headless Chrome
  (SwiftShader) don't find iOS memory or GPU-precision issues.
- GPU maths: use `highp` floats in fragment shaders (Apple GPUs honour it; `mediump` on iOS is 16-bit and bands badly).

## Where we are

- Server does everything: `POST /api/v1/process` runs decode → matting (U2NetP / BiRefNet Tiny via ONNX Runtime) →
  grain → upscale → filters → PNG round-trip as base64. Every slider change means a full upload + CPU pass.
- Background modes are only `blur` and `remove` (`app/api/schemas.py`); the mask is computed per request and only
  returned as a debug image.
- UI is a long scrolling page with accordion groups (`web/index.html`, `web/app.js`, ~1300 lines, no build step,
  `CACHE_VERSION` in `web/sw.js` bumped by hand).
- Container is limited to 3 CPUs / 8 GB, so the server is the bottleneck whenever more than one thing happens.

## Phase 1: Live preview on the device (WebGL)

**Goal:** sliders react instantly, with no network, on the phone. The server is only needed for the ML steps.

What moves to the phone, and what doesn't:

| Step | Today | Phase 1 | Notes |
|---|---|---|---|
| Rotate / flip | server (`/process` input) | **WebGL / canvas** | Trivial; do it first |
| Brightness, contrast, saturation, sepia, grayscale, vignette | `filters.py` | **WebGL fragment shader** | Pure per-pixel maths |
| Sharpness / unsharp mask | `filters.py` | **WebGL** (3×3 / 5×5 taps) | |
| Film grain | `grain.py` (Rec.601 luma-weighted) | **WebGL** (hash noise × luma weight) | Not bit-identical to the server; fine for preview, see parity below |
| Blur background / blur area | `background.py` | **WebGL** composite: sharp × mask + blurred × (1 − mask) | Needs the mask (see "Matte caching") |
| Before/after slider | server renders a comparison PNG | **CSS clip over two canvases** | Drops `comparison.py` from the hot path |
| Matting, SAM click-select, LaMa inpaint, anime/painting styles | server | stays on server | ML, heavy |
| 2×/4× upscale | server | stays on server (Phase 4 maybe) | |

Work items:

- [x] `web/gl.js`: one WebGL2 renderer (texture upload, ping-pong passes, `readPixels`/`toBlob` for export). No build step needed. *(1.10.0: live preview of grain, colour adjustments, sharpness, vignette and sepia)*
- [x] Shader for grain → colour adjustments → sharpness → vignette → sepia, in the server's order; separable gaussian passes for blur.
- [ ] Preview at screen resolution; **export re-runs the same shader at full size** (clamp to `MAX_TEXTURE_SIZE`, tile if
      needed; iOS Safari caps canvas area around 16 MP).
- [ ] Context-loss handling and a **server fallback** when WebGL2 is missing or the context is lost: the existing
      `/process` path stays and is the reference implementation.
- [ ] Settings → uniforms mapping lives in one small module, mirrored from `collectSettings()` so server and client agree.
- [ ] `CACHE_VERSION` bump + service-worker precache for the new files.
- [ ] Client-side downscale before upload (iPhone photos are 12–48 MP; the server caps at 1536 px anyway). Needs care: paint/SAM masks are sent in image coordinates.
- [x] **Parity check** (`tools/parity/run.py`, 14 cases, all within tolerance in 1.10.0), borrowed from vid-ed-mobile's `tools/parity`: render a handful of fixed images through the server
      and through headless Chrome (SwiftShader), compare with a tolerance. Grain is compared statistically (mean/variance),
      not per pixel.

**Matte caching** (the key idea from vid-ed-mobile): the expensive part is the mask, and it doesn't change when the user
moves a blur slider or swaps a background. So:

- [x] New `POST /api/v1/matte` returns just the mask (8-bit grayscale PNG at the server's processing size). *(1.11.0; the client caches it per photo + model)*
- [x] Client keeps the mask and does all compositing locally (blur and remove-preview, strength live); the server is called again only if the image, model or
      painted selection changes.
- [ ] `/process` stays for "full server render" and as the fallback.

**Done when:** dragging any colour/grain/vignette/blur slider updates the preview at ≥ 30 fps on a mid-range phone with
no request in flight, and the downloaded PNG matches the server render within tolerance.

**Known gaps after 1.11.0:** painted-background selections (Paint BG) still wait for Process; the first matte with
BiRefNet Tiny takes seconds on this CPU (U2NetP is ~0.7 s), so the badge shows "Finding subject…" meanwhile.
Process now reuses the matte the preview already fetched (1.12.1), so the model runs once per photo.

## Phase 2: Replace the background with another image

**Goal:** pick a second photo (or colour, or gradient) and put the subject in front of it.

Server contract (mirrors vid-ed-mobile's `bg.mode`):

- [x] Extend `BackgroundRequest.mode` to `blur | remove | color | image` (keep `remove` = transparent).
- [x] `POST /api/v1/process` takes an optional `background_image` upload (same size/type limits and HEIC/EXIF handling
      as the main image, via `read_image_bytes` / `decode_to_bgr`).
- [x] `apply_background_replace(img, mask, ...)` in `app/pipeline/background.py`: cover-fit the new background to the
      subject photo, then alpha-composite. Tests with synthetic masks, like the existing background tests.
- [x] Bad inputs return clear errors (unreadable background, absurd dimensions); no silent fallback to blur.

Making it look right (this is what separates a pasted-on cut-out from a good one; all of it is cheap maths):

- [x] **Edge tools, part 1**: soften edge and edge in/out (1.12.0; same maths in `refine_matte()` and the shader).
- [ ] **Edge tools, part 2**: remove colour fringe (desaturate edge pixels toward the new background) and an optional soft
      shadow under the subject, as vid-ed-mobile's *Edge & shadow*.
- [ ] **Foreground colour estimation** near the edge so the old background doesn't bleed through as a halo.
- [ ] **Match the subject to the scene**: optional "harmonise" (shift the subject's mean colour / brightness a little
      toward the new background).
- [x] Background controls: zoom and position sliders (1.12.0).
- [ ] Drag-to-position on the photo, flip, optional light blur on the replacement for depth of field.
- [ ] Order of operations: colour filters apply to the *subject* before compositing (as in vid-ed-mobile), so a replaced
      background keeps its own colours; grain goes over the whole result so the join disappears.

UI:

- [x] Background group has **Blur · Remove · Colour · Photo** as a segmented control; Photo takes one picked image
      (kept in memory for now).
- [ ] Recent backgrounds grid stored in the browser (IndexedDB), so a photo survives a reload.
- [x] Live compositing runs in the Phase 1 shader using the cached matte, so swapping backgrounds and dragging the
      edge sliders is instant. Server render is only for the final export when the browser can't do it.
- [ ] Presets: a *Swap background* preset (a preset can't carry the photo, so it would just switch to Photo mode).

**Done when:** pick photo A, pick photo B, get A's subject in front of B with a clean edge in under a second after the
first matte, and the PNG downloads at full resolution.

## Phase 3: Editor UI from vid-ed-mobile

The user prefers vid-ed-mobile's UI. What it does that this app doesn't:

| vid-ed-mobile | opencv-image-edit now | Proposal |
|---|---|---|
| Fixed full-height column: header, **stage** (takes free space), optional tool panel (max 42 dvh, scrolls inside), **bottom tool bar** that scrolls sideways. The page never scrolls. | Long scrolling page with accordion groups; bottom nav has Edit / Process / Result | Make the photo the stage; tools become a bottom bar (Background · Adjust · Grain · Blur area · Remove object · Upscale); one panel open at a time |
| Instant live preview on the stage | Press *Process*, wait, view result tab | Phase 1 makes this possible; Process becomes "Apply server steps" only for ML tools |
| Undo / redo | Only mask-stroke undo | Settings-level history (like `history.js`) |
| Autosave projects, project list | Nothing persists | Save original + settings + background pick in IndexedDB; reopen where you left off |
| Sheet (bottom modal) for export | Result tab | Export sheet: format (PNG/JPEG/WebP), scale, quality |
| Svelte 5 + Vite, vitest | Vanilla JS, no build | See decision below |
| Safe-area insets, 44 px targets, `touch-action`, 16 px inputs | Partly done (tap targets ≥ 44 px) | Adopt `--safe-top/--safe-bottom-lite`, 16 px inputs (stops iOS zoom), and the same tokens (`--accent`, `--surface-2`, …) |
| `reportClientError` posts webgl/shader failures to the server log | None | Add; essential once WebGL is in play |
| `IPHONE-CHECKLIST.md` for things only a real phone can confirm | None | Add one; it matters more once rendering is client-side |

Techniques from vid-ed-mobile that **do** map across: the packed-matte idea (→ cached matte), the fragment-shader
compositor, the edge tools, shader/server parity testing, quality tiers (Fast = U2NetP, Best = BiRefNet), client error
reporting, and the editor layout. The ones that **don't** (timeline, ffmpeg graph, job queue, drafts, transitions,
speed/reverse, music, accounts, backups) are video- or multi-user-specific and stay out.

**Decision needed: keep vanilla JS or move to Svelte 5 + Vite?** Recommendation: *move*, but only at the start of
Phase 3, after Phase 1 proves the shader module. `app.js` is already 1300 lines of imperative DOM code and the editor
layout wants reactive state (undo, panels, live preview). A build step costs a Node stage in the Dockerfile and a
break from the "no build step" rule in `AGENTS.md`; update that file if we go ahead. If we'd rather not, the same layout
can be done in vanilla JS with a small state store; it just takes longer.

Work items:

- [ ] Stage-first layout and bottom tool bar; keep the existing dark theme tokens.
- [ ] Tool panels reuse the existing controls (move, don't rewrite).
- [ ] Undo/redo + autosave (IndexedDB).
- [ ] Export sheet.
- [ ] Client error reporting endpoint + log line, `IPHONE-CHECKLIST.md`.
- [ ] Update screenshots in `docs/screenshots/` and README.

## Phase 4: ML on the device (stretch)

Only worth doing once Phases 1–3 are solid. The point is offline use and zero server load.

- [ ] **Matting in the browser** with `onnxruntime-web` (WebGPU where available, WASM/SIMD otherwise). U2NetP is ~4 MB
      and is the realistic first target; BiRefNet Tiny is probably too big and slow for phones.
- [ ] Models are fetched once and cached (Cache Storage / OPFS), so the PWA works offline after the first load.
- [ ] Capability probe at start: pick device matting if it's fast enough (time one warm-up), else use the server.
- [ ] Optional lightweight upscaler (ESRGAN-lite class) on WebGPU.
- [ ] SAM click-select, LaMa inpainting, anime/painting styles stay server-side; they are too large for phones today.

Risks: iOS Safari memory ceilings kill tabs on big models; WebGPU isn't on every iPhone; WASM is slow on older devices.
Hence the server fallback is not optional.

## Cross-cutting rules

- Server stays the **reference implementation**; every client-side step has a server equivalent and a parity test.
- Follow `AGENTS.md`: work on `master`; bump `app_version`, the compose `image:` tag and `tests/test_config.py` together;
  bump `CACHE_VERSION` and `?v=` strings when `web/` changes; `ruff format`; add tests with each feature.
- Background images and the cached matte are user content: keep them in the browser, don't add server storage.
- Each phase ships as its own minor version (1.10, 1.11, …).

## Suggested order

1. Phase 1a: `gl.js` + rotate/flip + colour/vignette shader + before/after slider (small, visible win)
2. Phase 1b: `/api/v1/matte` + cached mask + local blur composite
3. Phase 2: replace-with-image (server), then live shader compositing and edge tools
4. Phase 3: layout/undo/autosave (decide on Svelte first)
5. Phase 4: in-browser matting

## Open questions

- Svelte 5 + Vite, or stay vanilla? (recommended: Svelte, see above)
- Which iOS version is the 14 Pro on? (decides whether WebGPU is even a candidate for Phase 4)
- Is the replacement background always a photo the user supplies, or do we also ship a few built-in scenes/gradients?
- Should full-size export always happen on the phone when possible, or always on the server for consistency?
