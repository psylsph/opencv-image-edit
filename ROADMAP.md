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
- [x] **Colour fringe removal / foreground colour estimation** (1.14.0): edge pixels move toward the subject's own colour
      spread outward from its solid core (normalised convolution, sigma 6 px at the processing size);
      `defringe_colors()` on the server, a core pass + blurred (colour, core) pair in the shader; parity cases added.
- [ ] **Edge tools, part 2b**: an optional soft shadow under the subject, as vid-ed-mobile's *Edge & shadow*.
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

## Phase 5: Face replacement

**Done in 1.13.0 (neural engine first, one face):** take a face from a donor photo and put it on one face in the photo
being edited. Decisions: swap from another photo; go straight to the neural engine; one face at a time.

How it works (server, `app/models/face.py` + `app/pipeline/face.py`):

1. **Detect** with `cv2.FaceDetectorYN` (YuNet, 0.2 MB): box + 5 landmarks; faces are numbered left to right.
2. **Identity**: ArcFace `w600k_r50` embeds the donor face (unit-length 512-d vector).
3. **Swap** (1.15.0: **HyperSwap 1c, 256 px**; InSwapper 128 is the fallback if it is the only swapper installed): the
   target face is aligned with the layout the generators were trained on (`SWAP_FRACTIONS`, a bit more margin than
   ArcFace's own 112 layout), the generator renders the donor identity, a LAB colour match pulls skin tone toward the
   photo, and it is pasted back through the generator's own face mask times a soft ellipse (never touching the crop
   border, which used to leave a line across the forehead).
3b. **Face shape** (1.17.0, default 0.8): the generator alone keeps the *target's* outline, jaw and expression and only
   repaints the inner features. So before it runs, the donor's own face (aligned in the same layout, masked to skin and
   features with BiSeNet so no hair strands come along, colour-matched to the target's skin) is mixed into the target
   crop by the slider; the generator then harmonises it. 0 = the old behaviour, 1 = the donor's face shape and
   expression. Measured donor similarity 0.756 (0) -> 0.809 (0.8). Tried and dropped: cross-fading raw donor pixels
   after the swap (ghosting), pushing the identity vector away from the target (lower match), extra passes (tiny gain).
3c. **Placement and realism** (1.19.0): the detector's 5 points are rough on turned heads, so a 68-point landmark model
   (2DFAN4) refines them (eye centres, nose tip, mouth corners) before anything is aligned. With the donor pre-paste,
   a piecewise-affine warp (Delaunay triangles over the 68 points) moves the donor's eyes, brows, nose and mouth onto
   the target's, and lets the jaw outline follow the donor by the shape slider. Then: skin-to-skin LAB colour transfer
   using the parser's skin pixels (stops a pale face on a dark neck; default 0.8), a visible-face mask from the parser
   so hair, glasses and hands in front of the face stay on top, and texture matching (sharpen + grain to the photo's
   own fine-detail level, default 0.8) so the face is not airbrushed.
3d. **Pose-aware**: the donor-shape strength is reduced automatically when the two heads are turned differently (yaw
   from the 5 points), because a three-quarter face pasted onto a frontal one distorts; skin darker than the donor's
   own skin (hair strands, hat shadow) is kept out of the pre-paste.
4. **Sharpen** (optional, default 0.3): GFPGAN 1.4 on a 512 px FFHQ-aligned crop, blended back.
5. **Hair** (optional): BiSeNet face parsing finds hair on both people in a wide head crop. The target's hair is erased
   by LaMa inpainting (not the face), the donor's hair is pasted over, scaled/rotated to the target's head by the same
   landmark alignment, never over eyes/nose/mouth, and faded before the crop border.
6. Runs **before** background, grain and filters so those unify the result. The cached matte stays valid.

Why 1.15.0 changed the engine: measured with ArcFace cosine similarity (does the result look like the donor?), the
1.13 setup scored 0.60 on five test pairs; using the right alignment layout alone gave 0.80 with InSwapper, and
HyperSwap 1c gives 0.76 with a much cleaner, sharper face. Restorer and colour match both *lowered* identity (0.67 ->
0.60 at strength 0.8/0.5), so their defaults dropped to 0.3. `tools`-style experiments live in the session notes; rerun
the measurement (embed result vs donor) when changing the engine.

API: `POST /api/v1/faces` (boxes + landmarks, same frame as process), `GET /api/v1/faces/status` (models installed?),
and `face` settings + `face_donor` upload on `POST /api/v1/process` (`enabled`, `target`, `donor`, `blend`,
`color_match`, `restore`). 422 with a readable message when no face is found in either photo or an index is out of range.

UI: **Face swap** group with donor picker, numbered tappable rings on the photo, face chips when there are several faces,
"Also take their hair", and sliders for edge softness, skin-tone match and sharpening. About 5 s on this CPU for the
face, 7-15 s with hair.

Models (~1.1 GB, `scripts/download_models.py`): YuNet, ArcFace, HyperSwap 1c, GFPGAN 1.4, BiSeNet (hair/skin/occlusion), 2DFAN4 (landmarks); LaMa (already
there for Remove Object) erases the old hair. **They land in the image
build, but the `model-cache` volume hides them on an existing install**; copy them in with
`docker cp <file> opencv-image-edit:/home/app/models/` (or run `scripts/download_models.py /home/app/models` inside
the container). `GET /api/v1/faces/status` tells you what is missing.

Next:

- [ ] Hair limits to improve: hats on the target or donor, long hair over busy backgrounds (inpaint smears), donors whose
      shoulders the parser calls hair. A cleaner hair matte (matting instead of parsing) and colour-matching the donor's hair.
- [ ] Donors with hair across the face (or a hat shadow) can still bring a dark streak: the generators copy whatever is
      on the donor's face into the result. Tried cleaning the donor crop by inpainting before embedding it; it distorted the
      identity and made things worse, so it was reverted. Better: choose a clean donor frame, or a face-parsing-aware
      inpainting that keeps the identity features.
- [ ] Live local composite: the server returns just the swapped face patch + mask + placement and `gl.js` composites it,
      so edge softness / skin-tone sliders become live.
- [ ] Several faces at once ("swap all", one donor per face).
- [ ] Drag to nudge the paste; a manual mask brush for hair/glasses/hands over the face.
- [ ] Faster restorer (the 512 px GFPGAN pass is the slowest step) or a smaller model; fp16 swapper (halves the download).
- [ ] Run IPHONE-CHECKLIST.md on the iPhone 14 Pro (rings, tapping, HEIC donor).

Risks and limits:

- **Licences:** `inswapper_128` weights are non-commercial research only; HyperSwap/BiSeNet/GFPGAN/ArcFace have their own
  terms (check before redistributing). Fine for a private home tool, not if this app is
  ever distributed or sold. GFPGAN and ArcFace weights have their own terms; check before redistributing anything.
- **Consent:** the UI says to use faces only with the person's permission. Nothing is stored server-side.
- **Quality limits:** profile views, heavy occlusion (hands, hair, sunglasses) and strong lighting differences look worse.
  The swapper works at 128 px, so very large faces rely on the restorer for detail.
- **Cost:** the models add ~1.1 GB to the image and volume; the 180 s idle timeout frees their RAM.

## Saving results (1.19.4)

- Background -> Remove used to leave the *old background's colour* under the transparent pixels (alpha 0), so any viewer,
  converter or app that ignores alpha showed the old background. The colour under fully transparent pixels is now cleared
  (`clear_transparent_colour`), and the Result view has a **Save PNG** button that downloads the server's PNG byte for
  byte (plus **Share** where the browser supports sharing files) and a checkerboard behind the picture. Long-pressing the
  picture still works but can flatten it, and in the Edit view it saves the original photo, not the preview.

## Backlog (not scheduled)

- [x] **Tattoo removal** (1.16.0, rewritten 1.19.3 after three over-engineered versions): *Remove Tattoo* group; tap
      (MobileSAM) or paint a selection -> `POST /api/v1/tattoo`. The painted area is only where to look. Ink = selected
      pixels darker than the local skin by a margin (the sensitivity slider; 40 L units at 0 down to 6 at 1, never
      below ~3x the photo's grain) and not strongly coloured (so a saturated blue wall or red curtain is not "ink").
      The skin reference is built once per request, independent of the slider, so raising the slider can only add ink
      (a test enforces it; an earlier version got this backwards). The replacement is the flesh colour next to each
      ink pixel, taken only from skin pixels (never background, hair or ink) and filled inward from each patch's edge
      so it follows the local shading; then skin grain is added, edges are soft, and a second pass catches faint halo.
      The skin colour is chosen among skin-hued pixels (generous YCrCb) near the selection: choosing the most common
      colour instead locked onto a flat wall in loose selections, which was the real cause of the 'bleeding'.
      Lessons: every geometric safeguard I added (skin on 5 of 8 sides, enclosed holes, colour lines, skin gates) made it
      silently do nothing on real photos where the tattoo sits at the edge of the skin. Measured on a real photo with a
      drawn tattoo: colour error against the untattooed original 34.5 -> ~14, the letters are gone, a faint outline
      remains where ink overlapped hair. Limits: dark neutral backgrounds or hair inside the painted area are repainted
      too (paint close to the tattoo); a skin-coloured wall cannot be told from skin.
      1.19.4: the matting model runs under the covers (u2netp, ~0.5 s) and its person mask is a hard boundary: ink is only
      searched on the person, the skin colour is only sampled from the person, and only the person's pixels are ever
      repainted, so no backdrop colour (however skin-like) can bleed in; measured on a real photo, 4583 backdrop pixels
      were repainted without the mask and 0 with it. Without the matting model the tool still works, unprotected.
      1.19.5 (laser-like): after feedback that it painted the whole area, ink is now only (a) darker than the local skin
      by the sensitivity margin, (b) either very dark (>60 L below skin) or off the skin's hue by a gap that shrinks with
      sensitivity - shaded skin, creases and freckles keep the skin hue so they are never ink, (c) the tattoo's own colour
      (on the straight skin-to-ink colour line, with the ink colour taken from the strongest candidates), (d) a patch
      bigger than a speck. Each ink pixel is replaced by the colour of the *nearest real skin pixel* (distance-transform
      labels, so pores and grain come along) blended 65/35 with the smooth local skin, plus measured grain, 1 px feathered
      rim. The pool of skin pixels to copy from now includes shaded skin (leaving it out made the fill ~8 L too light,
      the pale ghost). Measured on real photos: pixels in the painted area that are not tattoo and changed 0.0% for a
      tight selection (7% with a sloppy mask that includes eyelashes), signed L error at the tattoo about 0.
      Ideas: show the detected ink and let the user veto, inpainting for complex backgrounds, parser protection for
      hair on faces (it mislabels dark ink as hair, so it needs care).

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
5. Phase 5: face replacement (can start any time after Phase 2; independent of the Phase 3 UI rework)
6. Phase 4: in-browser matting

## Open questions


- Svelte 5 + Vite, or stay vanilla? (recommended: Svelte, see above)
- Which iOS version is the 14 Pro on? (decides whether WebGPU is even a candidate for Phase 4)
- Is the replacement background always a photo the user supplies, or do we also ship a few built-in scenes/gradients?
- Should full-size export always happen on the phone when possible, or always on the server for consistency?
