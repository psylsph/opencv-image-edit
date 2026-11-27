// OpenCV Image Editor — frontend logic
// Vanilla ES2022 module, no build step, no framework.
//
// Wires together:
//   - file picker (with camera capture on mobile)
//   - preset pills (fetched from /presets)
//   - 4 control groups (background / grain / upscale / filters)
//   - POST to /api/v1/process with multipart (file + settings JSON)
//   - tabbed result display (final + 5 debug views)
//   - PWA service worker registration

import { createRenderer, isIdentity, liveParams, loadSource } from "./gl.js";

const BG_SWATCHES = ["#00b140", "#ffffff", "#111111", "#1e6bff", "#ff4fa3", "#ffd23f", "#7c4dff", "#ff7a1a"];

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  file: null,
  bgFile: null,          // replacement background photo (mode "image"), kept in memory
  bgThumbUrl: null,
  previewUrl: null,
  presets: {},
  currentPreset: null,
  processing: false,
  live: { renderer: null, image: null, raf: 0, token: 0, matteKey: null, matteBusy: null, matteFailed: null },  // on-device preview (gl.js)
  lastResult: null,
  inpaintResult: null,   // separate from main process result
  selections: {},        // tool -> its saved selection while another tool has the canvas
  // Mask canvas drawing state
  mask: {
    enabled: false,
    owner: "inpaint",       // which group owns the shared canvas: "inpaint" | "blur"
    mode: "click",          // "click" = SAM point-prompt, "paint" = manual brush
    ctx: null,              // 2D rendering context
    displayW: 0,            // canvas CSS size (matches RENDERED image, not element box)
    displayH: 0,
    offsetX: 0,             // letterbox offset within element (object-fit: contain)
    offsetY: 0,
    imageW: 0,              // original image size (for scaling on submit)
    imageH: 0,
    brushSize: 24,
    drawing: false,
    lastX: 0,
    lastY: 0,
    history: [],            // stack of ImageData snapshots (for undo, paint mode only)
    isDirty: false,         // has user painted anything?
    detectedMask: null,     // base64 PNG of SAM mask (click mode)
    detectionScore: 0,      // IoU score from SAM
    segmenting: false,      // request in flight
  },
};

// --- Defaults (used by Reset and on first load) ---
const DEFAULT_SETTINGS = {
  background: {
    enabled: false, mode: "blur", blur_strength: 15, model_name: "u2netp",
    color: "#00b140", zoom: 1, offset_x: 0, offset_y: 0, feather: 0, shrink: 0, defringe: 0,
  },
  face: { enabled: false, target: 0, donor: 0, blend: 0.5, color_match: 0.8, restore: 0.3, hair: false, shape: 0.8, detail: 0.8, all_faces: false, texture: 0 },
  grain: { enabled: false, intensity: 0.4 },
  upscale: { enabled: false, scale: 2, algorithm: "interp" },
  filters: {
    enabled: false,
    brightness: 1, contrast: 1, saturation: 1, sharpness: 1,
    vignette_strength: 0, sepia: false, grayscale_blend: 0,
    anime: false, anime_style: "hayao",
    painting: false, painting_style: "oil",
  },
};

// --- Init ---
async function init() {
  document.body.dataset.view = "edit";
  await loadPresets();
  renderPresetPills();
  bindEvents();
  applyPreset(null); // start at defaults
  setMaskMode("click"); // initialize mask mode UI
  loadModels();
  registerServiceWorker();
}

async function loadModels() {
  try {
    const resp = await fetch("/api/v1/models");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const select = $("#bg-model");
    if (!select) return;
    select.innerHTML = "";
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = m.display_name;
      if (!m.available) {
        opt.textContent += " (not downloaded)";
        opt.disabled = true;
      }
      select.appendChild(opt);
    }
    select.value = "u2netp";
    select.disabled = false;
  } catch (err) {
    console.error("Failed to load models:", err);
    // Keep the default u2netp option
    const select = $("#bg-model");
    if (select) select.disabled = false;
  }
}

async function loadPresets() {
  try {
    const resp = await fetch("/presets");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    state.presets = await resp.json();
  } catch (err) {
    console.error("Failed to load presets:", err);
    state.presets = {};
  }
}

function renderPresetPills() {
  const container = $("#preset-pills");
  container.innerHTML = "";
  for (const [name, meta] of Object.entries(state.presets)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset-pill";
    btn.dataset.preset = name;
    btn.textContent = meta.label;
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-label", `${meta.label} preset`);
    btn.addEventListener("click", () => applyPreset(name));
    container.appendChild(btn);
  }
}

function bindEvents() {
  // File input (capture=environment offers camera on mobile + gallery on desktop)
  $("#file-input").addEventListener("change", onFileSelected);
  $("#upload-btn").addEventListener("click", () => $("#file-input").click());

  // Clear button
  $("#clear-btn").addEventListener("click", clearFile);

  // Process button
  $("#process-btn").addEventListener("click", processImage);

  // Rotate / flip the photo itself so selections and every endpoint stay in sync
  $("#rot-ccw").addEventListener("click", () => transformImage("ccw"));
  $("#rot-cw").addEventListener("click", () => transformImage("cw"));
  $("#flip-h").addEventListener("click", () => transformImage("flipH"));
  $("#flip-v").addEventListener("click", () => transformImage("flipV"));

  // Live preview redraws on any control change (sliders fire "input", toggles fire "change")
  const controls = $(".controls-section");
  controls.addEventListener("input", scheduleLiveDraw);
  controls.addEventListener("change", scheduleLiveDraw);
  new ResizeObserver(scheduleLiveDraw).observe($(".preview-stack"));

  // "On" badges follow the enable checkboxes (programmatic changes call it too)
  $(".controls-section").addEventListener("change", syncGroupBadges);
  syncGroupBadges();

  // Reset button
  $("#reset-btn").addEventListener("click", () => applyPreset(null));

  // Background enabled toggle: dim sub-controls when off
  $("#bg-enabled").addEventListener("change", (e) => {
    setGroupDisabled("#bg-mode", !e.target.checked);
    setGroupDisabled("#bg-source", !e.target.checked);
    setRowDisabled("#blur-strength-row", !e.target.checked);
    updateBackgroundMaskControls();
  });

  // Background mode: hide blur-strength row when in "remove" mode
  $$("input[name='bg-mode']").forEach((r) => {
    r.addEventListener("change", updateBlurStrengthVisibility);
  });
  // Replacement background: swatches, custom colour, photo picker
  const swatches = $("#bg-swatches");
  for (const c of BG_SWATCHES) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "swatch";
    b.style.background = c;
    b.dataset.color = c;
    b.setAttribute("aria-label", `Background colour ${c}`);
    b.addEventListener("click", () => setBgColor(c));
    swatches.appendChild(b);
  }
  $("#bg-color").addEventListener("input", (e) => setBgColor(e.target.value));
  $("#bg-image-btn").addEventListener("click", () => $("#bg-image-input").click());
  $("#bg-image-input").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) setBgFile(f);
  });
  // Background source: auto matting or manual painted background mask
  $$("input[name='bg-source']").forEach((r) => {
    r.addEventListener("change", updateBackgroundMaskControls);
  });
  // Grain enabled toggle
  $("#grain-enabled").addEventListener("change", (e) => {
    setRowDisabled("#grain-intensity", !e.target.checked, "#grain-group");
  });

  // Upscale enabled toggle
  $("#upscale-enabled").addEventListener("change", (e) => {
    setGroupDisabled("#upscale-scale", !e.target.checked);
    const algoRow = $("#upscale-algorithm");
    if (algoRow) algoRow.disabled = !e.target.checked;
  });

  // Filters enabled toggle. Do not disable pointer events on the whole body:
  // that also blocks the checkbox itself, so manual enabling becomes impossible.
  $("#filters-enabled").addEventListener("change", updateFilterControls);

  // Anime toggle: show/hide style selector
  $("#anime").addEventListener("change", (e) => {
    const controls = $("#anime-controls");
    if (controls) controls.hidden = !e.target.checked;
  });

  // Painting toggle: show/hide style selector
  $("#painting").addEventListener("change", (e) => {
    const controls = $("#painting-controls");
    if (controls) controls.hidden = !e.target.checked;
  });

  // Auto-expand a section's <details> when its toggle is switched on
  [
    ["#bg-enabled", "#bg-group"],
    ["#grain-enabled", "#grain-group"],
    ["#upscale-enabled", "#upscale-group"],
    ["#filters-enabled", "#filters-group"],
    ["#inpaint-enabled", "#inpaint-group"],
  ].forEach(([cbSel, groupSel]) => {
    const cb = $(cbSel);
    const details = $(groupSel);
    if (!cb || !details) return;
    cb.addEventListener("change", () => {
      if (cb.checked) details.open = true;
    });
  });

  // Inpaint (object removal) and Blur Area share one selection canvas + toolbar
  // (also used by Background "Paint BG"); only one of the two groups owns it.
  const OWNERS = ["inpaint", "blur", "tattoo", "facekeep"];
  // Several tools can be on at once; each keeps its own selection. The canvas shows the one whose group is open.
  for (const owner of OWNERS) {
    const group = $(owner === "facekeep" ? "#face-group" : `#${owner}-group`);
    group.addEventListener("toggle", () => {
      if (group.open && $(`#${owner}-enabled`).checked && !(state.mask.enabled && state.mask.owner === owner)) {
        switchOwner(owner);
        updateMaskCanvasVisibility();
      }
    });
    $(`#${owner}-enabled`).addEventListener("change", (e) => {
      if (e.target.checked) {
        switchOwner(owner);
        group.open = true;
      } else {
        delete state.selections[owner];
        if (state.mask.enabled && state.mask.owner === owner) {
          state.mask.enabled = false;
          clearClickSelection(isBackgroundPaintActive());
          const next = OWNERS.find((o) => $(`#${o}-enabled`).checked);
          if (next) {
            state.mask.owner = next;
            state.mask.enabled = true;
            restoreSelection(next);
          }
        }
      }
      updateMaskCanvasVisibility();
      updateBackgroundMaskControls();
      if (e.target.checked && state.file) {
        $("#preview-container").scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }
  $("#blur-area-strength").addEventListener("input", (e) => {
    $("#blur-area-value").textContent = e.target.value;
  });
  for (const [slider, out] of [["#tattoo-strength", "#tattoo-strength-value"], ["#tattoo-grow", "#tattoo-grow-value"]]) {
    $(slider).addEventListener("input", (e) => { $(out).textContent = parseFloat(e.target.value).toFixed(2); });
  }
  // Shared selection toolbar
  $$("input[name='sel-mode']").forEach((r) => {
    r.addEventListener("change", (e) => {
      setMaskMode(e.target.value);
      updateMaskCanvasVisibility();
    });
  });
  $("#sel-brush-size").addEventListener("input", (e) => {
    state.mask.brushSize = parseInt(e.target.value, 10);
    $("#sel-brush-value").textContent = state.mask.brushSize;
  });
  $("#sel-undo").addEventListener("click", undoMaskStroke);
  $("#sel-clear").addEventListener("click", () => {
    clearMask();
    clearClickSelection();
    updateMaskCanvasVisibility();
  });

  // The Process button names the tool when it is the only thing that will run; any other step switched on changes that.
  document.addEventListener("change", (e) => {
    if (e.target instanceof HTMLInputElement && e.target.id.endsWith("-enabled")) updateProcessButton();
  });

  // Accordion: only one control group open at a time (keeps the page short)
  $$(".controls-section details.control-group").forEach((d) => {
    d.addEventListener("toggle", () => {
      if (!d.open) return;
      $$(".controls-section details.control-group").forEach((o) => { if (o !== d) o.open = false; });
    });
  });

  initFaceUi();

  // Bottom nav: Edit | Process | Result
  $("#nav-edit").addEventListener("click", () => setView("edit"));
  $("#nav-result").addEventListener("click", () => setView("result"));
  $("#tabs-more").addEventListener("click", () => {
    const tabs = $("#output-tabs");
    const more = tabs.dataset.more !== "true";
    tabs.dataset.more = String(more);
    $("#tabs-more").textContent = more ? "Less ▴" : "More ▾";
    $("#tabs-more").setAttribute("aria-expanded", String(more));
  });
  // Fill radius output update
  $("#inpaint-radius").addEventListener("input", (e) => {
    $("#inpaint-radius-value").textContent = e.target.value;
  });
  // AI quality/pass count live update
  $("#inpaint-iterations").addEventListener("input", (e) => {
    const passes = parseInt(e.target.value, 10);
    $("#inpaint-iterations-value").textContent = `${passes} ${passes === 1 ? "pass" : "passes"}`;
  });
  // Hide algorithm-specific controls when they don't apply
  function updateInpaintRadiusVisibility() {
    const checked = document.querySelector("input[name='inpaint-algo']:checked");
    if (!checked) return;
    const val = checked.value;
    const row = $("#inpaint-radius-row");
    const iterationsRow = $("#inpaint-iterations-row");
    // LaMa doesn't use radius; only NS/TELEA do
    if (row) row.style.display = (val === "ns" || val === "telea") ? "flex" : "none";
    // Iteration passes improve LaMa only; NS/TELEA use radius
    if (iterationsRow) iterationsRow.style.display = val === "lama" ? "flex" : "none";
  }
  $$("input[name='inpaint-algo']").forEach((r) => {
    r.addEventListener("change", updateInpaintRadiusVisibility);
  });
  updateInpaintRadiusVisibility();

  // Canvas pointer events for painting
  const canvas = $("#mask-canvas");
  initZoom(canvas);
  canvas.addEventListener("pointerdown", startMaskStroke);
  canvas.addEventListener("pointermove", continueMaskStroke);
  canvas.addEventListener("pointerup", endMaskStroke);
  canvas.addEventListener("pointerleave", endMaskStroke);
  canvas.addEventListener("pointercancel", endMaskStroke);

  // Slider live output updates
  const sliders = [
    ["#blur-strength", "#blur-value", (v) => v],
    ["#grain-intensity", "#grain-value", (v) => parseFloat(v).toFixed(2)],
    ["#brightness", "#brightness-value", (v) => parseFloat(v).toFixed(2)],
    ["#contrast", "#contrast-value", (v) => parseFloat(v).toFixed(2)],
    ["#saturation", "#saturation-value", (v) => parseFloat(v).toFixed(2)],
    ["#sharpness", "#sharpness-value", (v) => parseFloat(v).toFixed(2)],
    ["#vignette", "#vignette-value", (v) => parseFloat(v).toFixed(2)],
    ["#bg-zoom", "#bg-zoom-value", (v) => parseFloat(v).toFixed(2)],
    ["#bg-x", "#bg-x-value", (v) => parseFloat(v).toFixed(2)],
    ["#bg-y", "#bg-y-value", (v) => parseFloat(v).toFixed(2)],
    ["#bg-feather", "#bg-feather-value", (v) => parseFloat(v).toFixed(2)],
    ["#bg-defringe", "#bg-defringe-value", (v) => parseFloat(v).toFixed(2)],
    ["#bg-shrink", "#bg-shrink-value", (v) => parseFloat(v).toFixed(2)],
    ["#face-shape", "#face-shape-value", (v) => parseFloat(v).toFixed(2)],
    ["#face-blend", "#face-blend-value", (v) => parseFloat(v).toFixed(2)],
    ["#face-color", "#face-color-value", (v) => parseFloat(v).toFixed(2)],
    ["#face-restore", "#face-restore-value", (v) => parseFloat(v).toFixed(2)],
    ["#face-detail", "#face-detail-value", (v) => parseFloat(v).toFixed(2)],
    ["#face-texture", "#face-texture-value", (v) => parseFloat(v).toFixed(2)],
  ];
  sliders.forEach(([slider, output, fmt]) => {
    const s = $(slider), o = $(output);
    if (!s || !o) return;
    s.addEventListener("input", () => { o.textContent = fmt(s.value); });
  });

  // Output tabs
  $$(".output-tabs .tab[data-tab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      if (tab.disabled) return;
      const name = tab.dataset.tab;
      $$(".output-tabs .tab[data-tab]").forEach((t) => t.classList.toggle("active", t === tab));
      $$(".output-pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === name));
    });
  });
}

/** Mirror each group's "Enable" checkbox onto the <details> so collapsed groups show "On". */
function syncGroupBadges() {
  for (const [cb, group] of [
    ["#bg-enabled", "#bg-group"], ["#blur-enabled", "#blur-group"], ["#face-enabled", "#face-group"], ["#tattoo-enabled", "#tattoo-group"],
    ["#inpaint-enabled", "#inpaint-group"], ["#grain-enabled", "#grain-group"],
    ["#upscale-enabled", "#upscale-group"], ["#filters-enabled", "#filters-group"],
  ]) {
    const g = $(group);
    if (g) g.dataset.on = String(!!$(cb)?.checked);
  }
}

function setGroupDisabled(selector, disabled) {
  const el = $(selector);
  if (!el) return;
  el.style.opacity = disabled ? "0.5" : "1";
  el.style.pointerEvents = disabled ? "none" : "auto";
}

function setRowDisabled(rowSelector, disabled, groupSelector = null) {
  const row = $(rowSelector);
  if (row) {
    row.style.opacity = disabled ? "0.5" : "1";
    row.style.pointerEvents = "auto";  // sliders still respond for visual feedback
    const input = row.querySelector("input, select");
    if (input) input.disabled = disabled;
  }
}

function updateFilterControls() {
  const enabled = $("#filters-enabled")?.checked;
  const body = $("#filters-group .control-body");
  if (!body) return;
  body.style.opacity = "1";
  body.style.pointerEvents = "auto";
  const controls = [
    "#brightness",
    "#contrast",
    "#saturation",
    "#sharpness",
    "#vignette",
    "#sepia",
    "#anime",
    "#anime-style",
    "#painting",
    "input[name='painting-style']",
  ];
  controls.forEach((selector) => {
    const input = $(selector);
    if (!input) return;
    input.disabled = !enabled;
    const row = input.closest(".slider-row, .checkbox-row");
    if (row) row.style.opacity = enabled ? "1" : "0.5";
  });
}

function updateBlurStrengthVisibility() {
  const checked = document.querySelector("input[name='bg-mode']:checked");
  if (!checked) return;
  const mode = checked.value;
  const row = $("#blur-strength-row");
  if (row) row.style.display = mode === "blur" ? "flex" : "none";
  const replacing = mode === "color" || mode === "image";
  $("#bg-replace").hidden = !replacing;
  $("#bg-color-row").hidden = mode !== "color";
  $("#bg-image-row").hidden = mode !== "image";
  $("#bg-image-adjust").hidden = mode !== "image" || !state.bgFile;
}

function setBgColor(c) {
  $("#bg-color").value = c;
  $$("#bg-swatches .swatch").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.color === c.toLowerCase())));
  scheduleLiveDraw();
}

/** Choose (or clear with null) the replacement background photo. */
async function setBgFile(file) {
  state.bgFile = file;
  if (state.bgThumbUrl) URL.revokeObjectURL(state.bgThumbUrl);
  state.bgThumbUrl = file ? URL.createObjectURL(file) : null;
  const thumb = $("#bg-image-thumb");
  if (state.bgThumbUrl) thumb.src = state.bgThumbUrl; else thumb.removeAttribute("src");
  thumb.hidden = !file;
  $("#bg-image-btn").textContent = file ? "🖼️ Change background photo" : "🖼️ Choose background photo";
  updateBlurStrengthVisibility();
  await loadBgIntoRenderer();
}

async function loadBgIntoRenderer() {
  const r = state.live.renderer;
  if (!r) return;
  const file = state.bgFile;
  try {
    r.setBackgroundImage(file ? await loadSource(file) : null);
  } catch (err) {
    console.warn("background photo can't be previewed on this device (Process still works):", err);
    r.setBackgroundImage(null);
  }
  scheduleLiveDraw();
}

// --- one saved selection per tool -------------------------------------------------------------------------------
// The selection canvas belongs to one tool at a time; the others' selections wait here until their group is opened
// again, and are sent along when Process runs.
const EDIT_TOOLS = ["inpaint", "tattoo", "blur"];  // the order the server runs them in

function saveSelection(owner) {
  const c = $("#mask-canvas");
  if (!state.mask.ctx || !c.width || !c.height) return;
  state.selections[owner] = {
    mode: state.mask.mode,
    detected: state.mask.detectedMask,
    image: state.mask.ctx.getImageData(0, 0, c.width, c.height),
    dirty: state.mask.isDirty,
    history: state.mask.history.slice(),  // so Undo still works after switching tools and back
  };
}

function restoreSelection(owner) {
  const saved = state.selections[owner];
  delete state.selections[owner];
  setMaskMode(saved ? saved.mode : state.mask.mode);  // also clears the canvas
  if (!saved || !state.mask.ctx) return;
  const c = $("#mask-canvas");
  if (saved.image.width === c.width && saved.image.height === c.height) {
    state.mask.ctx.putImageData(saved.image, 0, 0);
    state.mask.history = saved.history || [];
  } else {
    // The canvas was resized while another tool had it (rotation, window size): scale the selection instead of
    // losing it. The undo steps were at the old size, so they go.
    const tmp = document.createElement("canvas");
    tmp.width = saved.image.width;
    tmp.height = saved.image.height;
    tmp.getContext("2d").putImageData(saved.image, 0, 0);
    state.mask.ctx.drawImage(tmp, 0, 0, c.width, c.height);
  }
  state.mask.detectedMask = saved.detected;
  state.mask.isDirty = saved.dirty;
  // A tap result that arrived while this tool was in the background has no overlay drawn yet.
  if (saved.mode === "click" && saved.detected && saved.overlayMissing) drawDetectedMaskOnCanvas(saved.detected);
}

function switchOwner(owner) {
  if (state.mask.enabled && state.mask.owner === owner) return;
  if (state.mask.enabled) saveSelection(state.mask.owner);
  state.mask.owner = owner;
  state.mask.enabled = true;
  restoreSelection(owner);
}

function savedHasContent(sel) {
  if (sel.mode === "click") return !!sel.detected;
  if (!sel.dirty) return false;
  const d = sel.image.data;
  for (let i = 3; i < d.length; i += 4) if (d[i] > 0) return true;
  return false;
}

/** The selection of ``owner`` as a full-size PNG (white = selected), whether it's on the canvas or saved; or null. */
async function selectionBlob(owner) {
  if (state.mask.enabled && state.mask.owner === owner) return hasMaskContent() ? await getMaskPngBlob() : null;
  const sel = state.selections[owner];
  if (!sel || !savedHasContent(sel)) return null;
  if (sel.mode === "click") return await (await fetch(sel.detected)).blob();
  const tmp = document.createElement("canvas");
  tmp.width = sel.image.width;
  tmp.height = sel.image.height;
  tmp.getContext("2d").putImageData(sel.image, 0, 0);
  return await getPaintedMaskPngBlob(tmp);
}

function isBackgroundPaintActive() {
  const bgEnabled = $("#bg-enabled")?.checked;
  const source = document.querySelector("input[name='bg-source']:checked")?.value || "auto";
  // Object-removal masking owns the shared canvas when enabled.
  return !!bgEnabled && source === "paint" && !state.mask.enabled;
}

/** Which selection tool currently owns the photo canvas: "inpaint" | "blur" | "bg" | null. */
function activeTool() {
  if (state.mask.enabled) return state.mask.owner;
  return isBackgroundPaintActive() ? "bg" : null;
}

const TOOL_LABELS = {
  inpaint: { title: "✂️ Remove object", verb: "remove" },
  tattoo: { title: "🩹 Remove tattoo", verb: "remove the tattoo from" },
  facekeep: { title: "🙂 Keep from the original", verb: "keep as in the original (hair, glasses, hand)" },
  blur: { title: "💧 Blur area", verb: "blur" },
  bg: { title: "🌫️ Paint the background", verb: "treat as background" },
};

function updateMaskCanvasVisibility() {
  const canvas = $("#mask-canvas");
  if (!canvas) return;
  const tool = activeTool();
  const visible = !!tool;
  canvas.hidden = !visible;
  $("#zoom-controls").hidden = !visible;
  if (!visible) resetView();
  const mode = tool === "bg" ? "paint" : state.mask.mode;
  // Click mode lets the page scroll under the finger; paint modes capture it.
  canvas.dataset.mode = mode;
  if (visible && state.mask.imageW) setupMaskCanvas();

  // Shared toolbar
  const bar = $("#select-toolbar");
  bar.hidden = !visible;
  if (visible) {
    const { title, verb } = TOOL_LABELS[tool];
    $("#sel-title").textContent = title;
    $("#sel-mode").hidden = tool === "bg";
    $("#sel-paint-controls").hidden = mode !== "paint";
    $("#sel-undo").hidden = mode !== "paint";
    if (!state.mask.detectedMask || mode !== "click") {
      $("#sel-hint").textContent = mode === "click"
        ? `Tap an object to select it, then press the button below to ${verb} it.`
        : `Drag over the area to ${verb}.`;
    }
  }
  updateProcessButton();
}

function updateBackgroundMaskControls() {
  const paint = isBackgroundPaintActive();
  const enabled = $("#bg-enabled")?.checked;
  const source = document.querySelector("input[name='bg-source']:checked")?.value || "auto";
  setGroupDisabled("#bg-source", !enabled);
  // Show model dropdown only in auto mode (not paint)
  const modelRow = $("#bg-model-row");
  if (modelRow) modelRow.style.display = source === "auto" ? "flex" : "none";
  updateMaskCanvasVisibility();
}

function onFileSelected(e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  loadFile(file);
}

function loadFile(file) {
  // Revoke previous preview URL
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);

  resetView();
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  onFaceSourceChanged();

  // Wire up natural-dimension capture for the mask canvas
  const img = $("#preview");
  img.src = state.previewUrl;
  img.onload = () => {
    state.mask.imageW = img.naturalWidth;
    state.mask.imageH = img.naturalHeight;
    if (state.mask.enabled || isBackgroundPaintActive()) setupMaskCanvas();
  };

  $("#preview-container").hidden = false;
  $("#upload-btn").hidden = true;
  startLivePreview(file);

  // Reset mask on new file
  clearClickSelection();
  if (state.mask.ctx) clearMask();
  state.mask.history = [];
  state.mask.isDirty = false;

  // Clear previous output when a new file is chosen
  $("#output-section").hidden = true;
  setResultAvailable(false);
  setStatus("");
  state.lastResult = null;
  state.inpaintResult = null;
  state.selections = {};

  updateProcessButton();
}

/** Rotate/flip the uploaded photo and swap it in as the working file. */
async function transformImage(op) {
  if (!state.file || state.processing) return;
  let bmp;
  try {
    bmp = await createImageBitmap(state.file);  // honours EXIF orientation
  } catch (err) {
    setStatus("This browser can't decode that image for rotating.", "error");
    return;
  }
  const w = bmp.width, h = bmp.height;
  const swap = op === "cw" || op === "ccw";
  const canvas = document.createElement("canvas");
  canvas.width = swap ? h : w;
  canvas.height = swap ? w : h;
  const ctx = canvas.getContext("2d");
  if (op === "cw") { ctx.translate(h, 0); ctx.rotate(Math.PI / 2); }
  else if (op === "ccw") { ctx.translate(0, w); ctx.rotate(-Math.PI / 2); }
  else if (op === "flipH") { ctx.translate(w, 0); ctx.scale(-1, 1); }
  else if (op === "flipV") { ctx.translate(0, h); ctx.scale(1, -1); }
  ctx.drawImage(bmp, 0, 0);
  bmp.close?.();

  // Keep PNG/alpha formats lossless; photos go back out as high-quality JPEG
  // so the result stays under the upload size limit.
  const keepPng = state.file.type === "image/png" || state.file.type === "image/webp";
  const toBlob = (type, q) => new Promise((res) => canvas.toBlob(res, type, q));
  let blob = keepPng ? await toBlob("image/png") : await toBlob("image/jpeg", 0.95);
  if (blob && blob.size > 9.5 * 1024 * 1024) blob = await toBlob("image/jpeg", 0.9);
  if (!blob) { setStatus("Could not rotate the image.", "error"); return; }

  const ext = blob.type === "image/png" ? "png" : "jpg";
  const base = (state.file.name || "photo").replace(/\.[^.]+$/, "");
  loadFile(new File([blob], `${base}.${ext}`, { type: blob.type }));
}

function clearFile() {
  stopLivePreview();
  state.file = null;
  onFaceSourceChanged();
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }
  $("#file-input").value = "";
  $("#preview").removeAttribute("src");
  $("#preview-container").hidden = true;
  $("#upload-btn").hidden = false;
  $("#output-section").hidden = true;
  setResultAvailable(false);
  setStatus("");
  state.lastResult = null;
  state.inpaintResult = null;
  state.selections = {};
  // Reset mask state
  state.mask.imageW = 0;
  state.mask.imageH = 0;
  if (state.mask.ctx) clearMask();
  state.mask.history = [];
  state.mask.isDirty = false;
  updateProcessButton();
}

function updateProcessButton() {
  document.body.classList.toggle("has-file", !!state.file);
  const btn = $("#process-btn");
  btn.disabled = !state.file || state.processing;
  if (state.processing) {
    btn.textContent = "⏳ Processing…";
  } else {
    const others = ["#face-enabled", "#bg-enabled", "#grain-enabled", "#upscale-enabled", "#filters-enabled"].some((sel) => $(sel)?.checked);
    const tools = EDIT_TOOLS.filter((t) => $(`#${t}-enabled`)?.checked).length;
    const tool = others || tools > 1 ? null : activeTool();
    btn.textContent = tool === "blur" ? "💧 Apply blur" : tool === "inpaint" ? "✂️ Remove object" : tool === "tattoo" ? "🩹 Remove tattoo" : "🚀 Process";
  }
}

function collectSettings() {
  const checkedRadio = (name) => {
    const el = document.querySelector(`input[name='${name}']:checked`);
    return el ? el.value : null;
  };
  return {
    face: {
      enabled: $("#face-enabled").checked,
      target: faceUi.target,
      donor: faceUi.donorIdx,
      blend: parseFloat($("#face-blend").value),
      color_match: parseFloat($("#face-color").value),
      restore: parseFloat($("#face-restore").value),
      hair: $("#face-hair").checked,
      shape: parseFloat($("#face-shape").value),
      detail: parseFloat($("#face-detail").value),
      all_faces: $("#face-all").checked,
      texture: parseFloat($("#face-texture").value),
    },
    background: {
      enabled: $("#bg-enabled").checked,
      mode: checkedRadio("bg-mode") || "blur",
      blur_strength: parseInt($("#blur-strength").value, 10),
      model_name: $("#bg-model")?.value || "u2netp",
      color: $("#bg-color").value,
      zoom: parseFloat($("#bg-zoom").value),
      offset_x: parseFloat($("#bg-x").value),
      offset_y: parseFloat($("#bg-y").value),
      feather: parseFloat($("#bg-feather").value),
      defringe: parseFloat($("#bg-defringe").value),
      shrink: parseFloat($("#bg-shrink").value),
    },
    grain: {
      enabled: $("#grain-enabled").checked,
      intensity: parseFloat($("#grain-intensity").value),
    },
    upscale: {
      enabled: $("#upscale-enabled").checked,
      scale: parseInt(checkedRadio("upscale-scale") || "2", 10),
      algorithm: $("#upscale-algorithm").value,
    },
    filters: {
      enabled: $("#filters-enabled").checked,
      brightness: parseFloat($("#brightness").value),
      contrast: parseFloat($("#contrast").value),
      saturation: parseFloat($("#saturation").value),
      sharpness: parseFloat($("#sharpness").value),
      vignette_strength: parseFloat($("#vignette").value),
      sepia: $("#sepia").checked,
      grayscale_blend: 0,
      anime: $("#anime").checked,
      anime_style: $("#anime-style").value || "hayao",
      painting: $("#painting").checked,
      painting_style: checkedRadio("painting-style") || "oil",
    },
  };
}

function setSlider(el, value, outputEl, fmt = (v) => v) {
  if (!el) return;
  el.value = value;
  if (outputEl) outputEl.textContent = fmt(value);
}

function applyPreset(name) {
  state.currentPreset = name;
  $$(".preset-pill").forEach((p) => p.classList.toggle("active", p.dataset.preset === name));

  let s, desc;
  if (name && state.presets[name]) {
    s = state.presets[name].settings;
    desc = state.presets[name].description;
  } else {
    s = JSON.parse(JSON.stringify(DEFAULT_SETTINGS));
    desc = "Custom settings";
  }
  $("#preset-description").textContent = desc;

  // Face swap (presets from older servers have no `face` block)
  const fd = { ...DEFAULT_SETTINGS.face, ...s.face };
  $("#face-enabled").checked = fd.enabled;
  $("#face-hair").checked = !!fd.hair;
  $("#face-all").checked = !!fd.all_faces;
  for (const [id, key] of [["face-texture", "texture"], ["face-blend", "blend"], ["face-color", "color_match"], ["face-restore", "restore"], ["face-shape", "shape"], ["face-detail", "detail"]]) {
    setSlider($(`#${id}`), fd[key], $(`#${id}-value`), (v) => parseFloat(v).toFixed(2));
  }
  faceUi.target = fd.target; faceUi.donorIdx = fd.donor;
  renderFaceUi();

  // Background
  $("#bg-enabled").checked = s.background.enabled;
  const bgMode = document.querySelector(`input[name='bg-mode'][value='${s.background.mode}']`);
  if (bgMode) bgMode.checked = true;
  setSlider($("#blur-strength"), s.background.blur_strength, $("#blur-value"));
  const bgSource = document.querySelector("input[name='bg-source'][value='auto']");
  if (bgSource) bgSource.checked = true;
  const bgModel = $("#bg-model");
  if (bgModel && s.background.model_name) bgModel.value = s.background.model_name;
  const bgd = { ...DEFAULT_SETTINGS.background, ...s.background };
  setBgColor(bgd.color);
  for (const [id, key] of [["bg-zoom", "zoom"], ["bg-x", "offset_x"], ["bg-y", "offset_y"], ["bg-feather", "feather"], ["bg-shrink", "shrink"], ["bg-defringe", "defringe"]]) {
    setSlider($(`#${id}`), bgd[key], $(`#${id}-value`), (v) => parseFloat(v).toFixed(2));
  }
  updateBlurStrengthVisibility();
  updateBackgroundMaskControls();
  // Re-fire change handlers to update visual disabled states
  $("#bg-enabled").dispatchEvent(new Event("change"));

  // Grain
  $("#grain-enabled").checked = s.grain.enabled;
  setSlider($("#grain-intensity"), s.grain.intensity, $("#grain-value"), (v) => parseFloat(v).toFixed(2));
  $("#grain-enabled").dispatchEvent(new Event("change"));

  // Upscale
  $("#upscale-enabled").checked = s.upscale.enabled;
  const scaleEl = document.querySelector(`input[name='upscale-scale'][value='${s.upscale.scale}']`);
  if (scaleEl) scaleEl.checked = true;
  $("#upscale-algorithm").value = s.upscale.algorithm;
  $("#upscale-enabled").dispatchEvent(new Event("change"));

  // Filters
  $("#filters-enabled").checked = s.filters.enabled;
  setSlider($("#brightness"), s.filters.brightness, $("#brightness-value"), (v) => parseFloat(v).toFixed(2));
  setSlider($("#contrast"), s.filters.contrast, $("#contrast-value"), (v) => parseFloat(v).toFixed(2));
  setSlider($("#saturation"), s.filters.saturation, $("#saturation-value"), (v) => parseFloat(v).toFixed(2));
  setSlider($("#sharpness"), s.filters.sharpness, $("#sharpness-value"), (v) => parseFloat(v).toFixed(2));
  setSlider($("#vignette"), s.filters.vignette_strength, $("#vignette-value"), (v) => parseFloat(v).toFixed(2));
  $("#sepia").checked = !!s.filters.sepia;
  $("#anime").checked = !!s.filters.anime;
  $("#anime-style").value = s.filters.anime_style || "hayao";
  const animeControls = $("#anime-controls");
  if (animeControls) animeControls.hidden = !$("#anime").checked;
  $("#painting").checked = !!s.filters.painting;
  const paintStyle = document.querySelector(`input[name='painting-style'][value='${s.filters.painting_style || "oil"}']`);
  if (paintStyle) paintStyle.checked = true;
  const paintControls = $("#painting-controls");
  if (paintControls) paintControls.hidden = !$("#painting").checked;
  $("#filters-enabled").dispatchEvent(new Event("change"));
  syncGroupBadges();
  scheduleLiveDraw();
}

// --- On-device live preview (WebGL) ---
// Grain, colour filters, sharpness, vignette and sepia are drawn in the browser as you drag the sliders.
// Background blur/removal and the AI steps (upscale, anime, painting, inpaint) still happen on Process.

async function startLivePreview(file) {
  stopLivePreview();
  const token = ++state.live.token;
  const canvas = $("#live-canvas");
  state.live.matteKey = state.live.matteBusy = state.live.matteFailed = null;
  try {
    if (!state.live.renderer) {
      state.live.renderer = createRenderer(canvas, (msg, err) => console.warn(msg, err));
      if (!state.live.renderer) return;  // no WebGL2: the plain <img> and the server do the work
      state.live.renderer.onlost = () => { canvas.hidden = true; };
      state.live.renderer.onrestored = () => { canvas.hidden = false; scheduleLiveDraw(); };
      loadBgIntoRenderer();
    }
    const image = await loadSource(file);
    if (token !== state.live.token) { image.source.close?.(); return; }  // a newer photo arrived meanwhile
    state.live.image = image;
    state.live.renderer.setImage(image);
    state.live.renderer.setMask(null);
    scheduleLiveDraw();
  } catch (err) {
    // e.g. desktop Chrome can't decode HEIC: just keep the plain preview
    console.warn("live preview unavailable:", err);
    canvas.hidden = true;
  }
}

function stopLivePreview() {
  state.live.token++;
  state.live.image = null;
  cancelAnimationFrame(state.live.raf);
  state.live.raf = 0;
  const canvas = $("#live-canvas");
  if (canvas) canvas.hidden = true;
}

function scheduleLiveDraw() {
  if (state.live.raf || !state.live.image) return;
  state.live.raf = requestAnimationFrame(drawLive);
}

function drawLive() {
  state.live.raf = 0;
  const { renderer, image } = state.live;
  const canvas = $("#live-canvas");
  if (!renderer || !image || renderer.lost) return;
  const params = liveParams(collectSettings());
  // Background blur/remove is live once the matte has arrived; painted-background selections still wait for Process.
  if (!wantsMatte()) params.bg = null;
  else syncMatte();
  const stack = $(".preview-stack");
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  // Fit the photo into the preview box (the <img> is at most 420 px tall), never above the source size.
  const k = Math.min(1, (stack.clientWidth * dpr) / image.source.width, (420 * dpr) / image.source.height);
  const w = Math.max(1, Math.round(image.source.width * k));
  const h = Math.max(1, Math.round(image.source.height * k));
  if (!renderer.draw(params, w, h)) return;
  canvas.hidden = false;
  const finding = !!params.bg && !renderer.hasMask;
  stack.classList.toggle("is-live", !isIdentity(params));
  $(".live-badge").textContent = finding ? "Finding subject…" : "Live preview";
}

// The subject matte comes from the server once per photo + model (POST /api/v1/matte); everything after that,
// blur strength included, is composited on the phone.
const wantsMatte = () =>
  $("#bg-enabled").checked && (document.querySelector("input[name='bg-source']:checked")?.value || "auto") === "auto";

const matteKeyNow = () =>
  `${state.file.name}|${state.file.size}|${state.file.lastModified}|${$("#bg-model")?.value || "u2netp"}`;

async function syncMatte() {
  const live = state.live;
  if (!live.renderer || !state.file || !live.image) return;
  const key = matteKeyNow();
  if (live.matteKey === key || live.matteBusy === key || live.matteFailed === key) return;
  live.matteBusy = key;
  const token = live.token;
  const slow = ($("#bg-model")?.value || "u2netp") !== "u2netp";
  setStatus(slow ? "Finding the subject… the best-quality model can take 20 s or more." : "Finding the subject…");
  try {
    const form = new FormData();
    form.append("file", state.file);
    form.append("model_name", $("#bg-model")?.value || "u2netp");
    const resp = await fetch("/api/v1/matte", { method: "POST", body: form });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const matteBlob = await resp.blob();
    const bmp = await createImageBitmap(matteBlob);
    if (token !== live.token || live.matteBusy !== key) { bmp.close?.(); return; }  // photo or model changed meanwhile
    live.renderer.setMask(bmp);
    live.matteKey = key;
    live.matteBlob = matteBlob;
    setStatus("");  // Process sends it back so the server doesn't run the model a second time
  } catch (err) {
    console.warn("matte unavailable, background applies on Process:", err);
    setStatus(`Couldn't find the subject (${err.message}). Tap Process to apply the background on the server.`, "error");
    if (token === live.token) live.matteFailed = key;
  } finally {
    if (live.matteBusy === key) live.matteBusy = null;
    if (token === live.token) scheduleLiveDraw();
  }
}

let statusTimer = null;
function setStatus(msg, kind = "") {
  const sb = $("#status-bar");
  sb.textContent = msg;
  sb.className = "status-bar" + (kind ? ` ${kind}` : "");
  clearTimeout(statusTimer);
  // Success toasts fade out; errors and progress stay until replaced.
  if (kind === "success" && msg) statusTimer = setTimeout(() => setStatus(""), 4000);
}

/** Switch between the Edit view (photo + controls) and the Result view. */
function setView(view) {
  if (view === "result" && $("#nav-result").disabled) return;
  document.body.dataset.view = view;
  $("#nav-edit").classList.toggle("active", view === "edit");
  $("#nav-result").classList.toggle("active", view === "result");
  $("#nav-edit").toggleAttribute("aria-current", view === "edit");
  window.scrollTo({ top: 0 });
}

function setResultAvailable(on) {
  $("#nav-result").disabled = !on;
  if (!on) setView("edit");
}

// ============================================================================
// Mask canvas (object-removal painting)
// ============================================================================

function setupMaskCanvas() {
  const canvas = $("#mask-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  state.mask.ctx = ctx;

  // Size canvas to match the RENDERED image area, not the element box.
  // #preview uses object-fit: contain + max-height:400px, so the actual
  // image may be letterboxed inside the element. If we size the canvas to
  // the element box, mask coordinates and overlays are misaligned.
  const img = $("#preview");
  const updateSize = () => {
    const elemW = img.offsetWidth;  // layout size: unaffected by the zoom transform
    const elemH = img.offsetHeight;
    const dpr = window.devicePixelRatio || 1;

    // Compute rendered image area within the element box (object-fit: contain)
    let renderedW, renderedH, offsetX, offsetY;
    if (state.mask.imageW && state.mask.imageH) {
      const imgRatio = state.mask.imageW / state.mask.imageH;
      const elemRatio = elemW / elemH;
      if (imgRatio > elemRatio) {
        // Wider than box → constrained by width, letterboxed top/bottom
        renderedW = elemW;
        renderedH = elemW / imgRatio;
        offsetX = 0;
        offsetY = (elemH - renderedH) / 2;
      } else {
        // Taller than box → constrained by height, letterboxed left/right
        renderedH = elemH;
        renderedW = elemH * imgRatio;
        offsetX = (elemW - renderedW) / 2;
        offsetY = 0;
      }
    } else {
      renderedW = elemW;
      renderedH = elemH;
      offsetX = 0;
      offsetY = 0;
    }

    state.mask.displayW = Math.round(renderedW);
    state.mask.displayH = Math.round(renderedH);
    state.mask.offsetX = Math.round(offsetX);
    state.mask.offsetY = Math.round(offsetY);

    // Position + size canvas to exactly cover the rendered image area
    canvas.style.width = renderedW + "px";
    canvas.style.height = renderedH + "px";
    canvas.style.left = offsetX + "px";
    canvas.style.top = offsetY + "px";

    // Only resize the canvas backing store if dimensions actually changed.
    // Setting canvas.width/height clears all content — on mobile, the URL bar
    // show/hide fires resize events that would wipe the mask if we blindly reset.
    const newW = Math.round(renderedW * dpr);
    const newH = Math.round(renderedH * dpr);
    if (canvas.width !== newW || canvas.height !== newH) {
      // Save existing content before resizing
      let saved = null;
      if (canvas.width > 0 && canvas.height > 0) {
        saved = ctx.getImageData(0, 0, canvas.width, canvas.height);
      }
      canvas.width = newW;
      canvas.height = newH;
      ctx.setTransform(1, 0, 0, 1, 0, 0); // reset transform before scaling
      ctx.scale(dpr, dpr);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      // Restore content if we had any
      if (saved) {
        ctx.setTransform(1, 0, 0, 1, 0, 0); // reset to draw at backing-store scale
        ctx.putImageData(saved, 0, 0);
        ctx.scale(dpr, dpr); // re-apply dpr scale for subsequent drawing
      }
    }
  };

  // Run after the image has loaded
  if (img.complete && img.naturalWidth) {
    updateSize();
  } else {
    img.addEventListener("load", updateSize, { once: true });
  }
  // Re-fit on window resize — but debounce so mobile URL bar show/hide
  // (which fires resize) doesn't thrash the canvas.
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    if (!state.mask.imageW) return;
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => updateSize(), 150);
  });
}

// ============================================================================
// Zoom / pan of the selection image (CSS transform on #zoom-layer)
// ============================================================================

const view = { s: 1, tx: 0, ty: 0, pan: false, ptrs: new Map(), g: null, drag: null, gestured: false };
const MAX_ZOOM = 8;

function stackPoint(clientX, clientY) {
  const r = $(".preview-stack").getBoundingClientRect();
  return { x: clientX - r.left, y: clientY - r.top };
}

function applyView() {
  const layer = $("#zoom-layer");
  const stack = $(".preview-stack");
  view.s = Math.min(MAX_ZOOM, Math.max(1, view.s));
  // Keep the picture covering the box: no empty margins when zoomed in.
  view.tx = Math.min(0, Math.max(stack.clientWidth * (1 - view.s), view.tx));
  view.ty = Math.min(0, Math.max(layer.offsetHeight * (1 - view.s), view.ty));
  if (view.s === 1) view.tx = view.ty = 0;
  layer.style.transform = view.s === 1 ? "" : `translate(${view.tx}px, ${view.ty}px) scale(${view.s})`;
  const canvas = $("#mask-canvas");
  canvas.dataset.zoomed = view.s > 1 || view.pan ? "1" : "0";
  canvas.style.cursor = view.pan ? "grab" : "";
}

function zoomAt(factor, cx, cy) {
  const ns = Math.min(MAX_ZOOM, Math.max(1, view.s * factor));
  const k = ns / view.s;
  view.tx = cx - (cx - view.tx) * k;
  view.ty = cy - (cy - view.ty) * k;
  view.s = ns;
  applyView();
}

function resetView() {
  view.s = 1;
  view.tx = view.ty = 0;
  view.pan = false;
  view.ptrs.clear();
  view.g = view.drag = null;
  $("#zoom-pan")?.setAttribute("aria-pressed", "false");
  applyView();
}

function initZoom(canvas) {
  const stack = $(".preview-stack");
  const centre = () => ({ x: stack.clientWidth / 2, y: stack.clientHeight / 2 });
  $("#zoom-in").addEventListener("click", () => { const c = centre(); zoomAt(1.6, c.x, c.y); });
  $("#zoom-out").addEventListener("click", () => { const c = centre(); zoomAt(1 / 1.6, c.x, c.y); });
  $("#zoom-fit").addEventListener("click", resetView);
  $("#zoom-pan").addEventListener("click", () => {
    view.pan = !view.pan;
    $("#zoom-pan").setAttribute("aria-pressed", String(view.pan));
    applyView();
  });
  stack.addEventListener("wheel", (e) => {
    if (canvas.hidden || !(e.ctrlKey || e.metaKey)) return;  // plain scrolling still scrolls the page
    e.preventDefault();
    const p = stackPoint(e.clientX, e.clientY);
    zoomAt(Math.exp(-e.deltaY * 0.01), p.x, p.y);
  }, { passive: false });

  canvas.addEventListener("pointerdown", (e) => {
    view.ptrs.set(e.pointerId, stackPoint(e.clientX, e.clientY));
    if (view.ptrs.size === 1) {
      view.gestured = false;
      if (view.pan || e.button === 1) {
        const p = view.ptrs.get(e.pointerId);
        view.drag = { x: p.x, y: p.y, tx: view.tx, ty: view.ty };
        try { canvas.setPointerCapture(e.pointerId); } catch { /* ignore */ }
        e.preventDefault();
      }
    } else if (view.ptrs.size === 2) {
      view.gestured = true;
      view.drag = null;
      if (state.mask.drawing) {  // the first finger was the start of a pinch, not a stroke: take its dot back
        state.mask.drawing = false;
        undoMaskStroke();
      }
      const [a, b] = [...view.ptrs.values()];
      view.g = { d0: Math.hypot(a.x - b.x, a.y - b.y) || 1, mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2, s0: view.s, tx0: view.tx, ty0: view.ty };
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!view.ptrs.has(e.pointerId)) return;
    const p = stackPoint(e.clientX, e.clientY);
    view.ptrs.set(e.pointerId, p);
    if (view.ptrs.size === 2 && view.g) {
      const [a, b] = [...view.ptrs.values()];
      const g = view.g;
      const ns = Math.min(MAX_ZOOM, Math.max(1, g.s0 * (Math.hypot(a.x - b.x, a.y - b.y) / g.d0)));
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      // The picture point that was under the fingers' start midpoint stays under their current midpoint.
      view.tx = mx - ((g.mx - g.tx0) / g.s0) * ns;
      view.ty = my - ((g.my - g.ty0) / g.s0) * ns;
      view.s = ns;
      applyView();
    } else if (view.drag) {
      view.tx = view.drag.tx + (p.x - view.drag.x);
      view.ty = view.drag.ty + (p.y - view.drag.y);
      applyView();
    }
  });
  const up = (e) => {
    view.ptrs.delete(e.pointerId);
    if (view.ptrs.size < 2) view.g = null;
    view.drag = null;
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
}

function getCanvasCoords(evt) {
  const canvas = $("#mask-canvas");
  const rect = canvas.getBoundingClientRect();
  // pointer events give clientX/Y directly
  // The canvas is drawn in unzoomed CSS px; the zoom layer scales it on screen.
  const x = ((evt.clientX ?? (evt.touches && evt.touches[0]?.clientX)) - rect.left) / view.s;
  const y = ((evt.clientY ?? (evt.touches && evt.touches[0]?.clientY)) - rect.top) / view.s;
  return { x, y };
}

function pushMaskHistory() {
  if (!state.mask.ctx) return;
  const canvas = $("#mask-canvas");
  // Limit history to last 20 strokes to bound memory
  if (state.mask.history.length >= 20) state.mask.history.shift();
  state.mask.history.push(state.mask.ctx.getImageData(0, 0, canvas.width, canvas.height));
}

function startMaskStroke(evt) {
  const bgPaint = isBackgroundPaintActive();
  if (!state.mask.ctx || (!state.mask.enabled && !bgPaint)) return;
  if (view.pan || view.ptrs.size > 1 || evt.button === 1) return;  // moving the picture, not selecting
  // In click mode, object-removal click detection takes over — don't draw a paint dot.
  if (state.mask.enabled && state.mask.mode === "click") {
    onMaskClick(evt);
    return;
  }
  evt.preventDefault();
  const { x, y } = getCanvasCoords(evt);
  state.mask.drawing = true;
  state.mask.lastX = x;
  state.mask.lastY = y;
  pushMaskHistory();
  paintMaskSegment(x, y, x, y);  // dot
  state.mask.isDirty = true;
}

function continueMaskStroke(evt) {
  const bgPaint = isBackgroundPaintActive();
  if (!state.mask.drawing || !state.mask.ctx || (!state.mask.enabled && !bgPaint)) return;
  evt.preventDefault();
  const { x, y } = getCanvasCoords(evt);
  paintMaskSegment(state.mask.lastX, state.mask.lastY, x, y);
  state.mask.lastX = x;
  state.mask.lastY = y;
}

function endMaskStroke() {
  state.mask.drawing = false;
}

function paintMaskSegment(x0, y0, x1, y1) {
  const ctx = state.mask.ctx;
  // Scale brush from display px to canvas px (we used dpr scale above,
  // so the context is already in display units)
  const brushSize = state.mask.brushSize / view.s;  // the brush keeps its on-screen size, so zooming in makes it finer
  const r = brushSize / 2;
  ctx.lineWidth = brushSize;
  const paintColour = isBackgroundPaintActive()
    ? "rgba(80, 180, 255, 0.85)"
    : "rgba(255, 60, 60, 0.85)";
  // Translucent overlay so user can see the image, but the alpha channel is
  // high enough that the underlying pixels read as selected in the mask PNG.
  ctx.strokeStyle = paintColour;
  ctx.fillStyle = paintColour;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  // Dot for single taps
  if (x0 === x1 && y0 === y1) {
    ctx.beginPath();
    ctx.arc(x0, y0, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function clearMask() {
  if (!state.mask.ctx) return;
  const canvas = $("#mask-canvas");
  state.mask.ctx.clearRect(0, 0, canvas.width, canvas.height);
  state.mask.history = [];
  state.mask.isDirty = false;
}

function undoMaskStroke() {
  if (!state.mask.ctx || state.mask.history.length === 0) return;
  const canvas = $("#mask-canvas");
  const prev = state.mask.history.pop();
  state.mask.ctx.putImageData(prev, 0, 0);
  state.mask.isDirty = state.mask.history.length > 0;
}

/** Returns a Promise<Blob> PNG of the painted mask at original image resolution.
 *  Transparent/black = keep, white = selected. */
async function getPaintedMaskPngBlob(displayCanvas = $("#mask-canvas")) {
  const out = document.createElement("canvas");
  out.width = state.mask.imageW;
  out.height = state.mask.imageH;
  const octx = out.getContext("2d");
  // Draw the painted strokes (transparent background + semi-transparent overlay)
  octx.drawImage(displayCanvas, 0, 0, out.width, out.height);
  // Convert any painted pixel to opaque white; transparent stays transparent
  // (which decodes as grayscale 0 = black = keep when cv2 reads it)
  octx.globalCompositeOperation = "source-in";
  octx.fillStyle = "#ffffff";
  octx.fillRect(0, 0, out.width, out.height);
  octx.globalCompositeOperation = "source-over";
  return new Promise((resolve) => out.toBlob(resolve, "image/png"));
}

/** Returns a Promise<Blob> PNG of the inpaint mask at original image resolution.
 *  Black = keep, white = remove. */
async function getMaskPngBlob() {
  // In click mode, the detected mask is already a clean PNG at original size
  if (state.mask.mode === "click" && state.mask.detectedMask) {
    const resp = await fetch(state.mask.detectedMask);
    return await resp.blob();
  }
  return await getPaintedMaskPngBlob();
}

function hasPaintMaskContent() {
  if (!state.mask.ctx || !state.mask.isDirty) return false;
  const canvas = $("#mask-canvas");
  const data = state.mask.ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > 0) return true;
  }
  return false;
}

function hasMaskContent() {
  // Click mode: we have a mask if SAM returned one
  if (state.mask.mode === "click") return !!state.mask.detectedMask;
  return hasPaintMaskContent();
}

// ============================================================================
// SAM (point-prompt segmentation)
// ============================================================================

function clearClickSelection(keepCanvas = false) {
  state.mask.detectedMask = null;
  state.mask.detectionScore = 0;
  if (!keepCanvas && state.mask.ctx) {
    state.mask.ctx.clearRect(0, 0, $("#mask-canvas").width, $("#mask-canvas").height);
  }
}

function drawDetectedMaskOnCanvas(dataUrl) {
  if (!state.mask.ctx) return;
  const ctx = state.mask.ctx;
  // Clear the canvas first
  ctx.clearRect(0, 0, $("#mask-canvas").width, $("#mask-canvas").height);
  // Load the SAM mask image and draw it scaled to display size
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, $("#mask-canvas").width, $("#mask-canvas").height);
    ctx.drawImage(img, 0, 0, state.mask.displayW, state.mask.displayH);
  };
  img.src = dataUrl;
}

async function runSegmentation(clientX, clientY) {
  if (state.mask.segmenting) return;
  // Convert client (display) coords to original image coords
  const dispX = (clientX - $("#mask-canvas").getBoundingClientRect().left) / view.s;
  const dispY = (clientY - $("#mask-canvas").getBoundingClientRect().top) / view.s;
  const imgX = Math.round(dispX * state.mask.imageW / state.mask.displayW);
  const imgY = Math.round(dispY * state.mask.imageH / state.mask.displayH);
  // Drop a small marker at the click point so the user sees what they clicked
  if (state.mask.ctx) {
    state.mask.ctx.fillStyle = "rgba(80, 200, 255, 0.95)";  // cyan dot
    state.mask.ctx.beginPath();
    state.mask.ctx.arc(dispX, dispY, 6 / view.s, 0, Math.PI * 2);
    state.mask.ctx.fill();
  }
  // Send to server
  const formData = new FormData();
  formData.append("file", state.file);
  formData.append("x", String(imgX));
  formData.append("y", String(imgY));
  // The user may open another tool (or load another photo) before the answer comes back; it must land on this one.
  const owner = state.mask.owner;
  const file = state.file;
  const stillHere = () =>
    state.file === file && state.mask.enabled && state.mask.owner === owner && state.mask.mode === "click";
  state.mask.segmenting = true;
  const statusEl = $("#sel-hint");
  statusEl.textContent = "Detecting… (this takes ~1s)";
  try {
    const resp = await fetch("/api/v1/segment", { method: "POST", body: formData });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || body.message || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    if (!stillHere()) {
      const saved = state.selections[owner];
      if (state.file === file && saved && saved.mode === "click") {
        saved.detected = data.mask;
        saved.dirty = true;
        saved.overlayMissing = true;
      }
      return;
    }
    state.mask.detectedMask = data.mask;
    state.mask.detectionScore = data.score;
    drawDetectedMaskOnCanvas(data.mask);
    const areaPct = (data.mask_area_pct * 100).toFixed(1);
    const borderText = data.selection_border_px ? `, +${data.selection_border_px}px edge margin` : "";
    const nextStep = { blur: "Apply blur", tattoo: "Remove tattoo", facekeep: "Process" }[state.mask.owner] || "Remove object";
    statusEl.textContent =
      `Selected ${areaPct}% of the image${borderText}. Press “${nextStep}”, or tap elsewhere to re-select.`;
  } catch (err) {
    setStatus(`Segmentation error: ${err.message}`, "error");
    if (stillHere()) clearClickSelection();
  } finally {
    state.mask.segmenting = false;
  }
}

// Click handler for click-mode (separate from drag-paint handlers)
function onMaskClick(evt) {
  if (!state.mask.enabled) return;
  if (state.mask.mode !== "click") return;
  if (state.mask.segmenting) return;
  if (!state.file) return;
  // We accept only single-tap (no drag). Check that the pointer didn't move much.
  const startX = evt.clientX, startY = evt.clientY;
  const onUp = (upEvt) => {
    $("#mask-canvas").removeEventListener("pointerup", onUp);
    const dx = upEvt.clientX - startX;
    const dy = upEvt.clientY - startY;
    if (Math.hypot(dx, dy) > 10 || view.gestured) return;  // it was a drag or a pinch, not a tap
    runSegmentation(startX, startY);
  };
  $("#mask-canvas").addEventListener("pointerup", onUp, { once: true });
}

// Mode switcher
function setMaskMode(mode) {
  state.mask.mode = mode;
  $("#mask-canvas").dataset.mode = mode;
  const radio = document.querySelector(`input[name='sel-mode'][value='${mode}']`);
  if (radio) radio.checked = true;
  clearClickSelection();
  // Reset paint history
  state.mask.history = [];
  state.mask.isDirty = false;
  if (state.mask.ctx) {
    state.mask.ctx.clearRect(0, 0, $("#mask-canvas").width, $("#mask-canvas").height);
  }
}

/** "Background ✓ → Remove tattoo… → Face swap" for the progress line. */
function progressLine(steps, done) {
  return steps.map((n, i) => (i < done ? `${n} ✓` : i === done ? `${n}…` : n)).join(" → ");
}

async function processImage() {
  if (!state.file || state.processing) return;
  state.processing = true;
  updateProcessButton();
  setStatus("Processing…", "");

  const settings = collectSettings();
  const formData = new FormData();
  formData.append("file", state.file);

  // The selection tools (remove object / tattoo / blur an area) are steps of the same run: the server does the
  // background first, then each tool in turn, then the face swap, grain, upscale and filters.
  const skipped = [];
  settings.edits = [];
  for (const t of EDIT_TOOLS) {
    if (!$(`#${t}-enabled`).checked) continue;
    let blob;
    try {
      blob = await selectionBlob(t);
    } catch (err) {
      setStatus(`Error preparing the selection: ${err.message}`, "error");
      state.processing = false;
      updateProcessButton();
      return;
    }
    if (!blob) { skipped.push(TOOL_LABELS[t].title.replace(/^\S+\s/, "")); continue; }
    formData.append("edit_masks", blob, `${t}.png`);
    settings.edits.push({
      kind: t,
      algorithm: t === "inpaint" ? document.querySelector("input[name='inpaint-algo']:checked").value : "lama",
      radius: parseInt($("#inpaint-radius").value, 10),
      iterations: parseInt($("#inpaint-iterations").value, 10),
      strength: parseFloat($("#tattoo-strength").value),
      grow: parseFloat($("#tattoo-grow").value),
      blur_strength: parseInt($("#blur-area-strength").value, 10),
      target: document.querySelector("input[name='blur-target']:checked").value,
    });
  }
  const otherSteps = settings.face.enabled || settings.background.enabled || settings.grain.enabled || settings.upscale.enabled || settings.filters.enabled;
  if (skipped.length && !settings.edits.length && !otherSteps) {
    setStatus(`Select an area first (tap it or paint over it) for: ${skipped.join(", ")}.`, "error");
    state.processing = false;
    updateProcessButton();
    return;
  }
  formData.append("settings", JSON.stringify(settings));

  if (settings.background.enabled && settings.background.mode === "image") {
    if (!state.bgFile) {
      setStatus("Choose a background photo first (Background → Photo).", "error");
      state.processing = false;
      updateProcessButton();
      return;
    }
    formData.append("background_image", state.bgFile, state.bgFile.name || "background");
  }

  const live = state.live;
  if (settings.background.enabled && wantsMatte() && live?.matteBlob && live.matteKey === matteKeyNow()) {
    formData.append("matte", live.matteBlob, "matte.png");
  }

  if (settings.face.enabled) {
    if (!faceUi.donorFile) {
      setStatus("Choose the photo with the new face first (Face swap → Choose photo).", "error");
      state.processing = false;
      updateProcessButton();
      return;
    }
    formData.append("face_donor", faceUi.donorFile, faceUi.donorFile.name || "face");
    for (const f of faceUi.extraFiles) formData.append("face_donor_extra", f, f.name || "face");
    if ($("#facekeep-enabled").checked) {
      try {
        const keep = await selectionBlob("facekeep");
        if (keep) formData.append("face_keep", keep, "face-keep.png");
      } catch (err) {
        setStatus(`Error preparing the keep mask: ${err.message}`, "error");
        state.processing = false;
        updateProcessButton();
        return;
      }
    }
    setStatus(settings.face.hair ? "Swapping the face and hair… this can take 20 s or more." : "Swapping the face… this can take 10 s or more.");
  }

  if (isBackgroundPaintActive()) {
    if (!hasPaintMaskContent()) {
      setStatus("Paint at least one background area first, or switch Background source back to Auto.", "error");
      state.processing = false;
      updateProcessButton();
      return;
    }
    try {
      const bgMaskBlob = await getPaintedMaskPngBlob();
      formData.append("background_mask", bgMaskBlob, "background-mask.png");
    } catch (err) {
      setStatus(`Error preparing background mask: ${err.message}`, "error");
      state.processing = false;
      updateProcessButton();
      return;
    }
  }

  // Poll which step the server is on, so a long run shows "Background ✓ → Face swap…" instead of just waiting.
  const job = Math.random().toString(36).slice(2, 12);
  formData.append("job", job);
  let polling = true;  // cleared when the run ends, so an answer still in flight can't overwrite "Done in …"
  const poll = setInterval(async () => {
    try {
      const r = await fetch(`/api/v1/progress/${job}`);
      if (!r.ok || !polling) return;
      const p = await r.json();
      if (!polling) return;
      if (p.queued) setStatus("Waiting for another run to finish…");
      else if (p.current) setStatus(progressLine(p.steps, p.done));
    } catch { /* the next poll will try again */ }
  }, 700);
  let resp;
  try {
    resp = await fetch("/api/v1/process", { method: "POST", body: formData });
    polling = false;
    clearInterval(poll);
  } catch (networkErr) {
    polling = false;
    clearInterval(poll);
    console.error(networkErr);
    setStatus(`Network error: ${networkErr.message}`, "error");
    state.processing = false;
    updateProcessButton();
    return;
  }

  if (!resp.ok) {
    let errMsg = `HTTP ${resp.status}`;
    try {
      const errBody = await resp.json();
      // Process route uses HTTPException(detail=...) or AppError(message=...)
      errMsg = errBody.detail || errBody.message || errMsg;
    } catch {
      errMsg = resp.statusText || errMsg;
    }
    setStatus(`Error: ${errMsg}`, "error");
    state.processing = false;
    updateProcessButton();
    return;
  }

  let data;
  try {
    data = await resp.json();
  } catch (jsonErr) {
    setStatus(`Error: invalid JSON response`, "error");
    state.processing = false;
    updateProcessButton();
    return;
  }

  state.lastResult = data;
  renderOutputs(data);
  const w = data.output_size && data.output_size.width;
  const h = data.output_size && data.output_size.height;
  const sizeStr = (w && h) ? ` · ${w}×${h}` : "";
  const reused = (data.steps || []).filter((st) => st.cached).map((st) => st.name.toLowerCase());
  const reusedStr = reused.length ? ` (reused ${reused.join(", ")})` : "";
  const extra = [...(data.notes || [])];
  if (skipped.length) extra.push(`Skipped ${skipped.join(", ")}: nothing selected.`);
  setStatus(`Done in ${data.elapsed_seconds.toFixed(2)}s${sizeStr}${reusedStr}.${extra.length ? " " + extra.join(" ") : ""}`, "success");
  state.processing = false;
  updateProcessButton();
}

/** Turn the result into a downloadable PNG (and a share action where the browser has one). */
let saveUrl = null;
async function prepareSave(dataUrl) {
  const btn = $("#save-btn"), share = $("#share-btn"), hint = $("#save-hint");
  if (!btn) return;
  if (saveUrl) { URL.revokeObjectURL(saveUrl); saveUrl = null; }
  if (!dataUrl) { btn.hidden = true; share.hidden = true; hint.hidden = true; return; }
  try {
    const blob = await (await fetch(dataUrl)).blob();   // the server's PNG, alpha included, byte for byte
    saveUrl = URL.createObjectURL(blob);
    const base = ((state.file && state.file.name) || "photo").replace(/\.[^.]+$/, "");
    const name = `${base}-edited.png`;
    btn.href = saveUrl;
    btn.download = name;
    btn.hidden = false;
    hint.hidden = false;
    const file = new File([blob], name, { type: "image/png" });
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      share.hidden = false;
      share.onclick = () => navigator.share({ files: [file], title: name }).catch(() => {});
    } else {
      share.hidden = true;
    }
  } catch (err) {
    console.warn("could not prepare the download:", err);
    btn.hidden = true; share.hidden = true; hint.hidden = true;
  }
}

function renderOutputs(data) {
  $("#output-section").hidden = false;
  const map = {
    final: "out-final",
    before_after: "out-before_after",
    diff: "out-diff",
    mask: "out-mask",
    grain: "out-grain",
    upscaled: "out-upscaled",
  };
  for (const [key, id] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (!el) continue;
    const dataUrl = data[key];
    if (dataUrl) {
      el.src = dataUrl;
      el.style.display = "block";
    } else {
      el.removeAttribute("src");
      el.style.display = "none";
    }
  }

  // Disable tabs whose data wasn't returned
  const tabAvail = {
    final: !!data.final,
    before_after: !!data.before_after,
    diff: !!data.diff,
    mask: !!data.mask,
    grain: !!data.grain,
    upscaled: !!data.upscaled,
  };
  let firstActive = null;
  $$(".output-tabs .tab[data-tab]").forEach((tab) => {
    const name = tab.dataset.tab;
    const avail = tabAvail[name];
    tab.disabled = !avail;
    tab.classList.remove("active");
    if (!avail) return;
    if (!firstActive) firstActive = tab;
  });
  if (firstActive) {
    firstActive.classList.add("active");
    const targetPane = $(`.output-pane[data-pane='${firstActive.dataset.tab}']`);
    $$(".output-pane").forEach((p) => p.classList.toggle("active", p === targetPane));
  }

  // "More" only appears when there are debug views to show
  const hasDebug = $$(".output-tabs .tab-debug").some((t) => !t.disabled);
  $("#tabs-more").hidden = !hasDebug;

  setResultAvailable(true);
  setView("result");
  prepareSave(data.final);

  // Output meta: bytes / format
  const meta = $("#output-meta");
  if (meta) {
    const size = data.final ? Math.round((data.final.length * 3) / 4) : 0;
    const sizeKB = (size / 1024).toFixed(1);
    const w = data.output_size && data.output_size.width;
    const h = data.output_size && data.output_size.height;
    meta.textContent = (w && h) ? `${w}×${h} px · ~${sizeKB} KB` : "";
  }
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  // Defer registration until after first paint for faster TTI
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/sw.js")
      .catch((err) => console.warn("SW registration failed:", err));
  });
}

init().catch((err) => console.error("Init failed:", err));


// --- Face swap ---
// Detection (POST /api/v1/faces) tells us how many faces each photo has and where, so the user can tap the one to
// replace. The swap itself is a server step that runs on Process, before background and filters.

const faceUi = {
  target: 0, donorIdx: 0,          // chosen face indexes (left to right, same order as the server)
  donorFile: null, donorThumbUrl: null,
  extraUrls: [], extraFiles: [],                  // more photos of the same person (identity is averaged)
  photo: null,                     // { file, data } detection result for state.file
  donor: null,                     // { file, data } detection result for the donor photo
  models: null,                    // /api/v1/faces/status
  busy: 0,
};

function initFaceUi() {
  $("#face-donor-btn").addEventListener("click", () => $("#face-donor-input").click());
  $("#face-donor-input").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) setDonorFile(f);
  });
  $("#face-extra-btn").addEventListener("click", () => $("#face-extra-input").click());
  $("#face-extra-input").addEventListener("change", (e) => {
    const added = Array.from(e.target.files || []);
    faceUi.extraFiles = [...faceUi.extraFiles, ...added].slice(0, 8);
    e.target.value = "";
    if (added.length) setFaceNote("");
    renderFaceUi();
  });
  $("#face-all").addEventListener("change", renderFaceUi);
  $("#face-enabled").addEventListener("change", () => { if ($("#face-enabled").checked) detectFaces(); renderFaceUi(); });
  $("#face-group").addEventListener("toggle", () => { if ($("#face-group").open) detectFaces(); renderFaceUi(); });
  fetch("/api/v1/faces/status").then((r) => r.json()).then((m) => { faceUi.models = m; renderFaceUi(); }).catch(() => {});
  renderFaceUi();
}

/** The edited photo changed (new file, rotate/flip, removed): forget its faces. */
function onFaceSourceChanged() {
  faceUi.photo = null;
  faceUi.target = 0;
  if (typeof renderFaceUi === "function") {
    renderFaceUi();
    if (state.file && ($("#face-enabled")?.checked || $("#face-group")?.open)) detectFaces();
  }
}

async function setDonorFile(file) {
  faceUi.donorFile = file;
  faceUi.donor = null;
  faceUi.donorIdx = 0;
  if (faceUi.donorThumbUrl) URL.revokeObjectURL(faceUi.donorThumbUrl);
  faceUi.donorThumbUrl = URL.createObjectURL(file);
  $("#face-donor-thumb").src = faceUi.donorThumbUrl;
  renderFaceUi();
  await detectFaces();
}

async function postFaces(file) {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch("/api/v1/faces", { method: "POST", body: form });
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try { const b = await resp.json(); msg = b.detail || b.message || msg; } catch { /* keep status */ }
    throw new Error(msg);
  }
  return resp.json();
}

async function detectFaces() {
  const jobs = [];
  const photo = state.file, donor = faceUi.donorFile;
  if (photo && !faceUi.photo) jobs.push(postFaces(photo).then((data) => { if (state.file === photo) faceUi.photo = { file: photo, data }; }));
  if (donor && !faceUi.donor) jobs.push(postFaces(donor).then((data) => { if (faceUi.donorFile === donor) faceUi.donor = { file: donor, data }; }));
  if (!jobs.length) return;
  faceUi.busy++; renderFaceUi();
  try {
    await Promise.all(jobs);
  } catch (err) {
    console.warn("face detection failed:", err);
    setFaceNote(`Couldn't look for faces: ${err.message}`, true);
  } finally {
    faceUi.busy--;
    const n = faceUi.photo?.data.faces.length || 0;
    if (faceUi.target >= n) faceUi.target = 0;
    const m = faceUi.donor?.data.faces.length || 0;
    if (faceUi.donorIdx >= m) faceUi.donorIdx = 0;
    renderFaceUi();
  }
}

function setFaceNote(text, isError = false) {
  const note = $("#face-note");
  note.textContent = text;
  note.hidden = !text;
  note.classList.toggle("is-error", isError);
}

function faceChips(containerSel, count, selected, onPick) {
  const box = $(containerSel);
  box.replaceChildren();
  for (let i = 0; i < count; i++) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "face-chip"; b.textContent = String(i + 1);
    b.setAttribute("aria-label", `Face ${i + 1}`);
    b.setAttribute("aria-pressed", String(i === selected));
    b.addEventListener("click", () => onPick(i));
    box.appendChild(b);
  }
}

function renderFaceUi() {
  if (!$("#face-info")) return;
  const models = faceUi.models;
  if (models && !models.ready) {
    setFaceNote("Face swap isn't installed on this server yet (missing models). Run scripts/download_models.py.", true);
  } else if ($("#face-note").classList.contains("is-error") && models?.ready && !faceUi.busy) {
    setFaceNote("");
  }
  $("#face-restore-row").hidden = !!(models && models.ready && !models.restorer);
  $("#face-hair-row").hidden = !!(models && models.ready && !models.hair);

  const nPhoto = faceUi.photo?.data.faces.length;
  const nDonor = faceUi.donor?.data.faces.length;
  const parts = [];
  if (faceUi.busy) parts.push("Looking for faces…");
  else {
    if (!state.file) parts.push("Load a photo first.");
    else if (nPhoto === 0) parts.push("No face found in your photo.");
    else if (nPhoto > 1) parts.push(`${nPhoto} faces in your photo: tap the one to replace.`);
    else if (nPhoto === 1) parts.push("1 face in your photo.");
    if (faceUi.donorFile && nDonor === 0) parts.push("No face found in the new-face photo.");
    else if (nDonor > 1) parts.push(`${nDonor} faces in the new-face photo: pick one.`);
    if (!faceUi.donorFile) parts.push("Now choose the photo with the new face.");
  }
  $("#face-info").textContent = parts.join(" ");

  $("#face-donor-pick").hidden = !(nDonor > 1);
  if (nPhoto > 1) faceChips("#face-target-chips", nPhoto, faceUi.target, (i) => { faceUi.target = i; renderFaceUi(); });
  if (nDonor > 1) faceChips("#face-donor-chips", nDonor, faceUi.donorIdx, (i) => { faceUi.donorIdx = i; renderFaceUi(); });
  $("#face-donor-thumb").hidden = !faceUi.donorFile;
  const nx = faceUi.extraFiles.length;
  $("#face-extra-info").hidden = !nx;
  $("#face-extra-info").textContent = nx ? `${nx} extra photo${nx > 1 ? "s" : ""} of the same person will be used (tap × to remove one).` : "";
  const list = $("#face-extra-list");
  list.replaceChildren();
  for (const url of faceUi.extraUrls) URL.revokeObjectURL(url);
  faceUi.extraUrls = faceUi.extraFiles.map((f) => URL.createObjectURL(f));
  faceUi.extraFiles.forEach((f, i) => {
    const wrap = document.createElement("div");
    wrap.className = "face-extra"; wrap.setAttribute("role", "listitem");
    const img = document.createElement("img");
    img.src = faceUi.extraUrls[i]; img.alt = `Extra photo ${i + 1}`;
    const x = document.createElement("button");
    x.type = "button"; x.textContent = "×"; x.setAttribute("aria-label", `Remove extra photo ${i + 1}`);
    x.addEventListener("click", () => { faceUi.extraFiles.splice(i, 1); renderFaceUi(); });
    wrap.append(img, x);
    list.appendChild(wrap);
  });
  $("#face-all-row").hidden = !(nPhoto > 1);
  $("#face-target-pick").hidden = !(nPhoto > 1) || $("#face-all").checked;
  $("#face-texture-row").hidden = !!(models && models.ready && !models.restorer);
  $("#face-donor-btn").textContent = faceUi.donorFile ? "🙂 Change photo with the new face" : "🙂 Choose photo with the new face";
  drawFaceOverlay();
}

/** Numbered rings over the photo while the Face swap group is open; tap a ring to choose that face. */
function drawFaceOverlay() {
  const svg = $("#face-overlay");
  const data = faceUi.photo?.data;
  const show = !!data && data.faces.length > 0 && $("#face-group").open && !!state.file && !state.mask.enabled;
  svg.toggleAttribute("hidden", !show);  // SVG elements have no .hidden property
  svg.replaceChildren();
  if (!show) return;
  svg.setAttribute("viewBox", `0 0 ${data.width} ${data.height}`);  // same letterboxing as the <img> (contain)
  const ns = "http://www.w3.org/2000/svg";
  data.faces.forEach((f, i) => {
    const [x, y, w, h] = f.box;
    const ring = document.createElementNS(ns, "ellipse");
    ring.setAttribute("cx", x + w / 2); ring.setAttribute("cy", y + h / 2);
    ring.setAttribute("rx", w * 0.62); ring.setAttribute("ry", h * 0.72);
    ring.setAttribute("class", "ring" + (i === faceUi.target ? " on" : ""));
    ring.addEventListener("click", () => { faceUi.target = i; renderFaceUi(); });
    svg.appendChild(ring);
    if (data.faces.length > 1) {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", x + w / 2); t.setAttribute("y", y - h * 0.1);
      t.style.fontSize = `${Math.max(data.width, data.height) / 28}px`;
      t.textContent = String(i + 1);
      svg.appendChild(t);
    }
  });
}
