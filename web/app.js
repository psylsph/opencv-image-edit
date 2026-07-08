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

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  file: null,
  previewUrl: null,
  presets: {},
  currentPreset: null,
  processing: false,
  lastResult: null,
  inpaintResult: null,   // separate from main process result
  // Mask canvas drawing state
  mask: {
    enabled: false,
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
  background: { enabled: false, mode: "blur", blur_strength: 15, model_name: "u2netp" },
  grain: { enabled: false, intensity: 0.4 },
  upscale: { enabled: false, scale: 2, algorithm: "interp" },
  filters: {
    enabled: false,
    brightness: 1, contrast: 1, saturation: 1, sharpness: 1,
    vignette_strength: 0, sepia: false, grayscale_blend: 0,
  },
};

// --- Init ---
async function init() {
  await loadPresets();
  renderPresetPills();
  bindEvents();
  applyPreset(null); // start at defaults
  setMaskMode("click"); // initialize mask mode UI
  registerServiceWorker();
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

  // Reset button
  $("#reset-btn").addEventListener("click", () => applyPreset(null));

  // Background enabled toggle: dim sub-controls when off
  $("#bg-enabled").addEventListener("change", (e) => {
    setGroupDisabled("#bg-mode", !e.target.checked);
    setRowDisabled("#blur-strength-row", !e.target.checked);
  });

  // Background mode: hide blur-strength row when in "remove" mode
  $$("input[name='bg-mode']").forEach((r) => {
    r.addEventListener("change", updateBlurStrengthVisibility);
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

  // Filters enabled toggle
  $("#filters-enabled").addEventListener("change", (e) => {
    const body = $("#filters-group .control-body");
    if (body) {
      body.style.opacity = e.target.checked ? "1" : "0.5";
      body.style.pointerEvents = e.target.checked ? "auto" : "none";
    }
  });

  // Auto-expand a section's <details> when its toggle is switched on
  $$(".control-group summary input[type='checkbox']").forEach((cb) => {
    cb.addEventListener("change", () => {
      const details = cb.closest("details");
      if (details && cb.checked) details.open = true;
    });
  });

  // Inpaint (object removal) toggle
  $("#inpaint-enabled").addEventListener("change", (e) => {
    state.mask.enabled = e.target.checked;
    $("#mask-canvas").hidden = !e.target.checked;
    if (e.target.checked) {
      if (state.mask.imageW) setupMaskCanvas();
    } else {
      clearClickSelection();
    }
  });
  // Mode switcher (Click vs Paint)
  $$("input[name='inpaint-mode']").forEach((r) => {
    r.addEventListener("change", (e) => setMaskMode(e.target.value));
  });
  // Brush size live update
  $("#brush-size").addEventListener("input", (e) => {
    state.mask.brushSize = parseInt(e.target.value, 10);
    $("#brush-value").textContent = state.mask.brushSize;
  });
  // Fill radius output update
  $("#inpaint-radius").addEventListener("input", (e) => {
    $("#inpaint-radius-value").textContent = e.target.value;
  });
  // Hide fill radius when AI (LaMa/SD) is selected — they don't use the param
  function updateInpaintRadiusVisibility() {
    const checked = document.querySelector("input[name='inpaint-algo']:checked");
    if (!checked) return;
    const val = checked.value;
    const row = $("#inpaint-radius-row");
    const promptRow = $("#inpaint-prompt-row");
    // LaMa and SD don't use radius; only NS/TELEA do
    if (row) row.style.display = (val === "ns" || val === "telea") ? "flex" : "none";
    // Prompt field shows for SD only
    if (promptRow) promptRow.style.display = val === "sd" ? "flex" : "none";
  }
  $$("input[name='inpaint-algo']").forEach((r) => {
    r.addEventListener("change", updateInpaintRadiusVisibility);
  });
  updateInpaintRadiusVisibility();

  // --- Generative model status (SD only) ---
  let sdAvailable = false;
  let sdPollTimer = null;

  async function checkSdStatus() {
    try {
      const resp = await fetch("/api/v1/sd/status");
      const data = await resp.json();
      sdAvailable = data.available;
      updateGenWarning();
    } catch {
      // endpoint not available — assume SD not set up
    }
  }

  function updateGenWarning() {
    const checked = document.querySelector("input[name='inpaint-algo']:checked");
    if (!checked) return;
    const warning = $("#gen-warning");
    const warningText = $("#gen-warning-text");
    const dlBtn = $("#sd-download-btn");
    if (!warning) return;

    if (checked.value === "sd" && !sdAvailable) {
      warning.hidden = false;
      if (warningText) warningText.textContent = "🎨 SD models not downloaded (~4GB).";
      if (dlBtn) dlBtn.style.display = "";
    } else {
      warning.hidden = true;
    }
  }

  // Re-check warning visibility when algorithm changes
  $$("#inpaint-algo input[type='radio']").forEach((r) => {
    r.addEventListener("change", updateGenWarning);
  });

  // Download button handler
  $("#sd-download-btn")?.addEventListener("click", async () => {
    const btn = $("#sd-download-btn");
    const progress = $("#sd-progress");
    const progressText = $("#sd-progress-text");
    const logEl = $("#sd-log");
    btn.disabled = true;
    btn.textContent = "Starting…";
    progress.hidden = false;

    try {
      const resp = await fetch("/api/v1/sd/download", { method: "POST" });
      const data = await resp.json();

      if (!data.started && data.task_id) {
        // Already running
        pollSdDownload(data.task_id);
        return;
      }
      if (!data.started) {
        progressText.textContent = "Error: " + (data.message || "could not start");
        btn.disabled = false;
        btn.textContent = "Download";
        return;
      }
      pollSdDownload(data.task_id);
    } catch (err) {
      progressText.textContent = "Error: " + err.message;
      btn.disabled = false;
      btn.textContent = "Download";
    }
  });

  function pollSdDownload(taskId) {
    const btn = $("#sd-download-btn");
    const progressText = $("#sd-progress-text");
    const logEl = $("#sd-log");
    if (sdPollTimer) clearInterval(sdPollTimer);

    sdPollTimer = setInterval(async () => {
      try {
        const resp = await fetch(`/api/v1/sd/download/${taskId}/status`);
        const data = await resp.json();

        if (data.status === "running") {
          progressText.textContent = `Downloading… (${data.elapsed_seconds}s)`;
          logEl.textContent = data.log_tail || "";
          logEl.scrollTop = logEl.scrollHeight;
        } else if (data.status === "completed") {
          clearInterval(sdPollTimer);
          sdPollTimer = null;
          progressText.textContent = "✅ Download complete!";
          sdAvailable = true;
          updateSdWarning();
        } else {
          clearInterval(sdPollTimer);
          sdPollTimer = null;
          progressText.textContent = `❌ Download failed (exit ${data.returncode})`;
          logEl.textContent = data.log_tail || "";
          btn.disabled = false;
          btn.textContent = "Retry";
        }
      } catch {
        // network blip — keep polling
      }
    }, 5000);
  }

  // Check SD + OpenAI status on page load
  checkSdStatus();
  // Undo + Clear (paint mode)
  $("#inpaint-undo").addEventListener("click", undoMaskStroke);
  $("#inpaint-clear").addEventListener("click", clearMask);
  // Click-mode actions
  $("#inpaint-accept").addEventListener("click", () => {
    // Hand off to processInpaint by setting a flag, then triggering processImage
    processImage();
  });
  $("#inpaint-retry").addEventListener("click", () => {
    clearClickSelection();
    $("#inpaint-click-status").hidden = false;
    $("#inpaint-click-status").textContent = "Tap another point to try again.";
  });
  $("#inpaint-cancel").addEventListener("click", () => {
    clearClickSelection();
  });

  // Canvas pointer events for painting
  const canvas = $("#mask-canvas");
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
  ];
  sliders.forEach(([slider, output, fmt]) => {
    const s = $(slider), o = $(output);
    if (!s || !o) return;
    s.addEventListener("input", () => { o.textContent = fmt(s.value); });
  });

  // Output tabs
  $$(".output-tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      if (tab.disabled) return;
      const name = tab.dataset.tab;
      $$(".output-tabs .tab").forEach((t) => t.classList.toggle("active", t === tab));
      $$(".output-pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === name));
    });
  });
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
    row.style.pointerEvents = disabled ? "auto" : "auto";  // sliders still respond for visual feedback
    const input = row.querySelector("input, select");
    if (input) input.disabled = disabled;
  }
}

function updateBlurStrengthVisibility() {
  const checked = document.querySelector("input[name='bg-mode']:checked");
  if (!checked) return;
  const remove = checked.value === "remove";
  const row = $("#blur-strength-row");
  if (row) row.style.display = remove ? "none" : "flex";
}

function onFileSelected(e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  // Revoke previous preview URL
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);

  state.file = file;
  state.previewUrl = URL.createObjectURL(file);

  // Wire up natural-dimension capture for the mask canvas
  const img = $("#preview");
  img.src = state.previewUrl;
  img.onload = () => {
    state.mask.imageW = img.naturalWidth;
    state.mask.imageH = img.naturalHeight;
    if (state.mask.enabled) setupMaskCanvas();
  };

  $("#preview-container").hidden = false;
  $("#upload-btn").hidden = true;

  // Reset mask on new file
  if (state.mask.ctx) clearMask();
  state.mask.history = [];
  state.mask.isDirty = false;

  // Clear previous output when a new file is chosen
  $("#output-section").hidden = true;
  setStatus("");
  state.lastResult = null;
  state.inpaintResult = null;

  updateProcessButton();
}

function clearFile() {
  state.file = null;
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }
  $("#file-input").value = "";
  $("#preview").removeAttribute("src");
  $("#preview-container").hidden = true;
  $("#upload-btn").hidden = false;
  $("#output-section").hidden = true;
  setStatus("");
  state.lastResult = null;
  state.inpaintResult = null;
  // Reset mask state
  state.mask.imageW = 0;
  state.mask.imageH = 0;
  if (state.mask.ctx) clearMask();
  state.mask.history = [];
  state.mask.isDirty = false;
  updateProcessButton();
}

function updateProcessButton() {
  const btn = $("#process-btn");
  btn.disabled = !state.file || state.processing;
  if (state.processing) {
    btn.textContent = "⏳ Processing…";
  } else {
    btn.textContent = "🚀 Process Image";
  }
}

function collectSettings() {
  const checkedRadio = (name) => {
    const el = document.querySelector(`input[name='${name}']:checked`);
    return el ? el.value : null;
  };
  return {
    background: {
      enabled: $("#bg-enabled").checked,
      mode: checkedRadio("bg-mode") || "blur",
      blur_strength: parseInt($("#blur-strength").value, 10),
      model_name: "u2netp",
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

  // Background
  $("#bg-enabled").checked = s.background.enabled;
  const bgMode = document.querySelector(`input[name='bg-mode'][value='${s.background.mode}']`);
  if (bgMode) bgMode.checked = true;
  setSlider($("#blur-strength"), s.background.blur_strength, $("#blur-value"));
  updateBlurStrengthVisibility();
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
  $("#filters-enabled").dispatchEvent(new Event("change"));
}

function setStatus(msg, kind = "") {
  const sb = $("#status-bar");
  sb.textContent = msg;
  sb.className = "status-bar" + (kind ? ` ${kind}` : "");
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
    const rect = img.getBoundingClientRect();
    const elemW = rect.width;
    const elemH = rect.height;
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

function getCanvasCoords(evt) {
  const canvas = $("#mask-canvas");
  const rect = canvas.getBoundingClientRect();
  // pointer events give clientX/Y directly
  const x = (evt.clientX ?? (evt.touches && evt.touches[0]?.clientX)) - rect.left;
  const y = (evt.clientY ?? (evt.touches && evt.touches[0]?.clientY)) - rect.top;
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
  if (!state.mask.enabled || !state.mask.ctx) return;
  // In click mode, the click handler takes over — don't draw a paint dot
  if (state.mask.mode === "click") {
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
  if (!state.mask.drawing || !state.mask.ctx) return;
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
  const r = state.mask.brushSize / 2;
  ctx.lineWidth = state.mask.brushSize;
  // Translucent red so user can see what they're painting, but the alpha
  // channel is high enough that the underlying pixels read as "to remove"
  ctx.strokeStyle = "rgba(255, 60, 60, 0.85)";
  ctx.fillStyle = "rgba(255, 60, 60, 0.85)";
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

/** Returns a Promise<Blob> PNG of the mask at original image resolution.
 *  Black = keep, white = remove. */
async function getMaskPngBlob() {
  // In click mode, the detected mask is already a clean PNG at original size
  if (state.mask.mode === "click" && state.mask.detectedMask) {
    const resp = await fetch(state.mask.detectedMask);
    return await resp.blob();
  }
  // Paint mode: render the display canvas (with red strokes) into a clean
  // black/white mask at original image resolution.
  // Do NOT pre-fill with opaque black — that would make source-in fill the
  // entire canvas white (every pixel becomes opaque). Instead, draw the
  // strokes on a transparent background, then convert non-transparent areas
  // to white via source-in.
  const displayCanvas = $("#mask-canvas");
  const out = document.createElement("canvas");
  out.width = state.mask.imageW;
  out.height = state.mask.imageH;
  const octx = out.getContext("2d");
  // Draw the painted strokes (transparent background + semi-transparent red)
  octx.drawImage(displayCanvas, 0, 0, out.width, out.height);
  // Convert any painted pixel to opaque white; transparent stays transparent
  // (which decodes as grayscale 0 = black = keep when cv2 reads it)
  octx.globalCompositeOperation = "source-in";
  octx.fillStyle = "#ffffff";
  octx.fillRect(0, 0, out.width, out.height);
  octx.globalCompositeOperation = "source-over";
  return new Promise((resolve) => out.toBlob(resolve, "image/png"));
}

function hasMaskContent() {
  // Click mode: we have a mask if SAM returned one
  if (state.mask.mode === "click") return !!state.mask.detectedMask;
  // Paint mode: scan the canvas for any non-transparent pixel
  if (!state.mask.ctx || !state.mask.isDirty) return false;
  const canvas = $("#mask-canvas");
  const data = state.mask.ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > 0) return true;
  }
  return false;
}

// ============================================================================
// SAM (point-prompt segmentation)
// ============================================================================

function clearClickSelection() {
  state.mask.detectedMask = null;
  state.mask.detectionScore = 0;
  if (state.mask.ctx) state.mask.ctx.clearRect(0, 0, $("#mask-canvas").width, $("#mask-canvas").height);
  $("#inpaint-click-actions").hidden = true;
  $("#inpaint-click-status").hidden = true;
  $("#inpaint-click-status").textContent = "";
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
  const dispX = clientX - $("#mask-canvas").getBoundingClientRect().left;
  const dispY = clientY - $("#mask-canvas").getBoundingClientRect().top;
  const imgX = Math.round(dispX * state.mask.imageW / state.mask.displayW);
  const imgY = Math.round(dispY * state.mask.imageH / state.mask.displayH);
  // Drop a small marker at the click point so the user sees what they clicked
  if (state.mask.ctx) {
    state.mask.ctx.fillStyle = "rgba(80, 200, 255, 0.95)";  // cyan dot
    state.mask.ctx.beginPath();
    state.mask.ctx.arc(dispX, dispY, 6, 0, Math.PI * 2);
    state.mask.ctx.fill();
  }
  // Send to server
  const formData = new FormData();
  formData.append("file", state.file);
  formData.append("x", String(imgX));
  formData.append("y", String(imgY));
  state.mask.segmenting = true;
  $("#inpaint-click-status").hidden = false;
  $("#inpaint-click-status").textContent = "Detecting… (this takes ~1s)";
  setStatus("Detecting object…", "");
  try {
    const resp = await fetch("/api/v1/segment", { method: "POST", body: formData });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || body.message || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    state.mask.detectedMask = data.mask;
    state.mask.detectionScore = data.score;
    drawDetectedMaskOnCanvas(data.mask);
    const areaPct = (data.mask_area_pct * 100).toFixed(1);
    $("#inpaint-click-status").textContent =
      `Detected (IoU ${data.score.toFixed(2)}, ${areaPct}% of image). ✅ Remove or 🎯 Try again.`;
    $("#inpaint-click-actions").hidden = false;
    setStatus(`Detected (IoU ${data.score.toFixed(2)}, ${areaPct}%)`, "");
  } catch (err) {
    setStatus(`Segmentation error: ${err.message}`, "error");
    clearClickSelection();
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
    if (Math.hypot(dx, dy) > 10) return;  // it was a drag, not a tap
    runSegmentation(startX, startY);
  };
  $("#mask-canvas").addEventListener("pointerup", onUp, { once: true });
}

// Mode switcher
function setMaskMode(mode) {
  state.mask.mode = mode;
  clearClickSelection();
  // Reset paint history
  state.mask.history = [];
  state.mask.isDirty = false;
  if (state.mask.ctx) {
    state.mask.ctx.clearRect(0, 0, $("#mask-canvas").width, $("#mask-canvas").height);
  }
  // Show/hide sub-controls
  $("#inpaint-paint-controls").hidden = (mode !== "paint");
  $("#inpaint-click-controls").hidden = (mode !== "click");
  // Update hint
  $("#inpaint-hint").textContent =
    mode === "click"
      ? "Tap an object to detect it automatically, then remove."
      : "Drag your finger over the object to remove it. Brush size below.";
}

async function processImage() {
  if (!state.file || state.processing) return;
  state.processing = true;
  updateProcessButton();
  setStatus("Processing…", "");
  $("#output-section").hidden = false;
  // Scroll to output so the user sees the status update
  $("#output-section").scrollIntoView({ behavior: "smooth", block: "start" });

  // Branch: if Remove Object is on AND the user has painted a mask, hit
  // the dedicated /api/v1/inpaint endpoint (it accepts a separate mask file).
  if (state.mask.enabled && hasMaskContent()) {
    await processInpaint();
    return;
  }

  const settings = collectSettings();
  const formData = new FormData();
  formData.append("file", state.file);
  formData.append("settings", JSON.stringify(settings));

  let resp;
  try {
    resp = await fetch("/api/v1/process", { method: "POST", body: formData });
  } catch (networkErr) {
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
  setStatus(`Done in ${data.elapsed_seconds.toFixed(2)}s${sizeStr}`, "success");
  state.processing = false;
  updateProcessButton();
}

async function processInpaint() {
  const radius = parseInt($("#inpaint-radius").value, 10);
  const algorithm = document.querySelector("input[name='inpaint-algo']:checked").value;
  const prompt = $("#inpaint-prompt") ? $("#inpaint-prompt").value : "";

  let maskBlob;
  try {
    maskBlob = await getMaskPngBlob();
  } catch (err) {
    setStatus(`Error preparing mask: ${err.message}`, "error");
    state.processing = false;
    updateProcessButton();
    return;
  }
  if (!maskBlob) {
    setStatus("Error: could not serialize mask", "error");
    state.processing = false;
    updateProcessButton();
    return;
  }

  const formData = new FormData();
  formData.append("file", state.file);
  formData.append("mask", maskBlob, "mask.png");
  formData.append("radius", String(radius));
  formData.append("algorithm", algorithm);
  if (algorithm === "sd" && prompt) {
    formData.append("prompt", prompt);
  }

  let resp;
  try {
    resp = await fetch("/api/v1/inpaint", { method: "POST", body: formData });
  } catch (networkErr) {
    setStatus(`Network error: ${networkErr.message}`, "error");
    state.processing = false;
    updateProcessButton();
    return;
  }

  if (!resp.ok) {
    let errMsg = `HTTP ${resp.status}`;
    try {
      const errBody = await resp.json();
      errMsg = errBody.detail || errBody.message || errMsg;
    } catch { /* ignore */ }
    setStatus(`Error: ${errMsg}`, "error");
    state.processing = false;
    updateProcessButton();
    return;
  }

  let data;
  try {
    data = await resp.json();
  } catch (jsonErr) {
    setStatus("Error: invalid JSON response", "error");
    state.processing = false;
    updateProcessButton();
    return;
  }

  // Build a normal-looking result for renderOutputs
  const w = data.output_size && data.output_size.width;
  const h = data.output_size && data.output_size.height;
  const wrapped = {
    final: data.final,
    inpainted: data.final,   // also show under the dedicated tab
    output_size: data.output_size,
    elapsed_seconds: 0,       // server doesn't return this; show 0
  };
  state.lastResult = wrapped;
  state.inpaintResult = data;
  renderOutputs(wrapped);
  const sizeStr = (w && h) ? ` · ${w}×${h}` : "";
  setStatus(`Inpainted (${algorithm}, r=${radius})${sizeStr}`, "success");
  state.processing = false;
  updateProcessButton();
}

function renderOutputs(data) {
  const map = {
    final: "out-final",
    inpainted: "out-inpainted",
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
    inpainted: !!data.inpainted,
    before_after: !!data.before_after,
    diff: !!data.diff,
    mask: !!data.mask,
    grain: !!data.grain,
    upscaled: !!data.upscaled,
  };
  let firstActive = null;
  $$(".output-tabs .tab").forEach((tab) => {
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
