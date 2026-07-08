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
  $("#preview").src = state.previewUrl;
  $("#preview-container").hidden = false;
  $("#upload-btn").hidden = true;

  // Clear previous output when a new file is chosen
  $("#output-section").hidden = true;
  setStatus("");
  state.lastResult = null;

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

async function processImage() {
  if (!state.file || state.processing) return;
  state.processing = true;
  updateProcessButton();
  setStatus("Processing…", "");
  $("#output-section").hidden = false;
  // Scroll to output so the user sees the status update
  $("#output-section").scrollIntoView({ behavior: "smooth", block: "start" });

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

function renderOutputs(data) {
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
