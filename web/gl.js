// On-device live preview (WebGL2).
//
// Mirrors the server's pipeline maths (app/pipeline/grain.py + filters.py) in the same order:
//   grain -> brightness -> contrast -> saturation -> sharpness -> vignette -> sepia -> grayscale
// The server stays the reference implementation; this is for instant feedback while dragging sliders.
// Known, accepted differences: the noise generator is a hash (not numpy's RNG), the blur edge handling
// is clamp (OpenCV reflects), and the preview runs at screen resolution rather than the processed size.
//
// Background blur / remove use a matte fetched once from the server (POST /api/v1/matte): the subject mask doesn't
// change when sliders move, so the blur strength and the composite are live on the phone.
//
// Everything is `highp`: Apple GPUs run `mediump` as 16-bit floats, which bands visibly.

const MAX_SOURCE = 2048; // longest edge of the texture kept for previewing (memory friendly on phones)
const SERVER_MAX = 1536; // server downsizes to this (MAX_IMAGE_DIMENSION), blur sigma is relative to it

const VERT = `#version 300 es
in vec2 p;
uniform float uFlip;
out vec2 uv;
void main() {
  uv = vec2(p.x * 0.5 + 0.5, uFlip > 0.5 ? 0.5 - p.y * 0.5 : p.y * 0.5 + 0.5);
  gl_Position = vec4(p, 0.0, 1.0);
}`;

// Pass 0: grain + brightness + contrast + saturation. Reads the photo, writes an 8-bit texture
// (the server also works in uint8, so quantising here keeps the passes comparable).
const FRAG_BASE = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uSize;
uniform float uGrain, uBright, uContrast, uSat, uSeed;
in vec2 uv;
out vec4 o;
float h(vec3 q) {
  q = fract(q * vec3(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
float gauss(vec3 q) {
  float u1 = max(h(q), 1e-4);
  float u2 = h(q + 17.7);
  return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}
void main() {
  vec3 c = texture(uTex, uv).rgb;
  if (uGrain > 0.0) {
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    vec2 px = floor(uv * uSize);
    vec3 n = vec3(gauss(vec3(px, uSeed)), gauss(vec3(px, uSeed + 1.0)), gauss(vec3(px, uSeed + 2.0)));
    c = clamp(c + n * (uGrain * 32.0 / 255.0) * (1.0 - l), 0.0, 1.0);
  }
  c = clamp(c * uBright, 0.0, 1.0);
  c = clamp(0.5019608 + uContrast * (c - 0.5019608), 0.0, 1.0);
  float mx = max(c.r, max(c.g, c.b));
  float mn = min(c.r, min(c.g, c.b));
  if (mx > 0.0 && mx > mn && uSat != 1.0) {
    float s = (mx - mn) / mx;
    float k = clamp(s * uSat, 0.0, 1.0) / s;
    c = mx - (mx - c) * k;
  }
  o = vec4(c, 1.0);
}`;

// Background: subject over a blurred copy, a checkerboard ("remove" preview), a flat colour or another photo.
// Colour/photo modes also refine the matte like the server's refine_matte(): optional pre-blurred matte (feather),
// then a levels adjustment (shrink: > 0 edge in, < 0 edge out).
const FRAG_BG = `#version 300 es
precision highp float;
uniform sampler2D uSrc, uBlur, uMask, uBgImg, uFg;
uniform float uMode;   // 0 blur, 1 remove, 2 colour, 3 photo
uniform float uShrink, uUseBlurMask, uDefringe;
uniform vec3 uColor;
uniform vec4 uRect;    // visible part of the photo: x, y, w, h as fractions (y from the top)
uniform vec2 uSize;
in vec2 uv;
out vec4 o;
void main() {
  vec2 f = vec2(uv.x, 1.0 - uv.y);   // uBlur was rendered, so its rows run the other way
  vec3 s = texture(uSrc, uv).rgb;
  float m = (uMode > 1.5 && uUseBlurMask > 0.5) ? texture(uBlur, f).r : texture(uMask, uv).r;
  vec3 bg;
  if (uMode > 2.5) {
    bg = texture(uBgImg, uRect.xy + uv * uRect.zw).rgb;
  } else if (uMode > 1.5) {
    bg = uColor;
  } else if (uMode > 0.5) {
    vec2 g = floor(uv * uSize / 16.0);
    bg = vec3(mix(0.80, 0.62, mod(g.x + g.y, 2.0)));
  } else {
    bg = texture(uBlur, f).rgb;
  }
  if (uMode > 1.5) {
    if (uShrink > 0.0) { float k = 0.5 * uShrink; m = (m - k) / (1.0 - k); }
    else if (uShrink < 0.0) { float k = 0.5 * uShrink; m = m / (1.0 + k); }
    m = clamp(m, 0.0, 1.0);
  }
  if (uMode > 1.5 && uDefringe > 0.0) {
    // Edge pixels still carry some of the old background; pull them toward the subject's own colour nearby.
    float core = clamp((texture(uMask, uv).r - 0.9) / 0.1, 0.0, 1.0);
    vec4 fg = texture(uFg, f);
    vec3 own = fg.rgb / max(fg.a, 1e-4);
    s += (own - s) * (uDefringe * (1.0 - core) * clamp(fg.a / 0.1, 0.0, 1.0));
  }
  o = vec4(mix(bg, s, m), 1.0);
}`;

// Colour-fringe removal, step 1: (colour * core, core) where "core" is the solidly-subject part of the matte.
// Blurred, the ratio of the two is the subject's own colour spread outward (normalised convolution).
const FRAG_CORE = `#version 300 es
precision highp float;
uniform sampler2D uSrc, uMask;
in vec2 uv;
out vec4 o;
void main() {
  float core = clamp((texture(uMask, uv).r - 0.9) / 0.1, 0.0, 1.0);
  o = vec4(texture(uSrc, uv).rgb * core, core);
}`;

// Separable gaussian (one direction per pass).
const FRAG_BLUR = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uStep;   // one texel along the blur direction
uniform float uSigma;
in vec2 uv;
out vec4 o;
void main() {
  vec4 sum = texture(uTex, uv);
  float wsum = 1.0;
  for (int i = 1; i <= 24; i++) {
    float fi = float(i);
    if (fi > uSigma * 3.0) break;
    float w = exp(-(fi * fi) / (2.0 * uSigma * uSigma));
    sum += w * (texture(uTex, uv + uStep * fi) + texture(uTex, uv - uStep * fi));
    wsum += 2.0 * w;
  }
  o = sum / wsum;   // all four channels: the colour-fringe step blurs (colour * core, core)
}`;

// Final pass: sharpness (blend with the blurred copy), vignette, sepia, grayscale -> canvas.
const FRAG_FINAL = `#version 300 es
precision highp float;
uniform sampler2D uA, uB;
uniform float uSharp, uVig, uSepia, uGray;
in vec2 uv;
out vec4 o;
void main() {
  vec3 a = texture(uA, uv).rgb;
  vec3 c = a;
  if (uSharp < 1.0) c = a * uSharp + texture(uB, uv).rgb * (1.0 - uSharp);
  else if (uSharp > 1.0) c = a * uSharp - texture(uB, uv).rgb * (uSharp - 1.0);
  c = clamp(c, 0.0, 1.0);
  if (uVig > 0.0) c *= clamp(1.0 - length(uv * 2.0 - 1.0) * uVig, 0.0, 1.0);
  if (uSepia > 0.5) {
    // Same numbers as the server's apply_sepia(): its kernel rows are applied to BGR pixels, so the textbook
    // matrix ends up with the red and blue inputs swapped (c.bgr below). Kept identical on purpose.
    vec3 q = c.bgr;
    c = clamp(vec3(dot(q, vec3(0.393, 0.769, 0.189)),
                   dot(q, vec3(0.349, 0.686, 0.168)),
                   dot(q, vec3(0.272, 0.534, 0.131))), 0.0, 1.0);
  }
  if (uGray > 0.0) c = mix(c, vec3(dot(c, vec3(0.299, 0.587, 0.114))), uGray);
  o = vec4(c, 1.0);
}`;

/** Turn the form's settings object (same shape the server takes) into shader parameters. */
export function liveParams(settings) {
  const g = settings.grain || {};
  const f = settings.filters || {};
  const fOn = !!f.enabled;
  const b = settings.background || {};
  const modes = ["blur", "remove", "color", "image"];
  const bg = !b.enabled ? null : {
    mode: modes.includes(b.mode) ? b.mode : "blur",
    strength: Math.max(1, Number(b.blur_strength) || 15),
    color: /^#[0-9a-f]{6}$/i.test(b.color || "") ? b.color : "#00b140",
    zoom: Math.max(1, Number(b.zoom) || 1),
    offsetX: Math.min(1, Math.max(-1, Number(b.offset_x) || 0)),
    offsetY: Math.min(1, Math.max(-1, Number(b.offset_y) || 0)),
    feather: Math.min(1, Math.max(0, Number(b.feather) || 0)),
    shrink: Math.min(1, Math.max(-1, Number(b.shrink) || 0)),
    defringe: Math.min(1, Math.max(0, Number(b.defringe) || 0)),
  };
  return {
    bg,
    grain: g.enabled ? Math.max(0, Number(g.intensity) || 0) : 0,
    brightness: fOn ? Number(f.brightness) : 1,
    contrast: fOn ? Number(f.contrast) : 1,
    saturation: fOn ? Number(f.saturation) : 1,
    sharpness: fOn ? Number(f.sharpness) : 1,
    vignette: fOn ? Math.max(0, Number(f.vignette_strength) || 0) : 0,
    sepia: fOn && !!f.sepia,
    gray: fOn ? Math.min(1, Math.max(0, Number(f.grayscale_blend) || 0)) : 0,
  };
}

/** True when `params` would leave the picture unchanged. */
export function isIdentity(p) {
  return !p.bg && !p.grain && p.brightness === 1 && p.contrast === 1 && p.saturation === 1 &&
    p.sharpness === 1 && !p.vignette && !p.sepia && !p.gray;
}

/** Decode a File/Blob and shrink it to at most `max` px on the long edge. Returns {source, width, height}. */
export async function loadSource(blob, max = MAX_SOURCE) {
  const bmp = await createImageBitmap(blob); // honours EXIF orientation
  const width = bmp.width;
  const height = bmp.height;
  const k = Math.min(1, max / Math.max(width, height));
  if (k === 1) return { source: bmp, width, height };
  const cv = document.createElement("canvas");
  cv.width = Math.max(1, Math.round(width * k));
  cv.height = Math.max(1, Math.round(height * k));
  const ctx = cv.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, cv.width, cv.height);
  bmp.close?.();
  return { source: cv, width, height };
}

/** Mirror of app/pipeline/background.py cover_rect(): the visible part of a bw x bh photo in a w x h frame. */
export function coverRect(bw, bh, w, h, zoom = 1, ox = 0, oy = 0) {
  const scale = Math.max(w / bw, h / bh) * Math.max(1, zoom);
  const vw = w / scale;
  const vh = h / scale;
  const cx = Math.min(1, Math.max(-1, ox));
  const cy = Math.min(1, Math.max(-1, oy));
  return [((bw - vw) / 2) * (1 + cx), ((bh - vh) / 2) * (1 + cy), vw, vh];
}

const hexToRgb = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16) / 255);

/**
 * Create a renderer on `canvas`. Returns null when WebGL2 or the shaders aren't available
 * (callers keep showing the plain <img> and the server path does the work).
 */
export function createRenderer(canvas, onError = () => {}) {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, premultipliedAlpha: false });
  if (!gl) return null;

  let progs;
  try {
    progs = {
      base: program(gl, VERT, FRAG_BASE),
      blur: program(gl, VERT, FRAG_BLUR),
      bg: program(gl, VERT, FRAG_BG),
      core: program(gl, VERT, FRAG_CORE),
      final: program(gl, VERT, FRAG_FINAL),
    };
  } catch (err) {
    onError("webgl shader failed", err);
    return null;
  }

  let quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

  let srcTex = null;
  let src = null;      // {source, width, height} so a lost context can be rebuilt
  let lost = false;
  let bgTex = null;
  let bgImage = null;  // {source,width,height} of the replacement photo
  let maskTex = null;
  let maskSrc = null;  // kept so a lost context can rebuild the texture
  const targets = []; // four render targets at output size
  const small = [];   // two quarter-ish size targets for the (downsampled) background blur
  let tw = 0, th = 0, sw = 0, sh = 0;

  function makeTarget(w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo };
  }

  function ensureTargets(w, h) {
    if (w === tw && h === th && targets.length) return;
    for (const t of targets) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); }
    targets.length = 0;
    for (let i = 0; i < 4; i++) targets.push(makeTarget(w, h));
    tw = w; th = h;
  }

  function ensureSmall(w, h) {
    if (w === sw && h === sh && small.length) return;
    for (const t of small) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); }
    small.length = 0;
    for (let i = 0; i < 2; i++) small.push(makeTarget(w, h));
    sw = w; sh = h;
  }

  function uploadBgImage() {
    if (bgTex) gl.deleteTexture(bgTex);
    bgTex = null;
    if (!bgImage) return;
    bgTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, bgTex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bgImage.source);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  function uploadMask() {
    if (maskTex) gl.deleteTexture(maskTex);
    maskTex = null;
    if (!maskSrc) return;
    maskTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, maskTex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, maskSrc);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  function uploadSource() {
    if (srcTex) gl.deleteTexture(srcTex);
    srcTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src.source);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  function pass(prog, target, textures, setUniforms, vw = tw, vh = th) {
    gl.useProgram(prog.id);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    gl.viewport(0, 0, vw, vh);
    textures.forEach((t, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, t);
    });
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    const loc = gl.getAttribLocation(prog.id, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    setUniforms((name, ...v) => {
      const u = gl.getUniformLocation(prog.id, name);
      if (u === null) return;
      if (v.length === 1) gl.uniform1f(u, v[0]); else gl.uniform2f(u, v[0], v[1]);
    });
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); lost = true; api.onlost?.(); });
  canvas.addEventListener("webglcontextrestored", () => {
    // Programs, buffers and textures are gone; rebuild what we need lazily.
    try {
      progs = {
        base: program(gl, VERT, FRAG_BASE),
        blur: program(gl, VERT, FRAG_BLUR),
        bg: program(gl, VERT, FRAG_BG),
        core: program(gl, VERT, FRAG_CORE),
        final: program(gl, VERT, FRAG_FINAL),
      };
    } catch (err) { onError("webgl restore failed", err); return; }
    quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    targets.length = 0; small.length = 0; tw = th = sw = sh = 0;
    if (src) uploadSource();
    uploadMask();
    uploadBgImage();
    lost = false;
    api.onrestored?.();
  });

  const api = {
    onlost: null,
    onrestored: null,
    get lost() { return lost; },

    /** Set the photo ({source, width, height} from loadSource). */
    setImage(image) {
      if (src && src.source !== image.source) src.source.close?.();
      src = image;
      if (!lost) uploadSource();
    },

    /** Set (or clear with null) the subject matte: an ImageBitmap/canvas whose red channel is 255 on the subject. */
    setMask(mask) {
      if (maskSrc && maskSrc !== mask) maskSrc.close?.();
      maskSrc = mask;
      if (!lost) uploadMask();
    },

    get hasMask() { return !!maskSrc; },

    /** Set (or clear with null) the replacement photo ({source, width, height} from loadSource). */
    setBackgroundImage(image) {
      if (bgImage && bgImage !== image) bgImage.source.close?.();
      bgImage = image;
      if (!lost) uploadBgImage();
    },

    get hasBackgroundImage() { return !!bgImage; },

    /** Draw `params` (from liveParams) at `w`x`h` canvas pixels. */
    draw(params, w, h) {
      if (lost || !src) return false;
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      ensureTargets(w, h);
      const [A, B, C, D] = targets;
      const serverW = src.width * Math.min(1, SERVER_MAX / Math.max(src.width, src.height));
      const blurPass = (from, to, dx, dy, sigma, flip, bw, bh) => pass(progs.blur, to, [from], (u) => {
        u("uFlip", flip);
        u("uStep", dx / bw, dy / bh);
        u("uSigma", sigma);
        gl.uniform1i(gl.getUniformLocation(progs.blur.id, "uTex"), 0);
      }, bw, bh);

      // Background (needs the matte): the server blurs with sigma = strength px at its processing size.
      let input = srcTex;
      let inputFlip = 1;
      const replacing = params.bg && (params.bg.mode === "color" || params.bg.mode === "image");
      if (params.bg && maskTex && (!replacing || params.bg.mode === "color" || bgTex)) {
        const bgp = params.bg;
        let blurTex = null;
        let useBlurMask = 0;
        // Blurs run on a smaller copy when big (much cheaper, looks the same).
        const smallBlur = (from, sigma, flip) => {
          const f = Math.max(1, sigma / 5);
          const bw = Math.max(1, Math.round(w / f));
          const bh = Math.max(1, Math.round(h / f));
          ensureSmall(bw, bh);
          blurPass(from, small[0], 1, 0, sigma / f, flip, bw, bh);
          blurPass(small[0].tex, small[1], 0, 1, sigma / f, 0, bw, bh);
          return small[1].tex;
        };
        if (bgp.mode === "blur") {
          blurTex = smallBlur(srcTex, Math.max(0.3, (bgp.strength * w) / serverW), 1);
        } else if (replacing && bgp.feather > 0) {
          blurTex = smallBlur(maskTex, Math.max(0.3, (bgp.feather * 12 * w) / serverW), 1);
          useBlurMask = 1;
        }
        // Colour-fringe removal: (colour * core, core) blurred at the server's sigma (6 px at its processing size).
        let fgTex = null;
        if (replacing && bgp.defringe > 0) {
          const sigma = Math.min(8, Math.max(0.3, (6 * w) / serverW));
          pass(progs.core, B, [srcTex, maskTex], (u) => {
            u("uFlip", 1);
            gl.uniform1i(gl.getUniformLocation(progs.core.id, "uSrc"), 0);
            gl.uniform1i(gl.getUniformLocation(progs.core.id, "uMask"), 1);
          });
          blurPass(B.tex, C, 1, 0, sigma, 0, w, h);
          blurPass(C.tex, B, 0, 1, sigma, 0, w, h);
          fgTex = B.tex;
        }
        const rect = bgp.mode === "image"
          ? coverRect(bgImage.width, bgImage.height, w, h, bgp.zoom, bgp.offsetX, bgp.offsetY)
          : [0, 0, 1, 1];
        const bgw = bgImage ? bgImage.width : 1;
        const bgh = bgImage ? bgImage.height : 1;
        pass(progs.bg, D, [srcTex, blurTex || srcTex, maskTex, bgTex || srcTex, fgTex || srcTex], (u) => {
          u("uFlip", 1);
          u("uMode", { blur: 0, remove: 1, color: 2, image: 3 }[bgp.mode]);
          u("uSize", w, h);
          u("uShrink", bgp.shrink);
          u("uDefringe", fgTex ? bgp.defringe : 0);
          u("uUseBlurMask", useBlurMask);
          const loc = (n) => gl.getUniformLocation(progs.bg.id, n);
          gl.uniform3fv(loc("uColor"), hexToRgb(bgp.color));
          gl.uniform4f(loc("uRect"), rect[0] / bgw, rect[1] / bgh, rect[2] / bgw, rect[3] / bgh);
          gl.uniform1i(loc("uSrc"), 0);
          gl.uniform1i(loc("uBlur"), 1);
          gl.uniform1i(loc("uMask"), 2);
          gl.uniform1i(loc("uBgImg"), 3);
          gl.uniform1i(loc("uFg"), 4);
        });
        input = D.tex;
        inputFlip = 0;
      }

      pass(progs.base, A, [input], (u) => {
        u("uFlip", inputFlip);
        u("uSize", w, h);
        u("uGrain", params.grain);
        u("uBright", params.brightness);
        u("uContrast", params.contrast);
        u("uSat", params.saturation);
        u("uSeed", 7);
        gl.uniform1i(gl.getUniformLocation(progs.base.id, "uTex"), 0);
      });
      let blurred = A.tex;
      if (params.sharpness !== 1) {
        // The server's sigma is 2 px at the (<= 1536 px) processed size; scale it to this output size.
        const sigma = Math.max(0.3, (2 * w) / serverW);
        blurPass(A.tex, B, 1, 0, sigma, 0, w, h);
        blurPass(B.tex, C, 0, 1, sigma, 0, w, h);
        blurred = C.tex;
      }
      pass(progs.final, null, [A.tex, blurred], (u) => {
        u("uFlip", 0);
        u("uSharp", params.sharpness);
        u("uVig", params.vignette);
        u("uSepia", params.sepia ? 1 : 0);
        u("uGray", params.gray);
        gl.uniform1i(gl.getUniformLocation(progs.final.id, "uA"), 0);
        gl.uniform1i(gl.getUniformLocation(progs.final.id, "uB"), 1);
      });
      return true;
    },

    destroy() {
      src?.source?.close?.();
      src = null;
      maskSrc?.close?.();
      maskSrc = null;
      bgImage?.source?.close?.();
      bgImage = null;
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    },
  };
  return api;
}

function program(gl, vs, fs) {
  const compile = (type, text) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, text);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader compile failed");
    return s;
  };
  const id = gl.createProgram();
  gl.attachShader(id, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(id, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(id);
  if (!gl.getProgramParameter(id, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(id) || "link failed");
  return { id };
}
