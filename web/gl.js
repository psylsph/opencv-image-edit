// On-device live preview (WebGL2).
//
// Mirrors the server's pipeline maths (app/pipeline/grain.py + filters.py) in the same order:
//   grain -> brightness -> contrast -> saturation -> sharpness -> vignette -> sepia -> grayscale
// The server stays the reference implementation; this is for instant feedback while dragging sliders.
// Known, accepted differences: the noise generator is a hash (not numpy's RNG), the blur edge handling
// is clamp (OpenCV reflects), and the preview runs at screen resolution rather than the processed size.
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

// Separable gaussian (one direction per pass).
const FRAG_BLUR = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uStep;   // one texel along the blur direction
uniform float uSigma;
in vec2 uv;
out vec4 o;
void main() {
  vec3 sum = texture(uTex, uv).rgb;
  float wsum = 1.0;
  for (int i = 1; i <= 24; i++) {
    float fi = float(i);
    if (fi > uSigma * 3.0) break;
    float w = exp(-(fi * fi) / (2.0 * uSigma * uSigma));
    sum += w * (texture(uTex, uv + uStep * fi).rgb + texture(uTex, uv - uStep * fi).rgb);
    wsum += 2.0 * w;
  }
  o = vec4(sum / wsum, 1.0);
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
  return {
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
  return !p.grain && p.brightness === 1 && p.contrast === 1 && p.saturation === 1 &&
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
  const targets = []; // three ping-pong render targets at output size
  let tw = 0, th = 0;

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
    for (let i = 0; i < 3; i++) targets.push(makeTarget(w, h));
    tw = w; th = h;
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

  function pass(prog, target, textures, setUniforms) {
    gl.useProgram(prog.id);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    gl.viewport(0, 0, tw, th);
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
        final: program(gl, VERT, FRAG_FINAL),
      };
    } catch (err) { onError("webgl restore failed", err); return; }
    quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    targets.length = 0; tw = th = 0;
    if (src) uploadSource();
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

    /** Draw `params` (from liveParams) at `w`x`h` canvas pixels. */
    draw(params, w, h) {
      if (lost || !src) return false;
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      ensureTargets(w, h);
      const [A, B, C] = targets;
      pass(progs.base, A, [srcTex], (u) => {
        u("uFlip", 1);
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
        const serverW = src.width * Math.min(1, SERVER_MAX / Math.max(src.width, src.height));
        const sigma = Math.max(0.3, (2 * w) / serverW);
        const blurPass = (from, to, dx, dy) => pass(progs.blur, to, [from], (u) => {
          u("uFlip", 0);
          u("uStep", dx / w, dy / h);
          u("uSigma", sigma);
          gl.uniform1i(gl.getUniformLocation(progs.blur.id, "uTex"), 0);
        });
        blurPass(A.tex, B, 1, 0);
        blurPass(B.tex, C, 0, 1);
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
