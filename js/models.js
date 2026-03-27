/**
 * Model loading + IndexedDB caching for NewFace browser engine
 */

const DB_NAME = 'newface-models';
const DB_VERSION = 1;
const STORE_NAME = 'blobs';

// Model registry — URLs will point to GitHub Releases on alphastack1/storage
const MODEL_REGISTRY = {
  det_10g:    { file: 'det_10g.onnx',              size: 16_923_827 },
  w600k_r50:  { file: 'w600k_r50.onnx',            size: 174_383_860 },
  inswapper:  { file: 'inswapper_128_fp16.onnx',    size: 277_680_638 },
  bisenet:    { file: 'bisenet_resnet_34.onnx',     size: 93_632_546 },
  emap:       { file: 'emap.bin',                   size: 1_048_576 },
};

// Use Netlify proxy on production (avoids CORS with GitHub Releases)
// Use local /models/ path for localhost development
let _baseUrl = (location.hostname === 'localhost' || location.hostname === '127.0.0.1')
  ? '/models/'
  : '/models-cdn/';

export function setModelBaseUrl(url) {
  _baseUrl = url.endsWith('/') ? url : url + '/';
}

// ── IndexedDB helpers ──────────────────────────────────────────────

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function cacheGet(key) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => reject(req.error);
  });
}

async function cachePut(key, value) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const req = store.put(value, key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

// ── Download with progress ─────────────────────────────────────────

/**
 * Download a file as ArrayBuffer, reporting progress.
 * @param {string} url
 * @param {function} onProgress - (loaded, total) => void
 * @returns {Promise<ArrayBuffer>}
 */
async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);

  const contentLength = +resp.headers.get('Content-Length') || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (onProgress) onProgress(loaded, contentLength || loaded);
  }

  const buf = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.length;
  }
  return buf.buffer;
}

// ── Public API ─────────────────────────────────────────────────────

/**
 * Load a model's raw bytes (from cache or network).
 * @param {string} name - Key in MODEL_REGISTRY
 * @param {function} onProgress - (loaded, total, name) => void
 * @returns {Promise<ArrayBuffer>}
 */
export async function loadModelBytes(name, onProgress) {
  // Check cache first
  const cached = await cacheGet(name);
  if (cached) {
    if (onProgress) onProgress(cached.byteLength, cached.byteLength, name);
    return cached;
  }

  // Download
  const info = MODEL_REGISTRY[name];
  if (!info) throw new Error(`Unknown model: ${name}`);
  const url = _baseUrl + info.file;

  const buf = await fetchWithProgress(url, (loaded, total) => {
    if (onProgress) onProgress(loaded, total || info.size, name);
  });

  // Cache for next time
  await cachePut(name, buf);
  return buf;
}

/**
 * Create an ort.InferenceSession for a model.
 * @param {string} name - Key in MODEL_REGISTRY
 * @param {function} onProgress
 * @returns {Promise<ort.InferenceSession>}
 */
export async function loadSession(name, onProgress) {
  const buf = await loadModelBytes(name, onProgress);

  console.log(`[Models] ${name}: navigator.gpu = ${!!navigator.gpu}`);
  if (!navigator.gpu) {
    throw new Error(`WebGPU not available. Please use Chrome 113+ with WebGPU enabled.`);
  }

  // Try WebGPU first — fall back to WASM only if WebGPU can't handle this model
  try {
    console.log(`[Models] ${name}: creating WebGPU session...`);
    const session = await ort.InferenceSession.create(buf.slice(0), {
      executionProviders: ['webgpu'],
    });
    console.log(`[Models] ✅ ${name} loaded (WebGPU)`);
    return session;
  } catch (e) {
    console.warn(`[Models] ⚠️ ${name} WebGPU failed (error: ${e?.message || e}), falling back to WASM`);
  }

  const session = await ort.InferenceSession.create(buf, {
    executionProviders: ['wasm'],
  });
  console.log(`[Models] ${name} loaded (WASM fallback)`);
  return session;
}

/**
 * Force-create a WASM session for a model (used as runtime fallback when WebGPU inference fails).
 */
export async function loadSessionWasm(name) {
  const buf = await loadModelBytes(name, () => {});
  const session = await ort.InferenceSession.create(buf, {
    executionProviders: ['wasm'],
  });
  console.log(`[Models] ${name} reloaded (WASM fallback)`);
  return session;
}

/**
 * Load the emap matrix (512x512 float32).
 * @param {function} onProgress
 * @returns {Promise<Float32Array>}
 */
export async function loadEmap(onProgress) {
  const buf = await loadModelBytes('emap', onProgress);
  return new Float32Array(buf);
}

/**
 * Total download size of all models in bytes.
 */
export function totalModelSize() {
  return Object.values(MODEL_REGISTRY).reduce((s, m) => s + m.size, 0);
}

/**
 * Check which models are already cached.
 * @returns {Promise<{cached: string[], missing: string[]}>}
 */
export async function checkCache() {
  const cached = [];
  const missing = [];
  for (const name of Object.keys(MODEL_REGISTRY)) {
    const data = await cacheGet(name);
    (data ? cached : missing).push(name);
  }
  return { cached, missing };
}
