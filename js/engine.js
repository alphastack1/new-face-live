/**
 * NewFace Browser Engine — orchestrates the face swap pipeline.
 * Manages model loading, camera, frame processing loop.
 */

import { loadSession, loadSessionWasm, loadSessionPreferGPU, loadEmap, loadModelBytes, checkCache, totalModelSize } from './models.js?v=12';
import {
  detectOneFace, alignFace, extractEmbedding, projectEmbedding,
  runSwap, pasteBack, parseFullFrame, createRegionMask,
  blendRegion, sharpen,
} from './pipeline.js?v=12';

export class Engine {
  constructor() {
    // Models
    this.detSession = null;
    this.recSession = null;
    this.swapSession = null;
    this.parseSession = null;
    this.emap = null;

    // State
    this.ready = false;
    this.running = false;
    this.sourceLatent = null;     // Projected embedding of reference face
    this.sourceEmbedding = null;  // Raw 512-dim embedding

    // Settings
    this.region = 'nose';
    this.opacity = 0.7;
    this.sharpness = 0;
    this.mirror = true;

    // Cached parsing (reuse across frames if face hasn't moved much)
    this._cachedParsing = null;
    this._cachedParsingBox = null;
    this._parseFrameCount = 0;

    // Cached detection (skip detect on intermediate frames)
    this._cachedFace = null;
    this._detectFrameCount = 0;
    this._detectEveryN = 3;  // Run detection every Nth frame, reuse kps otherwise

    // Performance
    this.fps = 0;
    this._frameTimestamps = [];
  }

  // ── Model Loading ──────────────────────────────────────────────

  /**
   * Load all models with progress reporting.
   * @param {function} onProgress - (modelName, loaded, total, overallPct) => void
   */
  async init(onProgress) {
    const models = ['det_10g', 'w600k_r50', 'inswapper', 'bisenet', 'emap'];
    const sizes = {
      det_10g: 16_923_827,
      w600k_r50: 174_383_860,
      inswapper: 277_680_638,
      bisenet: 93_632_546,
      emap: 1_048_576,
    };
    const totalSize = Object.values(sizes).reduce((a, b) => a + b, 0);
    const loaded = {};
    models.forEach(m => loaded[m] = 0);

    const progress = (l, t, name) => {
      loaded[name] = l;
      const overallLoaded = Object.values(loaded).reduce((a, b) => a + b, 0);
      if (onProgress) onProgress(name, l, t, overallLoaded / totalSize);
    };

    // Load models (sequentially to avoid memory pressure)
    // det_10g: WASM — WebGPU AveragePool shape computation uses ceil() internally (ORT bug)
    // w600k_r50: WASM — has ops WebGPU can't create session for (runs once, no perf impact)
    // inswapper: WebGPU REQUIRED — runs every frame, 20s on WASM vs <100ms on GPU
    // bisenet: WebGPU — runs every 15 frames for face parsing
    console.log('[Engine] Loading det_10g (WASM)...');
    this.detSession = await loadSessionWasm('det_10g', progress);

    console.log('[Engine] Loading w600k_r50 (WASM)...');
    this.recSession = await loadSessionWasm('w600k_r50', progress);

    console.log('[Engine] Loading inswapper (WebGPU)...');
    this.swapSession = await loadSession('inswapper', progress);

    console.log('[Engine] Loading bisenet (WebGPU)...');
    this.parseSession = await loadSession('bisenet', progress);

    console.log('[Engine] Loading emap...');
    this.emap = await loadEmap(progress);

    // Pre-warm sessions with dummy inference to trigger shader compilation
    console.log('[Engine] Pre-warming sessions...');
    await this._warmup();

    this.ready = true;
    console.log('[Engine] All models loaded. Ready.');
  }

  async _warmup() {
    try {
      // Warm WebGPU sessions to trigger shader compilation
      // (WASM sessions don't need warmup)
      const swapImg = new ort.Tensor('float32', new Float32Array(1 * 3 * 128 * 128), [1, 3, 128, 128]);
      const swapSrc = new ort.Tensor('float32', new Float32Array(512), [1, 512]);
      await this.swapSession.run({ 'target': swapImg, 'source': swapSrc });
      console.log('[Engine] Warmup complete');
    } catch (e) {
      console.warn('[Engine] Warmup error (non-fatal):', e.message);
    }
  }

  // ── Reference Face ─────────────────────────────────────────────

  /**
   * Set reference face from an image element or URL.
   * Detects the face, extracts embedding, and projects through emap.
   *
   * @param {HTMLImageElement|string} source - Image element or URL
   * @returns {Promise<boolean>} True if reference was set successfully
   */
  async setReference(source) {
    if (!this.ready) return false;

    // Cancel any in-progress reference setting
    this._refVersion = (this._refVersion || 0) + 1;
    const myVersion = this._refVersion;

    // Signal processFrame to stop and wait for current frame to finish
    this._settingReference = true;
    if (this._frameInProgress) {
      await this._frameInProgress;
    }

    try {
      // Get image data
      let img = source;
      if (typeof source === 'string') {
        img = new Image();
        img.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          img.src = source;
        });
      }

      // Bail if another setReference was called while we loaded the image
      if (this._refVersion !== myVersion) return false;

      const canvas = new OffscreenCanvas(img.naturalWidth || img.width, img.naturalHeight || img.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Detect face
      const face = await detectOneFace(this.detSession, imgData);
      if (!face) {
        console.warn('[Engine] No face detected in reference image');
        return false;
      }
      if (this._refVersion !== myVersion) return false;

      // Align to 112×112 for embedding
      const { data: alignedRGBA } = alignFace(
        imgData.data, imgData.width, imgData.height, face.kps, 112
      );

      // Extract embedding
      this.sourceEmbedding = await extractEmbedding(this.recSession, alignedRGBA);
      if (this._refVersion !== myVersion) return false;

      // Project through emap
      this.sourceLatent = projectEmbedding(this.sourceEmbedding, this.emap);

      // Clear cached face so detection runs fresh with new reference
      this._cachedFace = null;
      this._cachedParsing = null;
      this._detectFrameCount = 0;

      console.log('[Engine] Reference face set');
      return true;
    } finally {
      // Only clear the flag if we're still the active setReference call
      if (this._refVersion === myVersion) {
        this._settingReference = false;
      }
    }
  }

  // ── Frame Processing ───────────────────────────────────────────

  /**
   * Process a single video frame: detect, swap, parse, blend.
   *
   * @param {ImageData} frameData - Camera frame (RGBA)
   * @returns {Promise<ImageData|null>} Processed frame, or null if no face / no ref
   */
  async processFrame(frameData) {
    if (!this.ready || !this.sourceLatent || this._settingReference) return null;

    const promise = this._processFrameInner(frameData);
    this._frameInProgress = promise;
    try {
      return await promise;
    } finally {
      if (this._frameInProgress === promise) this._frameInProgress = null;
    }
  }

  async _processFrameInner(frameData) {
    const { width: W, height: H } = frameData;

    // 1. Detect face — skip on intermediate frames and reuse cached face
    this._detectFrameCount++;
    let face;
    if (this._cachedFace && (this._detectFrameCount % this._detectEveryN !== 0)) {
      face = this._cachedFace;
    } else {
      face = await detectOneFace(this.detSession, frameData);
      if (!face) {
        this._cachedFace = null;
        return null;
      }
      this._cachedFace = face;
    }

    // 2. Align target face to 128×128 for swapper
    const { data: aligned128, M } = alignFace(
      frameData.data, W, H, face.kps, 128
    );

    // 3. Run face swap
    const swappedFace = await runSwap(this.swapSession, aligned128, this.sourceLatent);

    // 4. Paste swapped face back into frame
    const fullSwapped = pasteBack(frameData.data, W, H, swappedFace, M);

    // 5. Regional masking (if not full-face swap)
    let result;
    if (this.region === 'full') {
      result = blendRegion(frameData.data, fullSwapped, null, this.opacity, W, H);
    } else {
      // Run parsing infrequently (every 15 frames or when bbox moves)
      const needsParsing = this._shouldReparse(face.bbox);
      if (needsParsing) {
        const parsed = await parseFullFrame(this.parseSession, frameData, face.bbox);
        this._cachedParsing = parsed;
        this._cachedParsingBox = [...face.bbox];
      }

      if (this._cachedParsing) {
        const { labels, cropBox, cropW, cropH } = this._cachedParsing;
        const mask = createRegionMask(
          labels, cropW, cropH, this.region, cropBox, face.kps, W, H
        );
        result = blendRegion(frameData.data, fullSwapped, mask, this.opacity, W, H);
      } else {
        result = blendRegion(frameData.data, fullSwapped, null, this.opacity, W, H);
      }
    }

    // 6. Sharpening
    if (this.sharpness > 0) {
      result = sharpen(result, W, H, this.sharpness);
    }

    // FPS tracking
    this._frameTimestamps.push(performance.now());
    while (this._frameTimestamps.length > 30) this._frameTimestamps.shift();
    if (this._frameTimestamps.length > 1) {
      const span = this._frameTimestamps[this._frameTimestamps.length - 1] - this._frameTimestamps[0];
      this.fps = Math.round((this._frameTimestamps.length - 1) / (span / 1000));
    }

    return new ImageData(result, W, H);
  }

  /**
   * Determine if we need to re-run face parsing.
   */
  _shouldReparse(bbox) {
    this._parseFrameCount++;

    // Parse every 15 frames (bisenet is expensive in WASM)
    if (this._parseFrameCount % 15 !== 0) return false;
    if (!this._cachedParsingBox) return true;

    // Reparse if bbox moved significantly
    const [x1, y1, x2, y2] = bbox;
    const [ox1, oy1, ox2, oy2] = this._cachedParsingBox;
    const shift = Math.abs(x1 - ox1) + Math.abs(y1 - oy1) + Math.abs(x2 - ox2) + Math.abs(y2 - oy2);
    const size = (x2 - x1 + y2 - y1);
    return shift / size > 0.15;
  }

  // ── Process Single Image (for preview) ─────────────────────────

  /**
   * Process a single still image (photo mode).
   */
  async processImage(imageData) {
    return this.processFrame(imageData);
  }
}
