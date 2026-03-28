/**
 * NewFace Browser Engine — orchestrates the face swap pipeline.
 * Manages model loading, camera, frame processing loop.
 */

import { loadSession, loadSessionWasm, loadEmap, loadModelBytes, checkCache, totalModelSize } from './models.js?v=7';
import {
  detectOneFace, alignFace, extractEmbedding, projectEmbedding,
  runSwap, pasteBack, parseFullFrame, createRegionMask,
  blendRegion, sharpen,
} from './pipeline.js?v=7';

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
    // det_10g: WASM — WebGPU lacks AveragePool ceil_mode support
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

    this.ready = true;
    console.log('[Engine] All models loaded. Ready.');
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

    // Align to 112×112 for embedding
    const { data: alignedRGBA } = alignFace(
      imgData.data, imgData.width, imgData.height, face.kps, 112
    );

    // Extract embedding
    this.sourceEmbedding = await extractEmbedding(this.recSession, alignedRGBA);

    // Project through emap
    this.sourceLatent = projectEmbedding(this.sourceEmbedding, this.emap);

    console.log('[Engine] Reference face set');
    return true;
  }

  // ── Frame Processing ───────────────────────────────────────────

  /**
   * Process a single video frame: detect, swap, parse, blend.
   *
   * @param {ImageData} frameData - Camera frame (RGBA)
   * @returns {Promise<ImageData|null>} Processed frame, or null if no face / no ref
   */
  async processFrame(frameData) {
    if (!this.ready || !this.sourceLatent) return null;

    const t0 = performance.now();
    const { width: W, height: H } = frameData;

    // Yield helper — lets RAF callbacks fire so the video stays live
    const yieldToUI = () => new Promise(r => setTimeout(r, 0));

    // 1. Detect face in frame
    const face = await detectOneFace(this.detSession, frameData);
    if (!face) return null;
    console.log(`[Frame] detect ${Math.round(performance.now() - t0)}ms`);
    await yieldToUI();

    // 2. Align target face to 128×128 for swapper
    const { data: aligned128, M } = alignFace(
      frameData.data, W, H, face.kps, 128
    );

    // 3. Run face swap
    const swappedFace = await runSwap(this.swapSession, aligned128, this.sourceLatent);
    console.log(`[Frame] swap ${Math.round(performance.now() - t0)}ms`);

    // One-time WASM comparison test
    if (!this._wasmTestDone) {
      this._wasmTestDone = true;
      console.log('[DEBUG] Running WASM comparison test...');
      try {
        const wasmSession = await ort.InferenceSession.create(
          await (await fetch('/models-cdn/inswapper_128.onnx')).arrayBuffer(),
          { executionProviders: ['wasm'] }
        );
        const wasmResult = await runSwap(wasmSession, aligned128, this.sourceLatent);
        // Compare first 20 pixels
        let gpuSample = [], wasmSample = [];
        for (let i = 0; i < 80; i += 4) {
          gpuSample.push(swappedFace[i]);
          wasmSample.push(wasmResult[i]);
        }
        console.log('[DEBUG] WebGPU output (R of first 20px):', gpuSample);
        console.log('[DEBUG] WASM output (R of first 20px):', wasmSample);
        // Overall diff
        let maxDiff = 0, totalDiff = 0;
        for (let i = 0; i < swappedFace.length; i++) {
          const d = Math.abs(swappedFace[i] - wasmResult[i]);
          if (d > maxDiff) maxDiff = d;
          totalDiff += d;
        }
        console.log(`[DEBUG] Max pixel diff: ${maxDiff}, avg diff: ${(totalDiff / swappedFace.length).toFixed(3)}`);
      } catch (e) {
        console.warn('[DEBUG] WASM comparison failed:', e.message);
      }
    }

    await yieldToUI();

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
        await yieldToUI();
        const parsed = await parseFullFrame(this.parseSession, frameData, face.bbox);
        this._cachedParsing = parsed;
        this._cachedParsingBox = [...face.bbox];
        console.log(`[Frame] parse ${Math.round(performance.now() - t0)}ms`);
        await yieldToUI();
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

    console.log(`[Frame] total ${Math.round(performance.now() - t0)}ms`);
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
