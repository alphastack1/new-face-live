/**
 * NewFace Browser Engine — orchestrates the face swap pipeline.
 * Uses a Web Worker for face detection (WASM) so it runs in parallel
 * with the WebGPU swap on the main thread.
 */

import { loadSession, loadSessionWasm, loadEmap, loadModelBytes, checkCache, totalModelSize } from './models.js?v=13';
import {
  detectOneFace, alignFace, extractEmbedding, projectEmbedding,
  runSwap, pasteBack, parseFullFrame, createRegionMask,
  blendRegion, sharpen,
} from './pipeline.js?v=13';

export class Engine {
  constructor() {
    // Models (main thread)
    this.detSession = null;   // WASM — only used for setReference
    this.recSession = null;
    this.swapSession = null;
    this.parseSession = null;
    this.emap = null;

    // Detection worker
    this._detWorker = null;
    this._workerReady = false;
    this._workerBusy = false;
    this._workerReqId = 0;
    this._workerCallbacks = new Map();  // id → resolve for ref detection
    this._latestDetection = null;       // Latest face from worker

    // State
    this.ready = false;
    this.running = false;
    this.sourceLatent = null;
    this.sourceEmbedding = null;

    // Settings
    this.region = 'nose';
    this.opacity = 0.7;
    this.sharpness = 0;
    this.mirror = true;

    // Cached parsing
    this._cachedParsing = null;
    this._cachedParsingBox = null;
    this._parseFrameCount = 0;

    // Performance
    this.fps = 0;
    this._frameTimestamps = [];
  }

  // ── Model Loading ──────────────────────────────────────────────

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

    // 1. Load det_10g model bytes and send to worker
    console.log('[Engine] Loading det_10g...');
    const detBytes = await loadModelBytes('det_10g', progress);
    await this._initDetectionWorker(detBytes);

    // Also load a main-thread WASM session for setReference (infrequent)
    console.log('[Engine] Loading det_10g (WASM, for ref detection)...');
    this.detSession = await loadSessionWasm('det_10g', (l, t, name) => {
      // Don't double-count progress — det_10g already reported above
    });

    // 2. Load remaining models
    console.log('[Engine] Loading w600k_r50 (WASM)...');
    this.recSession = await loadSessionWasm('w600k_r50', progress);

    console.log('[Engine] Loading inswapper (WebGPU)...');
    this.swapSession = await loadSession('inswapper', progress);

    console.log('[Engine] Loading bisenet (WebGPU)...');
    this.parseSession = await loadSession('bisenet', progress);

    console.log('[Engine] Loading emap...');
    this.emap = await loadEmap(progress);

    // Pre-warm WebGPU sessions
    console.log('[Engine] Pre-warming sessions...');
    await this._warmup();

    this.ready = true;
    console.log('[Engine] All models loaded. Ready.');
  }

  async _initDetectionWorker(modelBytes) {
    return new Promise((resolve, reject) => {
      this._detWorker = new Worker('./js/detection-worker.js');

      this._detWorker.onmessage = (e) => {
        const { type } = e.data;

        if (type === 'ready') {
          this._workerReady = true;
          console.log('[Engine] Detection worker ready');
          resolve();
        }
        else if (type === 'result') {
          this._workerBusy = false;
          // If there's a callback (ref detection), resolve it
          const cb = this._workerCallbacks.get(e.data.id);
          if (cb) {
            this._workerCallbacks.delete(e.data.id);
            cb(e.data.face);
          } else {
            // Frame detection — store as latest result
            this._latestDetection = e.data.face;
          }
        }
        else if (type === 'error') {
          this._workerBusy = false;
          console.error('[Engine] Worker error:', e.data.message);
          const cb = this._workerCallbacks.get(e.data.id);
          if (cb) {
            this._workerCallbacks.delete(e.data.id);
            cb(null);
          }
          if (!this._workerReady) reject(new Error(e.data.message));
        }
      };

      this._detWorker.onerror = (err) => {
        console.error('[Engine] Worker fatal error:', err);
        if (!this._workerReady) reject(err);
      };

      // Send model bytes to worker (transfer, zero-copy)
      // Clone first since we also need bytes for main thread session
      const bytesCopy = modelBytes.slice(0);
      this._detWorker.postMessage({ type: 'init', modelBytes: bytesCopy }, [bytesCopy]);
    });
  }

  /**
   * Send a frame to the detection worker (fire-and-forget).
   * The result will appear in this._latestDetection when ready.
   */
  _sendFrameToWorker(frameData) {
    if (!this._workerReady || this._workerBusy) return;
    this._workerBusy = true;

    // Copy pixel data for transfer to worker
    const pixelsCopy = frameData.data.buffer.slice(0);
    this._detWorker.postMessage(
      { type: 'detect', pixels: pixelsCopy, width: frameData.width, height: frameData.height, id: 0 },
      [pixelsCopy]  // Transfer (zero-copy send)
    );
  }

  async _warmup() {
    try {
      const swapImg = new ort.Tensor('float32', new Float32Array(1 * 3 * 128 * 128), [1, 3, 128, 128]);
      const swapSrc = new ort.Tensor('float32', new Float32Array(512), [1, 512]);
      await this.swapSession.run({ 'target': swapImg, 'source': swapSrc });
      console.log('[Engine] Warmup complete');
    } catch (e) {
      console.warn('[Engine] Warmup error (non-fatal):', e.message);
    }
  }

  // ── Reference Face ─────────────────────────────────────────────

  async setReference(source) {
    if (!this.ready) return false;

    this._refVersion = (this._refVersion || 0) + 1;
    const myVersion = this._refVersion;

    this._settingReference = true;
    if (this._frameInProgress) {
      await this._frameInProgress;
    }

    try {
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

      if (this._refVersion !== myVersion) return false;

      const canvas = new OffscreenCanvas(img.naturalWidth || img.width, img.naturalHeight || img.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Detect face using main-thread session (not the worker)
      const face = await detectOneFace(this.detSession, imgData);
      if (!face) {
        console.warn('[Engine] No face detected in reference image');
        return false;
      }
      if (this._refVersion !== myVersion) return false;

      const { data: alignedRGBA } = alignFace(
        imgData.data, imgData.width, imgData.height, face.kps, 112
      );

      this.sourceEmbedding = await extractEmbedding(this.recSession, alignedRGBA);
      if (this._refVersion !== myVersion) return false;

      this.sourceLatent = projectEmbedding(this.sourceEmbedding, this.emap);

      // Clear caches
      this._latestDetection = null;
      this._cachedParsing = null;
      this._parseFrameCount = 0;

      console.log('[Engine] Reference face set');
      return true;
    } finally {
      if (this._refVersion === myVersion) {
        this._settingReference = false;
      }
    }
  }

  // ── Frame Processing (Pipelined) ──────────────────────────────

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

    // Send this frame to the worker for NEXT detection (pipeline)
    this._sendFrameToWorker(frameData);

    // Use the latest detection result from the worker
    const face = this._latestDetection;
    if (!face) {
      // No detection yet — run synchronously on main thread for first frame
      const firstFace = await detectOneFace(this.detSession, frameData);
      if (!firstFace) return null;
      this._latestDetection = firstFace;
      return this._processWithFace(frameData, W, H, firstFace);
    }

    return this._processWithFace(frameData, W, H, face);
  }

  async _processWithFace(frameData, W, H, face) {
    // 1. Align target face to 128×128 for swapper
    const { data: aligned128, M } = alignFace(
      frameData.data, W, H, face.kps, 128
    );

    // 2. Run face swap (WebGPU — this runs in parallel with worker detection)
    const swappedFace = await runSwap(this.swapSession, aligned128, this.sourceLatent);

    // 3. Paste swapped face back into frame
    const fullSwapped = pasteBack(frameData.data, W, H, swappedFace, M);

    // 4. Regional masking
    let result;
    if (this.region === 'full') {
      result = blendRegion(frameData.data, fullSwapped, null, this.opacity, W, H);
    } else {
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

    // 5. Sharpening
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

  _shouldReparse(bbox) {
    this._parseFrameCount++;
    if (this._parseFrameCount % 15 !== 0) return false;
    if (!this._cachedParsingBox) return true;

    const [x1, y1, x2, y2] = bbox;
    const [ox1, oy1, ox2, oy2] = this._cachedParsingBox;
    const shift = Math.abs(x1 - ox1) + Math.abs(y1 - oy1) + Math.abs(x2 - ox2) + Math.abs(y2 - oy2);
    const size = (x2 - x1 + y2 - y1);
    return shift / size > 0.15;
  }

  // ── Process Single Image (for preview) ─────────────────────────

  async processImage(imageData) {
    return this.processFrame(imageData);
  }
}
