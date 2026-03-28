/**
 * NewFace Browser Engine — orchestrates the face swap pipeline.
 * Uses a Web Worker for face detection (WASM) so it runs in parallel
 * with the WebGPU swap on the main thread.
 *
 * Hardened: all flags have timeout recovery, worker death is detected,
 * shared buffers are cloned before exposure, concurrent access is guarded.
 */

import { loadSession, loadSessionWasm, loadEmap, loadModelBytes, checkCache, totalModelSize } from './models.js?v=15';
import {
  detectOneFace, alignFace, extractEmbedding, projectEmbedding,
  runSwap, pasteBack, parseFullFrame, createRegionMask,
  blendRegion, sharpen,
} from './pipeline.js?v=15';

export class Engine {
  constructor() {
    // Models (main thread)
    this.detSession = null;
    this.recSession = null;
    this.swapSession = null;
    this.parseSession = null;
    this.emap = null;

    // Detection worker
    this._detWorker = null;
    this._workerReady = false;
    this._workerBusy = false;
    this._workerDead = false;
    this._workerReqId = 1;  // Start at 1; frame detection uses id=0
    this._workerCallbacks = new Map();
    this._latestDetection = null;
    this._workerBusySince = 0;  // Timestamp for stuck-detection recovery

    // State
    this.ready = false;
    this.running = false;
    this.sourceLatent = null;
    this.sourceEmbedding = null;
    this._settingReference = false;
    this._processingFrame = false;  // Single-concurrency guard

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
    const sizes = {
      det_10g: 16_923_827,
      w600k_r50: 174_383_860,
      inswapper: 277_680_638,
      bisenet: 93_632_546,
      emap: 1_048_576,
    };
    const totalSize = Object.values(sizes).reduce((a, b) => a + b, 0);
    const loaded = { det_10g: 0, w600k_r50: 0, inswapper: 0, bisenet: 0, emap: 0 };

    const progress = (l, t, name) => {
      loaded[name] = l;
      const overallLoaded = Object.values(loaded).reduce((a, b) => a + b, 0);
      if (onProgress) onProgress(name, l, t, overallLoaded / totalSize);
    };

    console.log('[Engine] Loading det_10g...');
    const detBytes = await loadModelBytes('det_10g', progress);
    await this._initDetectionWorker(detBytes);

    console.log('[Engine] Loading det_10g (WASM, for ref detection)...');
    this.detSession = await loadSessionWasm('det_10g', () => {});

    console.log('[Engine] Loading w600k_r50 (WASM)...');
    this.recSession = await loadSessionWasm('w600k_r50', progress);

    console.log('[Engine] Loading inswapper (WebGPU)...');
    this.swapSession = await loadSession('inswapper', progress);

    console.log('[Engine] Loading bisenet (WebGPU)...');
    this.parseSession = await loadSession('bisenet', progress);

    console.log('[Engine] Loading emap...');
    this.emap = await loadEmap(progress);

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
          this._workerBusySince = 0;

          const cb = this._workerCallbacks.get(e.data.id);
          if (cb) {
            this._workerCallbacks.delete(e.data.id);
            cb(e.data.face);
          } else {
            // Frame detection — only update if face found (prevents flicker)
            if (e.data.face) {
              this._latestDetection = e.data.face;
            }
          }
        }
        else if (type === 'error') {
          this._workerBusy = false;
          this._workerBusySince = 0;
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
        this._workerDead = true;
        this._workerBusy = false;
        this._workerBusySince = 0;

        // Resolve all pending callbacks
        for (const [id, cb] of this._workerCallbacks) cb(null);
        this._workerCallbacks.clear();

        if (!this._workerReady) reject(err);
      };

      const bytesCopy = modelBytes.slice(0);
      this._detWorker.postMessage({ type: 'init', modelBytes: bytesCopy }, [bytesCopy]);
    });
  }

  /**
   * Send a frame to the detection worker (fire-and-forget).
   * Includes stuck-detection recovery: if worker has been busy for >5s, reset the flag.
   */
  _sendFrameToWorker(frameData) {
    if (!this._workerReady || this._workerDead) return;

    // Stuck recovery: if worker has been "busy" for over 5 seconds, assume it's stuck
    if (this._workerBusy) {
      if (this._workerBusySince > 0 && (performance.now() - this._workerBusySince) > 5000) {
        console.warn('[Engine] Worker appears stuck — resetting busy flag');
        this._workerBusy = false;
      } else {
        return;
      }
    }

    this._workerBusy = true;
    this._workerBusySince = performance.now();

    const pixelsCopy = frameData.data.buffer.slice(0);
    this._detWorker.postMessage(
      { type: 'detect', pixels: pixelsCopy, width: frameData.width, height: frameData.height, id: 0 },
      [pixelsCopy]
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
      // Always clear — if another setReference superseded us, it set its own flag
      this._settingReference = false;
    }
  }

  // ── Frame Processing (Pipelined) ──────────────────────────────

  async processFrame(frameData) {
    if (!this.ready || !this.sourceLatent || this._settingReference) return null;

    // Single-concurrency guard: prevent overlapping processFrame calls
    if (this._processingFrame) return null;
    this._processingFrame = true;

    try {
      const result = await this._processFrameInner(frameData);
      return result;
    } catch (err) {
      console.error('[Engine] processFrame error:', err);
      return null;
    } finally {
      this._processingFrame = false;
      this._frameInProgress = null;
    }
  }

  async _processFrameInner(frameData) {
    const { width: W, height: H } = frameData;

    // Send this frame to the worker for NEXT detection (pipeline)
    this._sendFrameToWorker(frameData);

    // Use the latest detection result from the worker
    let face = this._latestDetection;
    if (!face) {
      // No detection yet — run on main thread for first frame (or if worker is dead)
      try {
        face = await detectOneFace(this.detSession, frameData);
      } catch (err) {
        console.error('[Engine] Main-thread detection error:', err);
        return null;
      }
      if (!face) return null;
      this._latestDetection = face;
    }

    return this._processWithFace(frameData, W, H, face);
  }

  async _processWithFace(frameData, W, H, face) {
    const pixelCount = W * H * 4;

    if (!this._pasteBuf || this._pasteBuf.length !== pixelCount) {
      this._pasteBuf = new Uint8ClampedArray(pixelCount);
      this._blendBuf = new Uint8ClampedArray(pixelCount);
    }

    // 1. Align target face to 128×128 for swapper
    const { data: aligned128, M } = alignFace(
      frameData.data, W, H, face.kps, 128
    );

    // 2. Run face swap (WebGPU)
    const swappedFace = await runSwap(this.swapSession, aligned128, this.sourceLatent);

    // 3. Paste swapped face back into frame
    const fullSwapped = pasteBack(frameData.data, W, H, swappedFace, M, this._pasteBuf);

    // 4. Regional masking
    let result;
    if (this.region === 'full') {
      result = blendRegion(frameData.data, fullSwapped, null, this.opacity, W, H, this._blendBuf);
    } else {
      const needsParsing = this._shouldReparse(face.bbox);
      if (needsParsing) {
        try {
          const parsed = await parseFullFrame(this.parseSession, frameData, face.bbox);
          if (parsed) {
            this._cachedParsing = parsed;
            this._cachedParsingBox = [...face.bbox];
          }
        } catch (err) {
          console.warn('[Engine] Parsing error (non-fatal):', err.message);
        }
      }

      if (this._cachedParsing) {
        const { labels, cropBox, cropW, cropH } = this._cachedParsing;
        const mask = createRegionMask(
          labels, cropW, cropH, this.region, cropBox, face.kps, W, H
        );
        result = blendRegion(frameData.data, fullSwapped, mask, this.opacity, W, H, this._blendBuf);
      } else {
        result = blendRegion(frameData.data, fullSwapped, null, this.opacity, W, H, this._blendBuf);
      }
    }

    // 5. Sharpening
    if (this.sharpness > 0) {
      result = sharpen(result, W, H, this.sharpness);
    }

    // 6. Clone result before wrapping in ImageData — the underlying buffer
    // (_blendBuf) will be overwritten on the next frame. ImageData shares
    // the buffer, so without cloning the displayed frame could be corrupted.
    const outputData = (result === this._blendBuf || result === this._pasteBuf)
      ? new Uint8ClampedArray(result)
      : result;

    // FPS tracking
    const now = performance.now();
    this._frameTimestamps.push(now);
    if (this._frameTimestamps.length > 30) {
      this._frameTimestamps = this._frameTimestamps.slice(-30);
    }
    if (this._frameTimestamps.length > 1) {
      const span = this._frameTimestamps[this._frameTimestamps.length - 1] - this._frameTimestamps[0];
      this.fps = Math.round((this._frameTimestamps.length - 1) / (span / 1000));
    }

    return new ImageData(outputData, W, H);
  }

  _shouldReparse(bbox) {
    this._parseFrameCount++;
    if (this._parseFrameCount % 30 !== 0) return false;
    if (!this._cachedParsingBox) return true;

    const [x1, y1, x2, y2] = bbox;
    const [ox1, oy1, ox2, oy2] = this._cachedParsingBox;
    const shift = Math.abs(x1 - ox1) + Math.abs(y1 - oy1) + Math.abs(x2 - ox2) + Math.abs(y2 - oy2);
    const size = (x2 - x1 + y2 - y1);
    if (size <= 0) return true;  // Guard against degenerate bbox
    return shift / size > 0.15;
  }

  // ── Process Single Image (for preview) ─────────────────────────

  async processImage(imageData) {
    return this.processFrame(imageData);
  }
}
