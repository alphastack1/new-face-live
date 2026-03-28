/**
 * Web Worker for face detection — runs SCRFD (det_10g) on WASM
 * in a separate thread so it doesn't block the WebGPU swap pipeline.
 *
 * Messages IN:
 *   { type: 'init', modelBytes: ArrayBuffer }
 *   { type: 'detect', pixels: Uint8ClampedArray, width, height, id }
 *
 * Messages OUT:
 *   { type: 'ready' }
 *   { type: 'result', face: {bbox, kps, score}|null, id }
 *   { type: 'error', message, id? }
 */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js');

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
ort.env.wasm.numThreads = 1;  // Worker is already a separate thread
ort.env.logLevel = 'error';

// ── Detection Constants ──────────────────────────────────────────

const DET_INPUT_SIZE = 192;
const DET_STRIDES = [8, 16, 32];
const DET_NUM_ANCHORS = 2;
const DET_SCORE_THRESH = 0.3;
const DET_NMS_THRESH = 0.4;

// ── State ────────────────────────────────────────────────────────

let session = null;
let srcCanvas = null, srcCtx = null;
let resCanvas = null, resCtx = null;

// ── Preprocessing ────────────────────────────────────────────────

function preprocessDetect(pixels, width, height) {
  const W = width, H = height;
  const size = DET_INPUT_SIZE;

  const ratio = Math.min(size / W, size / H);
  const newW = Math.round(W * ratio);
  const newH = Math.round(H * ratio);

  // Resize using cached OffscreenCanvases
  if (!srcCanvas || srcCanvas.width !== W || srcCanvas.height !== H) {
    srcCanvas = new OffscreenCanvas(W, H);
    srcCtx = srcCanvas.getContext('2d');
  }
  const imgData = new ImageData(pixels, W, H);
  srcCtx.putImageData(imgData, 0, 0);

  if (!resCanvas) {
    resCanvas = new OffscreenCanvas(size, size);
    resCtx = resCanvas.getContext('2d');
  }
  resCtx.clearRect(0, 0, size, size);
  resCtx.drawImage(srcCanvas, 0, 0, W, H, 0, 0, newW, newH);

  const resized = resCtx.getImageData(0, 0, size, size);
  const px = resized.data;

  const tensor = new Float32Array(1 * 3 * size * size);
  const planeSize = size * size;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const si = (y * size + x) * 4;
      const di = y * size + x;
      tensor[0 * planeSize + di] = (px[si]     - 127.5) / 128.0;
      tensor[1 * planeSize + di] = (px[si + 1] - 127.5) / 128.0;
      tensor[2 * planeSize + di] = (px[si + 2] - 127.5) / 128.0;
    }
  }

  return { tensor, scale: ratio };
}

// ── NMS ──────────────────────────────────────────────────────────

function nms(boxes, scores, threshold) {
  const n = boxes.length;
  if (n === 0) return [];

  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => scores[b] - scores[a]);

  const keep = [];
  const suppressed = new Uint8Array(n);

  for (const i of order) {
    if (suppressed[i]) continue;
    keep.push(i);

    const [ax1, ay1, ax2, ay2] = boxes[i];
    const areaA = (ax2 - ax1) * (ay2 - ay1);

    for (const j of order) {
      if (suppressed[j] || j === i) continue;

      const [bx1, by1, bx2, by2] = boxes[j];
      const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
      const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
      const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
      const inter = iw * ih;
      const areaB = (bx2 - bx1) * (by2 - by1);
      const iou = inter / (areaA + areaB - inter);

      if (iou > threshold) suppressed[j] = 1;
    }
  }
  return keep;
}

// ── Detection ────────────────────────────────────────────────────

async function detectFaces(pixels, width, height) {
  const { tensor, scale } = preprocessDetect(pixels, width, height);

  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, DET_INPUT_SIZE, DET_INPUT_SIZE]);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  const results = await session.run(feeds);
  const outputNames = session.outputNames;

  const numScales = DET_STRIDES.length;
  const allBoxes = [];
  const allScores = [];
  const allKps = [];

  for (let si = 0; si < numScales; si++) {
    const stride = DET_STRIDES[si];
    const fmH = Math.floor(DET_INPUT_SIZE / stride);
    const fmW = Math.floor(DET_INPUT_SIZE / stride);

    const scoresData = results[outputNames[si]].data;
    const bboxData   = results[outputNames[si + numScales]].data;
    const kpsData    = results[outputNames[si + numScales * 2]].data;

    for (let row = 0; row < fmH; row++) {
      for (let col = 0; col < fmW; col++) {
        const cx = col * stride;
        const cy = row * stride;

        for (let a = 0; a < DET_NUM_ANCHORS; a++) {
          const idx = (row * fmW + col) * DET_NUM_ANCHORS + a;
          const score = scoresData[idx];
          if (score < DET_SCORE_THRESH) continue;

          const bi = idx * 4;
          const x1 = (cx - bboxData[bi]     * stride) / scale;
          const y1 = (cy - bboxData[bi + 1] * stride) / scale;
          const x2 = (cx + bboxData[bi + 2] * stride) / scale;
          const y2 = (cy + bboxData[bi + 3] * stride) / scale;

          const ki = idx * 10;
          const kps = [];
          for (let k = 0; k < 5; k++) {
            kps.push([
              (cx + kpsData[ki + k * 2]     * stride) / scale,
              (cy + kpsData[ki + k * 2 + 1] * stride) / scale,
            ]);
          }

          allBoxes.push([x1, y1, x2, y2]);
          allScores.push(score);
          allKps.push(kps);
        }
      }
    }
  }

  const keep = nms(allBoxes, allScores, DET_NMS_THRESH);
  return keep.map(i => ({
    bbox: allBoxes[i],
    kps: allKps[i],
    score: allScores[i],
  }));
}

async function detectOneFace(pixels, width, height) {
  const faces = await detectFaces(pixels, width, height);
  if (faces.length === 0) return null;
  // Pick the largest face (best for single-user webcam)
  faces.sort((a, b) => {
    const areaA = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
    const areaB = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
    return areaB - areaA;
  });
  return faces[0];
}

// ── Message Handler ──────────────────────────────────────────────

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      session = await ort.InferenceSession.create(e.data.modelBytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
        enableMemPattern: true,
        logSeverityLevel: 3,
      });
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  else if (type === 'detect') {
    try {
      const face = await detectOneFace(
        new Uint8ClampedArray(e.data.pixels),
        e.data.width,
        e.data.height
      );
      self.postMessage({ type: 'result', face, id: e.data.id, gen: e.data.gen });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message, id: e.data.id, gen: e.data.gen });
    }
  }
};
