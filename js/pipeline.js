/**
 * ML pipeline: face detection, recognition, swapping, parsing, blending.
 * All inference runs through onnxruntime-web (WebGPU or WASM fallback).
 *
 * Performance: hot-path buffers are pre-allocated and reused across frames.
 */

import {
  estimateSimilarityTransform, invertAffine, affinePoint,
  warpAffine, warpAffineMask, nms, vecNormalize, vecMatMul,
} from './math.js?v=14';

// ── Constants ──────────────────────────────────────────────────────

const ARCFACE_DST_112 = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

const ARCFACE_DST_128 = ARCFACE_DST_112.map(([x, y]) => [x + 8.0, y]);

const DET_INPUT_SIZE = 192;
const DET_STRIDES = [8, 16, 32];
const DET_NUM_ANCHORS = 2;
const DET_SCORE_THRESH = 0.5;
const DET_NMS_THRESH = 0.4;

const REGION_CLASSES = {
  nose: [10],
  lips: [11, 12, 13],
  eyes: [4, 5],
  brow: [2, 3],
  chin: [1],
};

// ── Pre-allocated Buffers ─────────────────────────────────────────
// Reused across frames to reduce GC pressure.

const SWAP_SIZE = 128;
const SWAP_PLANE = SWAP_SIZE * SWAP_SIZE;

// runSwap: input tensor + output RGBA
const _swapTensor = new Float32Array(1 * 3 * SWAP_PLANE);
const _swapRGBA = new Uint8ClampedArray(SWAP_PLANE * 4);

// Detection preprocessing
let _detSrcCanvas = null, _detSrcCtx = null;
let _detResCanvas = null, _detResCtx = null;
const _detTensor = new Float32Array(1 * 3 * DET_INPUT_SIZE * DET_INPUT_SIZE);

// parseFace: cached canvases
let _parseSrcCanvas = null, _parseSrcCtx = null;
let _parseResCanvas = null, _parseResCtx = null;

// ── Face Detection (RetinaFace / det_10g) ──────────────────────────

export function preprocessDetect(imgData) {
  const { width: W, height: H } = imgData;
  const size = DET_INPUT_SIZE;

  const ratio = Math.min(size / W, size / H);
  const newW = Math.round(W * ratio);
  const newH = Math.round(H * ratio);

  if (!_detSrcCanvas || _detSrcCanvas.width !== W || _detSrcCanvas.height !== H) {
    _detSrcCanvas = new OffscreenCanvas(W, H);
    _detSrcCtx = _detSrcCanvas.getContext('2d');
  }
  _detSrcCtx.putImageData(imgData, 0, 0);

  if (!_detResCanvas) {
    _detResCanvas = new OffscreenCanvas(size, size);
    _detResCtx = _detResCanvas.getContext('2d');
  }
  _detResCtx.clearRect(0, 0, size, size);
  _detResCtx.drawImage(_detSrcCanvas, 0, 0, W, H, 0, 0, newW, newH);

  const px = _detResCtx.getImageData(0, 0, size, size).data;
  const planeSize = size * size;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const si = (y * size + x) * 4;
      const di = y * size + x;
      _detTensor[0 * planeSize + di] = (px[si]     - 127.5) / 128.0;
      _detTensor[1 * planeSize + di] = (px[si + 1] - 127.5) / 128.0;
      _detTensor[2 * planeSize + di] = (px[si + 2] - 127.5) / 128.0;
    }
  }

  return { tensor: _detTensor, scale: ratio, newW, newH };
}

export async function detectFaces(session, imgData) {
  const { tensor, scale } = preprocessDetect(imgData);

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

export async function detectOneFace(session, imgData) {
  const faces = await detectFaces(session, imgData);
  if (faces.length === 0) return null;
  faces.sort((a, b) => a.bbox[0] - b.bbox[0]);
  return faces[0];
}

// ── Face Alignment ─────────────────────────────────────────────────

export function alignFace(srcData, srcW, srcH, kps, outSize) {
  const dst = outSize === 128 ? ARCFACE_DST_128 : ARCFACE_DST_112;
  const M = estimateSimilarityTransform(kps, dst);
  const data = warpAffine(srcData, srcW, srcH, M, outSize, outSize);
  return { data, M };
}

// ── ArcFace Embedding Extraction ───────────────────────────────────

export async function extractEmbedding(session, alignedRGBA) {
  const size = 112;
  const planeSize = size * size;
  const tensor = new Float32Array(1 * 3 * planeSize);

  for (let i = 0; i < planeSize; i++) {
    const si = i * 4;
    tensor[0 * planeSize + i] = (alignedRGBA[si]     - 127.5) / 127.5;
    tensor[1 * planeSize + i] = (alignedRGBA[si + 1] - 127.5) / 127.5;
    tensor[2 * planeSize + i] = (alignedRGBA[si + 2] - 127.5) / 127.5;
  }

  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, size, size]);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  const results = await session.run(feeds);
  const outTensor = results[session.outputNames[0]];
  const rawData = outTensor.getData ? await outTensor.getData() : outTensor.data;
  const embedding = new Float32Array(rawData);

  vecNormalize(embedding);
  return embedding;
}

// ── InSwapper Face Swap ────────────────────────────────────────────

export async function runSwap(session, alignedRGBA, sourceLatent) {
  // Reuse pre-allocated tensor buffer
  for (let i = 0; i < SWAP_PLANE; i++) {
    const si = i * 4;
    _swapTensor[0 * SWAP_PLANE + i] = alignedRGBA[si]     / 255.0;
    _swapTensor[1 * SWAP_PLANE + i] = alignedRGBA[si + 1] / 255.0;
    _swapTensor[2 * SWAP_PLANE + i] = alignedRGBA[si + 2] / 255.0;
  }

  const targetTensor = new ort.Tensor('float32', _swapTensor, [1, 3, SWAP_SIZE, SWAP_SIZE]);
  const sourceTensor = new ort.Tensor('float32', new Float32Array(sourceLatent), [1, 512]);
  const feeds = { 'target': targetTensor, 'source': sourceTensor };

  const results = await session.run(feeds);
  const outTensor = results[session.outputNames[0]];
  const outData = outTensor.getData ? await outTensor.getData() : outTensor.data;

  // Reuse pre-allocated RGBA buffer
  for (let i = 0; i < SWAP_PLANE; i++) {
    _swapRGBA[i * 4]     = Math.round(outData[0 * SWAP_PLANE + i] * 255);
    _swapRGBA[i * 4 + 1] = Math.round(outData[1 * SWAP_PLANE + i] * 255);
    _swapRGBA[i * 4 + 2] = Math.round(outData[2 * SWAP_PLANE + i] * 255);
    _swapRGBA[i * 4 + 3] = 255;
  }
  return _swapRGBA;
}

export function projectEmbedding(embedding, emap) {
  const latent = vecMatMul(embedding, emap, 512, 512);
  vecNormalize(latent);
  return latent;
}

// ── Paste-Back with Blending Mask ──────────────────────────────────

function createSwapMask(size) {
  const mask = new Float32Array(size * size);
  const cx = size / 2, cy = size / 2;
  const r = size * 0.42;
  const feather = size * 0.1;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      if (dist < r - feather) {
        mask[y * size + x] = 1.0;
      } else if (dist < r) {
        mask[y * size + x] = (r - dist) / feather;
      }
    }
  }
  return mask;
}

const SWAP_MASK_128 = createSwapMask(128);

/**
 * Paste swapped face back into frame. Writes directly into the output buffer.
 * Pass an existing buffer as `outBuf` to avoid allocation, or null to create one.
 */
export function pasteBack(frameRGBA, frameW, frameH, swappedRGBA, M, outBuf) {
  const size = 128;
  const Minv = invertAffine(M);

  // Compute face bounding box in frame coords
  const corners = [[0,0],[size,0],[size,size],[0,size]];
  let minX = frameW, minY = frameH, maxX = 0, maxY = 0;
  for (const [cx, cy] of corners) {
    const fx = Minv[0][0] * cx + Minv[0][1] * cy + Minv[0][2];
    const fy = Minv[1][0] * cx + Minv[1][1] * cy + Minv[1][2];
    if (fx < minX) minX = fx;
    if (fx > maxX) maxX = fx;
    if (fy < minY) minY = fy;
    if (fy > maxY) maxY = fy;
  }
  const pad = Math.max((maxX - minX), (maxY - minY)) * 0.1;
  const bx1 = Math.max(0, Math.floor(minX - pad));
  const by1 = Math.max(0, Math.floor(minY - pad));
  const bx2 = Math.min(frameW, Math.ceil(maxX + pad));
  const by2 = Math.min(frameH, Math.ceil(maxY + pad));

  // Copy frame into output (reuse buffer if provided)
  const out = outBuf && outBuf.length === frameRGBA.length
    ? (outBuf.set(frameRGBA), outBuf)
    : new Uint8ClampedArray(frameRGBA);

  for (let y = by1; y < by2; y++) {
    for (let x = bx1; x < bx2; x++) {
      const sx = M[0][0] * x + M[0][1] * y + M[0][2];
      const sy = M[1][0] * x + M[1][1] * y + M[1][2];

      if (sx < 0 || sx >= size - 1 || sy < 0 || sy >= size - 1) continue;

      const ix = Math.floor(sx), iy = Math.floor(sy);
      const fx = sx - ix, fy = sy - iy;
      const mi = iy * size + ix;
      const alpha =
        SWAP_MASK_128[mi] * (1 - fx) * (1 - fy) +
        SWAP_MASK_128[mi + 1] * fx * (1 - fy) +
        SWAP_MASK_128[mi + size] * (1 - fx) * fy +
        SWAP_MASK_128[mi + size + 1] * fx * fy;

      if (alpha < 0.001) continue;

      const si4 = (iy * size + ix) * 4;
      const pi = (y * frameW + x) * 4;
      for (let c = 0; c < 3; c++) {
        const v =
          swappedRGBA[si4 + c] * (1 - fx) * (1 - fy) +
          swappedRGBA[si4 + 4 + c] * fx * (1 - fy) +
          swappedRGBA[si4 + size * 4 + c] * (1 - fx) * fy +
          swappedRGBA[si4 + size * 4 + 4 + c] * fx * fy;
        out[pi + c] = frameRGBA[pi + c] * (1 - alpha) + v * alpha;
      }
      out[pi + 3] = 255;
    }
  }
  return out;
}

// ── BiSeNet Face Parsing ───────────────────────────────────────────

export async function parseFace(session, cropRGBA, cropW, cropH) {
  const size = 512;
  const planeSize = size * size;

  // Reuse cached canvases
  if (!_parseSrcCanvas || _parseSrcCanvas.width !== cropW || _parseSrcCanvas.height !== cropH) {
    _parseSrcCanvas = new OffscreenCanvas(cropW, cropH);
    _parseSrcCtx = _parseSrcCanvas.getContext('2d');
  }
  _parseSrcCtx.putImageData(new ImageData(cropRGBA, cropW, cropH), 0, 0);

  if (!_parseResCanvas) {
    _parseResCanvas = new OffscreenCanvas(size, size);
    _parseResCtx = _parseResCanvas.getContext('2d');
  }
  _parseResCtx.drawImage(_parseSrcCanvas, 0, 0, cropW, cropH, 0, 0, size, size);
  const resized = _parseResCtx.getImageData(0, 0, size, size).data;

  const tensor = new Float32Array(1 * 3 * planeSize);
  for (let i = 0; i < planeSize; i++) {
    const si = i * 4;
    tensor[0 * planeSize + i] = (resized[si]     - 127.5) / 127.5;
    tensor[1 * planeSize + i] = (resized[si + 1] - 127.5) / 127.5;
    tensor[2 * planeSize + i] = (resized[si + 2] - 127.5) / 127.5;
  }

  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, size, size]);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  const results = await session.run(feeds);
  const outTensor = results[session.outputNames[0]];
  const logits = outTensor.getData ? await outTensor.getData() : outTensor.data;

  const labels512 = new Uint8Array(planeSize);
  for (let i = 0; i < planeSize; i++) {
    let maxVal = -Infinity, maxCls = 0;
    for (let c = 0; c < 19; c++) {
      const v = logits[c * planeSize + i];
      if (v > maxVal) { maxVal = v; maxCls = c; }
    }
    labels512[i] = maxCls;
  }

  const labels = new Uint8Array(cropW * cropH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const sx = Math.round(x * (size - 1) / (cropW - 1));
      const sy = Math.round(y * (size - 1) / (cropH - 1));
      labels[y * cropW + x] = labels512[sy * size + sx];
    }
  }
  return labels;
}

export async function parseFullFrame(session, frameData, bbox) {
  const { width: W, height: H, data } = frameData;

  const bw = bbox[2] - bbox[0], bh = bbox[3] - bbox[1];
  const px = Math.round(bw * 0.25), py = Math.round(bh * 0.25);
  const x1 = Math.max(0, Math.round(bbox[0]) - px);
  const y1 = Math.max(0, Math.round(bbox[1]) - py);
  const x2 = Math.min(W, Math.round(bbox[2]) + px);
  const y2 = Math.min(H, Math.round(bbox[3]) + py);
  const cropW = x2 - x1, cropH = y2 - y1;

  const cropRGBA = new Uint8ClampedArray(cropW * cropH * 4);
  for (let row = 0; row < cropH; row++) {
    const srcOff = ((y1 + row) * W + x1) * 4;
    const dstOff = row * cropW * 4;
    cropRGBA.set(data.subarray(srcOff, srcOff + cropW * 4), dstOff);
  }

  const labels = await parseFace(session, cropRGBA, cropW, cropH);
  return { labels, cropBox: [x1, y1, x2, y2], cropW, cropH };
}

export function createRegionMask(labels, cropW, cropH, region, cropBox, kps, frameW, frameH) {
  const classes = REGION_CLASSES[region];
  if (!classes) return null;

  const cropMask = new Uint8Array(cropW * cropH);
  for (let i = 0; i < cropW * cropH; i++) {
    cropMask[i] = classes.includes(labels[i]) ? 255 : 0;
  }

  if (region === 'chin' && kps) {
    const mouthY = (kps[3][1] + kps[4][1]) / 2;
    const localMouthY = Math.round(mouthY - cropBox[1]);
    for (let y = 0; y < Math.min(localMouthY, cropH); y++) {
      for (let x = 0; x < cropW; x++) {
        cropMask[y * cropW + x] = 0;
      }
    }
  }

  const [x1, y1] = cropBox;
  const fullMask = new Float32Array(frameW * frameH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      fullMask[(y1 + y) * frameW + (x1 + x)] = cropMask[y * cropW + x] / 255.0;
    }
  }

  const faceW = cropBox[2] - cropBox[0];
  const feather = Math.max(5, Math.round(faceW * 0.07));
  gaussianBlurInPlace(fullMask, frameW, frameH, feather);

  let maxVal = 0;
  for (let i = 0; i < fullMask.length; i++) if (fullMask[i] > maxVal) maxVal = fullMask[i];
  if (maxVal > 0) for (let i = 0; i < fullMask.length; i++) fullMask[i] /= maxVal;

  return fullMask;
}

function gaussianBlurInPlace(data, w, h, radius) {
  const temp = new Float32Array(w * h);

  for (let pass = 0; pass < 3; pass++) {
    for (let y = 0; y < h; y++) {
      let sum = 0, count = 0;
      for (let x = 0; x <= radius && x < w; x++) {
        sum += data[y * w + x];
        count++;
      }
      for (let x = 0; x < w; x++) {
        temp[y * w + x] = sum / count;
        const addX = x + radius + 1;
        const remX = x - radius;
        if (addX < w) { sum += data[y * w + addX]; count++; }
        if (remX >= 0) { sum -= data[y * w + remX]; count--; }
      }
    }

    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let y = 0; y <= radius && y < h; y++) {
        sum += temp[y * w + x];
        count++;
      }
      for (let y = 0; y < h; y++) {
        data[y * w + x] = sum / count;
        const addY = y + radius + 1;
        const remY = y - radius;
        if (addY < h) { sum += temp[addY * w + x]; count++; }
        if (remY >= 0) { sum -= temp[remY * w + x]; count--; }
      }
    }
  }
}

// ── Blending & Post-Processing ─────────────────────────────────────

/**
 * Blend swapped frame with original. Writes into outBuf if provided.
 */
export function blendRegion(original, swapped, regionMask, opacity, w, h, outBuf) {
  const out = outBuf && outBuf.length === original.length
    ? outBuf
    : new Uint8ClampedArray(original.length);
  const n = w * h;

  // Copy original first, then blend changed pixels
  out.set(original);

  for (let i = 0; i < n; i++) {
    const alpha = (regionMask ? regionMask[i] : 1.0) * opacity;
    if (alpha < 0.001) continue;
    const pi = i * 4;
    out[pi]     = original[pi]     * (1 - alpha) + swapped[pi]     * alpha;
    out[pi + 1] = original[pi + 1] * (1 - alpha) + swapped[pi + 1] * alpha;
    out[pi + 2] = original[pi + 2] * (1 - alpha) + swapped[pi + 2] * alpha;
  }
  return out;
}

export function sharpen(rgba, w, h, amount, regionMask) {
  if (amount <= 0) return rgba;
  const strength = amount / 50.0;
  const out = new Uint8ClampedArray(rgba);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      if (regionMask && regionMask[y * w + x] < 0.001) continue;
      const ci = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        const center = rgba[ci + c];
        const blur =
          (rgba[((y-1)*w + x-1) * 4 + c] + rgba[((y-1)*w + x) * 4 + c] + rgba[((y-1)*w + x+1) * 4 + c] +
           rgba[(y*w + x-1) * 4 + c]      +                                rgba[(y*w + x+1) * 4 + c] +
           rgba[((y+1)*w + x-1) * 4 + c] + rgba[((y+1)*w + x) * 4 + c] + rgba[((y+1)*w + x+1) * 4 + c]) / 8;
        const sharpened = center + (center - blur) * strength;
        out[ci + c] = Math.max(0, Math.min(255, Math.round(sharpened)));
      }
    }
  }
  return out;
}
