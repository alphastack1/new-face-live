/**
 * ML pipeline: face detection, recognition, swapping, parsing, blending.
 * All inference runs through onnxruntime-web (WebGPU or WASM fallback).
 */

import {
  estimateSimilarityTransform, invertAffine, affinePoint,
  warpAffine, warpAffineMask, nms, vecNormalize, vecMatMul,
} from './math.js?v=8';

// ── Constants ──────────────────────────────────────────────────────

// Canonical face landmarks for ArcFace alignment (112×112 space)
const ARCFACE_DST_112 = [
  [38.2946, 51.6963],  // left eye
  [73.5318, 51.5014],  // right eye
  [56.0252, 71.7366],  // nose tip
  [41.5493, 92.3655],  // left mouth
  [70.7299, 92.2041],  // right mouth
];

// For 128×128 inswapper input: scale all coordinates from 112-space to 128-space
// (insightface does: src = arcface_src * image_size / 112)
const ARCFACE_DST_128 = ARCFACE_DST_112.map(([x, y]) => [x * 128 / 112, y * 128 / 112]);

// Detection model config
const DET_INPUT_SIZE = 192;
const DET_STRIDES = [8, 16, 32];
const DET_NUM_ANCHORS = 2;
const DET_SCORE_THRESH = 0.5;
const DET_NMS_THRESH = 0.4;

// BiSeNet parsing classes
const REGION_CLASSES = {
  nose: [10],
  lips: [11, 12, 13],
  eyes: [4, 5],
  brow: [2, 3],
  chin: [1],
};

// ── Face Detection (RetinaFace / det_10g) ──────────────────────────

/**
 * Preprocess a video frame for face detection.
 * Resizes preserving aspect ratio, pads to square, normalizes.
 *
 * @param {ImageData} imgData - Source frame (RGBA)
 * @returns {{ tensor: Float32Array, scale: number, padW: number, padH: number }}
 */
// Cached canvases for detection preprocessing (avoid per-frame allocation)
let _detSrcCanvas = null, _detSrcCtx = null;
let _detResCanvas = null, _detResCtx = null;

export function preprocessDetect(imgData) {
  const { width: W, height: H, data } = imgData;
  const size = DET_INPUT_SIZE;

  // Compute resize (preserve aspect ratio)
  const ratio = Math.min(size / W, size / H);
  const newW = Math.round(W * ratio);
  const newH = Math.round(H * ratio);

  // Resize using cached offscreen canvases
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

  const resized = _detResCtx.getImageData(0, 0, size, size);
  const px = resized.data;

  // Convert RGBA → RGB float32 NCHW, normalize: (px - 127.5) / 128.0
  const tensor = new Float32Array(1 * 3 * size * size);
  const planeSize = size * size;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const si = (y * size + x) * 4;
      const di = y * size + x;
      tensor[0 * planeSize + di] = (px[si]     - 127.5) / 128.0;  // R
      tensor[1 * planeSize + di] = (px[si + 1] - 127.5) / 128.0;  // G
      tensor[2 * planeSize + di] = (px[si + 2] - 127.5) / 128.0;  // B
    }
  }

  return { tensor, scale: ratio, newW, newH };
}

/**
 * Run face detection and decode outputs.
 * @param {ort.InferenceSession} session
 * @param {ImageData} imgData
 * @returns {Promise<Array<{bbox: number[], kps: number[][], score: number}>>}
 */
export async function detectFaces(session, imgData) {
  const { tensor, scale } = preprocessDetect(imgData);

  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, DET_INPUT_SIZE, DET_INPUT_SIZE]);
  const feeds = {};
  const inputName = session.inputNames[0];
  feeds[inputName] = inputTensor;

  const results = await session.run(feeds);
  const outputNames = session.outputNames;

  // Decode multi-scale outputs
  // Output order: [scores_s8, scores_s16, scores_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]
  const numScales = DET_STRIDES.length;
  const allBoxes = [];
  const allScores = [];
  const allKps = [];

  for (let si = 0; si < numScales; si++) {
    const stride = DET_STRIDES[si];
    const fmH = Math.floor(DET_INPUT_SIZE / stride);
    const fmW = Math.floor(DET_INPUT_SIZE / stride);
    const numAnchors = fmH * fmW * DET_NUM_ANCHORS;

    const scoresData = results[outputNames[si]].data;
    const bboxData   = results[outputNames[si + numScales]].data;
    const kpsData    = results[outputNames[si + numScales * 2]].data;

    // Generate anchor centers
    for (let row = 0; row < fmH; row++) {
      for (let col = 0; col < fmW; col++) {
        const cx = col * stride;
        const cy = row * stride;

        for (let a = 0; a < DET_NUM_ANCHORS; a++) {
          const idx = (row * fmW + col) * DET_NUM_ANCHORS + a;
          const score = scoresData[idx];
          if (score < DET_SCORE_THRESH) continue;

          // Decode bbox: distance from anchor to edges
          const bi = idx * 4;
          const x1 = (cx - bboxData[bi])     / scale;
          const y1 = (cy - bboxData[bi + 1]) / scale;
          const x2 = (cx + bboxData[bi + 2]) / scale;
          const y2 = (cy + bboxData[bi + 3]) / scale;

          // Decode keypoints
          const ki = idx * 10;
          const kps = [];
          for (let k = 0; k < 5; k++) {
            kps.push([
              (cx + kpsData[ki + k * 2])     / scale,
              (cy + kpsData[ki + k * 2 + 1]) / scale,
            ]);
          }

          allBoxes.push([x1, y1, x2, y2]);
          allScores.push(score);
          allKps.push(kps);
        }
      }
    }
  }

  // NMS
  const keep = nms(allBoxes, allScores, DET_NMS_THRESH);
  return keep.map(i => ({
    bbox: allBoxes[i],
    kps: allKps[i],
    score: allScores[i],
  }));
}

/**
 * Get the primary face (leftmost / largest).
 */
export async function detectOneFace(session, imgData) {
  const faces = await detectFaces(session, imgData);
  if (faces.length === 0) return null;
  // Return face with smallest x1 (leftmost), matching Python behavior
  faces.sort((a, b) => a.bbox[0] - b.bbox[0]);
  return faces[0];
}

// ── Face Alignment ─────────────────────────────────────────────────

/**
 * Align a face to a canonical position using 5 keypoints.
 *
 * @param {Uint8ClampedArray} srcData - Source RGBA pixels
 * @param {number} srcW
 * @param {number} srcH
 * @param {number[][]} kps - 5 keypoints [[x,y], ...]
 * @param {number} outSize - Output size (112 for recognition, 128 for swap)
 * @returns {{ data: Uint8ClampedArray, M: number[][] }}
 */
export function alignFace(srcData, srcW, srcH, kps, outSize) {
  const dst = outSize === 128 ? ARCFACE_DST_128 : ARCFACE_DST_112;
  const M = estimateSimilarityTransform(kps, dst);
  const data = warpAffine(srcData, srcW, srcH, M, outSize, outSize);
  return { data, M };
}

// ── ArcFace Embedding Extraction ───────────────────────────────────

/**
 * Extract a 512-dim face embedding from an aligned 112×112 face.
 * @param {ort.InferenceSession} session - w600k_r50 session
 * @param {Uint8ClampedArray} alignedRGBA - 112×112 RGBA pixels
 * @returns {Promise<Float32Array>} Normalized 512-dim embedding
 */
export async function extractEmbedding(session, alignedRGBA) {
  const size = 112;
  const planeSize = size * size;
  const tensor = new Float32Array(1 * 3 * planeSize);

  // RGBA → RGB NCHW, normalize: (px - 127.5) / 127.5
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

  // L2 normalize
  vecNormalize(embedding);
  return embedding;
}

// ── InSwapper Face Swap ────────────────────────────────────────────

/**
 * Run the inswapper model to generate a swapped face.
 *
 * @param {ort.InferenceSession} session - inswapper session
 * @param {Uint8ClampedArray} alignedRGBA - 128×128 target face RGBA
 * @param {Float32Array} sourceLatent - 512-dim projected+normalized embedding
 * @returns {Promise<Uint8ClampedArray>} 128×128 swapped face RGBA
 */
export async function runSwap(session, alignedRGBA, sourceLatent) {
  const size = 128;
  const planeSize = size * size;

  // Preprocess: RGBA → RGB NCHW, normalize to [0, 1]
  const imgTensor = new Float32Array(1 * 3 * planeSize);
  for (let i = 0; i < planeSize; i++) {
    const si = i * 4;
    imgTensor[0 * planeSize + i] = alignedRGBA[si]     / 255.0;
    imgTensor[1 * planeSize + i] = alignedRGBA[si + 1] / 255.0;
    imgTensor[2 * planeSize + i] = alignedRGBA[si + 2] / 255.0;
  }

  const targetTensor = new ort.Tensor('float32', imgTensor, [1, 3, size, size]);
  // Clone sourceLatent — ORT proxy mode may transfer the underlying buffer
  const sourceTensor = new ort.Tensor('float32', new Float32Array(sourceLatent), [1, 512]);

  // Use named feeds (safer than positional — input order may vary)
  const feeds = { 'target': targetTensor, 'source': sourceTensor };

  // One-time fixed-input test to compare WebGPU vs Python CPU
  if (!runSwap._fixedTestDone) {
    runSwap._fixedTestDone = true;
    console.log('[FIXED TEST] Running with known inputs...');
    const fixedImg = new Float32Array(1 * 3 * planeSize);
    fixedImg.fill(0.5); // uniform grey
    const fixedLat = new Float32Array(512);
    // Simple known pattern: [0.044, 0.044, 0.044, ...]  (≈ normalized 512-dim)
    for (let i = 0; i < 512; i++) fixedLat[i] = 1.0 / Math.sqrt(512);
    const fixedTarget = new ort.Tensor('float32', fixedImg, [1, 3, size, size]);
    const fixedSource = new ort.Tensor('float32', fixedLat, [1, 512]);
    const fixedResults = await session.run({ 'target': fixedTarget, 'source': fixedSource });
    const fixedOut = fixedResults[session.outputNames[0]];
    const fixedData = fixedOut.getData ? await fixedOut.getData() : fixedOut.data;
    // Log first 20 values of each channel
    const r20 = Array.from(fixedData.slice(0, 20)).map(v => v.toFixed(4));
    const g20 = Array.from(fixedData.slice(planeSize, planeSize + 20)).map(v => v.toFixed(4));
    const b20 = Array.from(fixedData.slice(planeSize * 2, planeSize * 2 + 20)).map(v => v.toFixed(4));
    console.log('[FIXED TEST] R[0:20]:', r20.join(', '));
    console.log('[FIXED TEST] G[0:20]:', g20.join(', '));
    console.log('[FIXED TEST] B[0:20]:', b20.join(', '));

    // Also test with ZERO latent to see if model uses it at all
    const zeroLat = new Float32Array(512); // all zeros
    const zeroSource = new ort.Tensor('float32', zeroLat, [1, 512]);
    const zeroResults = await session.run({ 'target': fixedTarget, 'source': zeroSource });
    const zeroOut = zeroResults[session.outputNames[0]];
    const zeroData = zeroOut.getData ? await zeroOut.getData() : zeroOut.data;
    const zr20 = Array.from(zeroData.slice(0, 20)).map(v => v.toFixed(4));
    console.log('[FIXED TEST] Zero-latent R[0:20]:', zr20.join(', '));
    // Check if latent matters
    let latentDiff = 0;
    for (let i = 0; i < fixedData.length; i++) latentDiff += Math.abs(fixedData[i] - zeroData[i]);
    console.log('[FIXED TEST] Latent impact (total abs diff):', latentDiff.toFixed(4));
  }

  const results = await session.run(feeds);
  const outTensor = results[session.outputNames[0]];
  const outData = outTensor.getData ? await outTensor.getData() : outTensor.data;

  // Convert NCHW float32 [0,1] → RGBA uint8
  const rgba = new Uint8ClampedArray(planeSize * 4);
  for (let i = 0; i < planeSize; i++) {
    rgba[i * 4]     = Math.round(outData[0 * planeSize + i] * 255);
    rgba[i * 4 + 1] = Math.round(outData[1 * planeSize + i] * 255);
    rgba[i * 4 + 2] = Math.round(outData[2 * planeSize + i] * 255);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/**
 * Project source embedding through emap matrix for inswapper input.
 * @param {Float32Array} embedding - 512-dim normed embedding
 * @param {Float32Array} emap - 512×512 matrix (row-major)
 * @returns {Float32Array} 512-dim projected and normalized latent
 */
export function projectEmbedding(embedding, emap) {
  // latent = embedding @ emap (row-vector × matrix: (1,512) × (512,512) → (1,512))
  const latent = vecMatMul(embedding, emap, 512, 512);
  vecNormalize(latent);
  return latent;
}

// ── Paste-Back with Blending Mask ──────────────────────────────────

/**
 * Create a circular blending mask for a 128×128 face.
 * Returns a Float32Array (128×128) with soft edges.
 */
function createSwapMask(size) {
  const mask = new Float32Array(size * size);
  const cx = size / 2, cy = size / 2;
  const r = size * 0.42; // slightly smaller than half
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
 * Paste a swapped 128×128 face back into the full frame.
 *
 * @param {Uint8ClampedArray} frameRGBA - Full frame RGBA
 * @param {number} frameW
 * @param {number} frameH
 * @param {Uint8ClampedArray} swappedRGBA - 128×128 swapped face RGBA
 * @param {number[][]} M - Forward affine (used for alignment)
 * @returns {Uint8ClampedArray} Modified frame RGBA
 */
export function pasteBack(frameRGBA, frameW, frameH, swappedRGBA, M) {
  const size = 128;
  const Minv = invertAffine(M);

  // Compute bounding box of the face in frame coords by projecting the 128×128 corners
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
  // Pad by 10% and clamp
  const pad = Math.max((maxX - minX), (maxY - minY)) * 0.1;
  const bx1 = Math.max(0, Math.floor(minX - pad));
  const by1 = Math.max(0, Math.floor(minY - pad));
  const bx2 = Math.min(frameW, Math.ceil(maxX + pad));
  const by2 = Math.min(frameH, Math.ceil(maxY + pad));

  // Only iterate within the face bounding box (huge speedup vs full frame)
  const out = new Uint8ClampedArray(frameRGBA);
  for (let y = by1; y < by2; y++) {
    for (let x = bx1; x < bx2; x++) {
      // Inverse-map frame (x,y) → source (128×128) coords
      const sx = M[0][0] * x + M[0][1] * y + M[0][2];
      const sy = M[1][0] * x + M[1][1] * y + M[1][2];

      // Skip if outside 128×128 source
      if (sx < 0 || sx >= size - 1 || sy < 0 || sy >= size - 1) continue;

      // Bilinear lookup in swap mask
      const ix = Math.floor(sx), iy = Math.floor(sy);
      const fx = sx - ix, fy = sy - iy;
      const mi = iy * size + ix;
      const alpha =
        SWAP_MASK_128[mi] * (1 - fx) * (1 - fy) +
        SWAP_MASK_128[mi + 1] * fx * (1 - fy) +
        SWAP_MASK_128[mi + size] * (1 - fx) * fy +
        SWAP_MASK_128[mi + size + 1] * fx * fy;

      if (alpha < 0.001) continue;

      // Bilinear lookup in swapped face
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

/**
 * Run BiSeNet face parsing on a face crop.
 *
 * @param {ort.InferenceSession} session
 * @param {Uint8ClampedArray} cropRGBA - Face crop RGBA
 * @param {number} cropW
 * @param {number} cropH
 * @returns {Promise<Uint8Array>} Class labels (cropH × cropW), values 0-18
 */
export async function parseFace(session, cropRGBA, cropW, cropH) {
  const size = 512;
  const planeSize = size * size;

  // Resize crop to 512×512
  const srcCanvas = new OffscreenCanvas(cropW, cropH);
  const srcCtx = srcCanvas.getContext('2d');
  srcCtx.putImageData(new ImageData(cropRGBA, cropW, cropH), 0, 0);

  const resCanvas = new OffscreenCanvas(size, size);
  const resCtx = resCanvas.getContext('2d');
  resCtx.drawImage(srcCanvas, 0, 0, cropW, cropH, 0, 0, size, size);
  const resized = resCtx.getImageData(0, 0, size, size).data;

  // RGBA → RGB NCHW, normalize: (px - 127.5) / 127.5
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
  const logits = outTensor.getData ? await outTensor.getData() : outTensor.data; // (1, 19, 512, 512)

  // Argmax over class dimension → (512, 512) class labels
  const labels512 = new Uint8Array(planeSize);
  for (let i = 0; i < planeSize; i++) {
    let maxVal = -Infinity, maxCls = 0;
    for (let c = 0; c < 19; c++) {
      const v = logits[c * planeSize + i];
      if (v > maxVal) { maxVal = v; maxCls = c; }
    }
    labels512[i] = maxCls;
  }

  // Resize labels back to crop size (nearest neighbor)
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

/**
 * Run face parsing on the full frame, cropping around the detected face.
 * Returns a full-frame label map.
 *
 * @param {ort.InferenceSession} session
 * @param {ImageData} frameData - Full frame
 * @param {number[]} bbox - [x1, y1, x2, y2] face bbox
 * @returns {Promise<{labels: Uint8Array, cropBox: number[]}>}
 */
export async function parseFullFrame(session, frameData, bbox) {
  const { width: W, height: H, data } = frameData;

  // Pad bbox by 25%
  const bw = bbox[2] - bbox[0], bh = bbox[3] - bbox[1];
  const px = Math.round(bw * 0.25), py = Math.round(bh * 0.25);
  const x1 = Math.max(0, Math.round(bbox[0]) - px);
  const y1 = Math.max(0, Math.round(bbox[1]) - py);
  const x2 = Math.min(W, Math.round(bbox[2]) + px);
  const y2 = Math.min(H, Math.round(bbox[3]) + py);
  const cropW = x2 - x1, cropH = y2 - y1;

  // Extract crop RGBA
  const cropRGBA = new Uint8ClampedArray(cropW * cropH * 4);
  for (let row = 0; row < cropH; row++) {
    const srcOff = ((y1 + row) * W + x1) * 4;
    const dstOff = row * cropW * 4;
    cropRGBA.set(data.subarray(srcOff, srcOff + cropW * 4), dstOff);
  }

  const labels = await parseFace(session, cropRGBA, cropW, cropH);
  return { labels, cropBox: [x1, y1, x2, y2], cropW, cropH };
}

/**
 * Create a feathered region mask from parsing labels.
 *
 * @param {Uint8Array} labels - Parsing labels for the crop
 * @param {number} cropW
 * @param {number} cropH
 * @param {string} region - 'nose', 'lips', 'eyes', 'brow', 'chin'
 * @param {number[]} cropBox - [x1, y1, x2, y2] in frame coords
 * @param {number[][]} kps - 5 keypoints (for chin special case)
 * @param {number} frameW
 * @param {number} frameH
 * @returns {Float32Array} Full-frame mask (frameH × frameW), values 0-1
 */
export function createRegionMask(labels, cropW, cropH, region, cropBox, kps, frameW, frameH) {
  const classes = REGION_CLASSES[region];
  if (!classes) return null;

  // Binary mask within crop
  const cropMask = new Uint8Array(cropW * cropH);
  for (let i = 0; i < cropW * cropH; i++) {
    cropMask[i] = classes.includes(labels[i]) ? 255 : 0;
  }

  // Chin special case: zero out everything above mouth line
  if (region === 'chin' && kps) {
    const mouthY = (kps[3][1] + kps[4][1]) / 2;
    const localMouthY = Math.round(mouthY - cropBox[1]);
    for (let y = 0; y < Math.min(localMouthY, cropH); y++) {
      for (let x = 0; x < cropW; x++) {
        cropMask[y * cropW + x] = 0;
      }
    }
  }

  // Project crop mask into full frame
  const [x1, y1] = cropBox;
  const fullMask = new Float32Array(frameW * frameH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      fullMask[(y1 + y) * frameW + (x1 + x)] = cropMask[y * cropW + x] / 255.0;
    }
  }

  // Gaussian blur for feathering
  const faceW = cropBox[2] - cropBox[0];
  const feather = Math.max(5, Math.round(faceW * 0.07));
  gaussianBlurInPlace(fullMask, frameW, frameH, feather);

  // Renormalize to [0, 1]
  let maxVal = 0;
  for (let i = 0; i < fullMask.length; i++) if (fullMask[i] > maxVal) maxVal = fullMask[i];
  if (maxVal > 0) for (let i = 0; i < fullMask.length; i++) fullMask[i] /= maxVal;

  return fullMask;
}

/**
 * Simple box-blur approximation of Gaussian blur (3 passes).
 * Operates in-place on a Float32Array representing a 2D grid.
 */
function gaussianBlurInPlace(data, w, h, radius) {
  const temp = new Float32Array(w * h);

  for (let pass = 0; pass < 3; pass++) {
    // Horizontal pass
    for (let y = 0; y < h; y++) {
      let sum = 0, count = 0;
      // Initialize window
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

    // Vertical pass
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
 * Blend swapped frame with original using a region mask and opacity.
 *
 * @param {Uint8ClampedArray} original - Original frame RGBA
 * @param {Uint8ClampedArray} swapped - Full-swap result RGBA
 * @param {Float32Array|null} regionMask - Per-pixel mask (0-1), null = full blend
 * @param {number} opacity - Global opacity (0-1)
 * @param {number} w
 * @param {number} h
 * @returns {Uint8ClampedArray} Blended RGBA
 */
export function blendRegion(original, swapped, regionMask, opacity, w, h) {
  const out = new Uint8ClampedArray(original);
  const n = w * h;

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

/**
 * Apply unsharp mask sharpening.
 * @param {Uint8ClampedArray} rgba
 * @param {number} w
 * @param {number} h
 * @param {number} amount - 0-100
 * @returns {Uint8ClampedArray}
 */
export function sharpen(rgba, w, h, amount, regionMask) {
  if (amount <= 0) return rgba;
  const strength = amount / 50.0;
  const out = new Uint8ClampedArray(rgba);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      // Skip pixels outside the region mask if provided
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
