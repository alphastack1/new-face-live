/**
 * Math utilities for face alignment: affine transforms, NMS, linear algebra.
 */

// ── Similarity Transform Estimation ────────────────────────────────

/**
 * Estimate a 2D similarity transform (rotation + scale + translation)
 * from source points to destination points via least-squares.
 *
 * @param {number[][]} src - Array of [x,y] source points
 * @param {number[][]} dst - Array of [x,y] destination points
 * @returns {number[][]} 2x3 affine matrix [[a, -b, tx], [b, a, ty]]
 */
export function estimateSimilarityTransform(src, dst) {
  const n = src.length;

  // Build normal equations for: dx = a*sx - b*sy + tx, dy = b*sx + a*sy + ty
  // Unknowns: [a, b, tx, ty]
  const ATA = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  const ATb = [0,0,0,0];

  for (let i = 0; i < n; i++) {
    const sx = src[i][0], sy = src[i][1];
    const dx = dst[i][0], dy = dst[i][1];

    // Row 1: [sx, -sy, 1, 0] → dx
    // Row 2: [sy,  sx, 0, 1] → dy
    const r1 = [sx, -sy, 1, 0];
    const r2 = [sy,  sx, 0, 1];

    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 4; k++) {
        ATA[j][k] += r1[j] * r1[k] + r2[j] * r2[k];
      }
      ATb[j] += r1[j] * dx + r2[j] * dy;
    }
  }

  const params = solve4x4(ATA, ATb);
  const [a, b, tx, ty] = params;

  return [[a, -b, tx], [b, a, ty]];
}

/**
 * Solve a 4x4 linear system Ax = b via Gaussian elimination with partial pivoting.
 */
function solve4x4(A, b) {
  // Deep copy
  const M = A.map(r => [...r]);
  const rhs = [...b];
  const n = 4;

  // Forward elimination
  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxVal = Math.abs(M[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxVal) {
        maxVal = Math.abs(M[row][col]);
        maxRow = row;
      }
    }
    if (maxRow !== col) {
      [M[col], M[maxRow]] = [M[maxRow], M[col]];
      [rhs[col], rhs[maxRow]] = [rhs[maxRow], rhs[col]];
    }

    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-12) continue;

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / pivot;
      for (let k = col; k < n; k++) {
        M[row][k] -= factor * M[col][k];
      }
      rhs[row] -= factor * rhs[col];
    }
  }

  // Back substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = rhs[i];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = sum / M[i][i];
  }
  return x;
}

// ── Affine Matrix Operations ───────────────────────────────────────

/**
 * Invert a 2x3 affine matrix.
 * @param {number[][]} M - 2x3 matrix
 * @returns {number[][]} 2x3 inverse matrix
 */
export function invertAffine(M) {
  const a = M[0][0], b = M[0][1], tx = M[0][2];
  const c = M[1][0], d = M[1][1], ty = M[1][2];
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12) throw new Error('Singular affine matrix');
  const invDet = 1.0 / det;

  return [
    [ d * invDet, -b * invDet, (b * ty - d * tx) * invDet],
    [-c * invDet,  a * invDet, (c * tx - a * ty) * invDet],
  ];
}

/**
 * Apply affine transform to a point.
 * @param {number[][]} M - 2x3 matrix
 * @param {number} x
 * @param {number} y
 * @returns {number[]} [x', y']
 */
export function affinePoint(M, x, y) {
  return [
    M[0][0] * x + M[0][1] * y + M[0][2],
    M[1][0] * x + M[1][1] * y + M[1][2],
  ];
}

// ── Warp Affine (inverse mapping with bilinear interpolation) ──────

/**
 * Warp an image using an affine transform (same convention as cv2.warpAffine).
 * M is the FORWARD transform (src→dst). Internally we invert it to sample src.
 *
 * @param {Uint8ClampedArray} srcData - Source RGBA pixel data
 * @param {number} srcW - Source width
 * @param {number} srcH - Source height
 * @param {number[][]} M - 2x3 forward affine matrix
 * @param {number} dstW - Output width
 * @param {number} dstH - Output height
 * @returns {Uint8ClampedArray} Output RGBA pixel data
 */
export function warpAffine(srcData, srcW, srcH, M, dstW, dstH) {
  const Minv = invertAffine(M);
  const out = new Uint8ClampedArray(dstW * dstH * 4);

  for (let dy = 0; dy < dstH; dy++) {
    for (let dx = 0; dx < dstW; dx++) {
      // Map destination pixel back to source
      const sx = Minv[0][0] * dx + Minv[0][1] * dy + Minv[0][2];
      const sy = Minv[1][0] * dx + Minv[1][1] * dy + Minv[1][2];

      // Bilinear interpolation
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const x1 = x0 + 1, y1 = y0 + 1;
      const fx = sx - x0, fy = sy - y0;

      if (x0 < 0 || y0 < 0 || x1 >= srcW || y1 >= srcH) continue; // border = 0

      const w00 = (1 - fx) * (1 - fy);
      const w01 = fx * (1 - fy);
      const w10 = (1 - fx) * fy;
      const w11 = fx * fy;

      const i00 = (y0 * srcW + x0) * 4;
      const i01 = (y0 * srcW + x1) * 4;
      const i10 = (y1 * srcW + x0) * 4;
      const i11 = (y1 * srcW + x1) * 4;

      const oi = (dy * dstW + dx) * 4;
      for (let c = 0; c < 4; c++) {
        out[oi + c] = srcData[i00 + c] * w00 + srcData[i01 + c] * w01 +
                      srcData[i10 + c] * w10 + srcData[i11 + c] * w11;
      }
    }
  }
  return out;
}

/**
 * Warp only the alpha/mask channel (single channel float32).
 */
export function warpAffineMask(srcMask, srcW, srcH, M, dstW, dstH) {
  const Minv = invertAffine(M);
  const out = new Float32Array(dstW * dstH);

  for (let dy = 0; dy < dstH; dy++) {
    for (let dx = 0; dx < dstW; dx++) {
      const sx = Minv[0][0] * dx + Minv[0][1] * dy + Minv[0][2];
      const sy = Minv[1][0] * dx + Minv[1][1] * dy + Minv[1][2];

      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const x1 = x0 + 1, y1 = y0 + 1;
      const fx = sx - x0, fy = sy - y0;

      if (x0 < 0 || y0 < 0 || x1 >= srcW || y1 >= srcH) continue;

      out[dy * dstW + dx] =
        srcMask[y0 * srcW + x0] * (1-fx)*(1-fy) +
        srcMask[y0 * srcW + x1] * fx*(1-fy) +
        srcMask[y1 * srcW + x0] * (1-fx)*fy +
        srcMask[y1 * srcW + x1] * fx*fy;
    }
  }
  return out;
}

// ── Non-Maximum Suppression ────────────────────────────────────────

/**
 * NMS for axis-aligned bounding boxes.
 * @param {number[][]} boxes - Array of [x1, y1, x2, y2]
 * @param {number[]} scores
 * @param {number} threshold - IoU threshold
 * @returns {number[]} Indices of kept detections
 */
export function nms(boxes, scores, threshold) {
  const n = boxes.length;
  if (n === 0) return [];

  // Sort by score descending
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

// ── Vector operations ──────────────────────────────────────────────

/** L2 norm of a Float32Array. */
export function vecNorm(v) {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  return Math.sqrt(sum);
}

/** Normalize vector in-place. */
export function vecNormalize(v) {
  const n = vecNorm(v);
  if (n > 1e-12) for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

/** Matrix-vector multiply: (rows×cols) × (cols,) → (rows,) */
export function matVecMul(mat, vec, rows, cols) {
  const out = new Float32Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      sum += mat[i * cols + j] * vec[j];
    }
    out[i] = sum;
  }
  return out;
}
