/**
 * Principal component analysis (top k components) by power iteration with
 * deflation. Uses the d×d covariance matrix or, when there are fewer points
 * than dimensions, the n×n Gram matrix (same non-zero eigenvalues).
 *
 * Used to project high-dimensional things to 2-D/3-D for display (training
 * trajectories, embeddings): the UI always says that it is a projection and
 * reports how much variance it keeps.
 */

export interface PCAResult {
  mean: Float64Array;
  /** Unit-length principal directions (d-dimensional). */
  components: Float64Array[];
  /** Variance along each component. */
  variances: number[];
  /** Total variance of the data (sum over all directions). */
  totalVariance: number;
}

function powerIteration(matrix: Float64Array, n: number, iterations = 300): { vector: Float64Array; value: number } {
  let v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = 1 + ((i * 7919) % 13) / 13; // deterministic start
  let value = 0;
  const next = new Float64Array(n);
  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < n; i++) {
      let s = 0;
      const row = i * n;
      for (let j = 0; j < n; j++) s += matrix[row + j] * v[j];
      next[i] = s;
    }
    let norm = 0;
    for (let i = 0; i < n; i++) norm += next[i] * next[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-300) return { vector: v, value: 0 };
    let change = 0;
    for (let i = 0; i < n; i++) {
      const x = next[i] / norm;
      change += Math.abs(x - v[i]);
      v[i] = x;
    }
    value = norm;
    if (change < 1e-12) break;
  }
  // Rayleigh quotient for the (signed) eigenvalue.
  let num = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += matrix[i * n + j] * v[j];
    num += v[i] * s;
  }
  value = num;
  v = Float64Array.from(v);
  return { vector: v, value };
}

/** Makes the sign of a direction deterministic (largest |component| positive). */
function canonicalSign(v: Float64Array): void {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (Math.abs(v[i]) > Math.abs(v[best])) best = i;
  if (v[best] < 0) for (let i = 0; i < v.length; i++) v[i] = -v[i];
}

export function pca(points: readonly ArrayLike<number>[], k: number): PCAResult {
  const n = points.length;
  const d = n ? points[0].length : 0;
  const mean = new Float64Array(d);
  for (const p of points) for (let j = 0; j < d; j++) mean[j] += p[j] / n;
  const X = points.map((p) => {
    const row = new Float64Array(d);
    for (let j = 0; j < d; j++) row[j] = p[j] - mean[j];
    return row;
  });
  let total = 0;
  for (const row of X) for (let j = 0; j < d; j++) total += row[j] * row[j];
  const denom = Math.max(1, n - 1);
  total /= denom;

  const components: Float64Array[] = [];
  const variances: number[] = [];
  const count = Math.min(k, d, Math.max(0, n - 1));
  if (count === 0) return { mean, components, variances, totalVariance: total };

  if (d <= n) {
    const C = new Float64Array(d * d);
    for (const row of X) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i * d + j] += (row[i] * row[j]) / denom;
    for (let c = 0; c < count; c++) {
      const { vector, value } = powerIteration(C, d);
      canonicalSign(vector);
      components.push(vector);
      variances.push(Math.max(0, value));
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i * d + j] -= value * vector[i] * vector[j];
    }
  } else {
    const G = new Float64Array(n * n);
    for (let a = 0; a < n; a++) {
      for (let b = a; b < n; b++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += X[a][j] * X[b][j];
        G[a * n + b] = s / denom;
        G[b * n + a] = s / denom;
      }
    }
    for (let c = 0; c < count; c++) {
      const { vector: u, value } = powerIteration(G, n);
      const v = new Float64Array(d);
      for (let a = 0; a < n; a++) for (let j = 0; j < d; j++) v[j] += u[a] * X[a][j];
      let norm = 0;
      for (let j = 0; j < d; j++) norm += v[j] * v[j];
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < d; j++) v[j] /= norm;
      canonicalSign(v);
      components.push(v);
      variances.push(Math.max(0, value));
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) G[a * n + b] -= value * u[a] * u[b];
    }
  }
  return { mean, components, variances, totalVariance: total };
}

/** Coordinates of `point` along the components (after subtracting the mean). */
export function project(point: ArrayLike<number>, result: PCAResult): number[] {
  return result.components.map((c) => {
    let s = 0;
    for (let j = 0; j < c.length; j++) s += (point[j] - result.mean[j]) * c[j];
    return s;
  });
}
