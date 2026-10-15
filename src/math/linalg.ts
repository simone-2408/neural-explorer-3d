/**
 * Dense linear algebra on flat, row-major Float64Arrays.
 *
 * A weight matrix with shape [rows, cols] stores element (r, c) at
 * index r * cols + c — the same memory layout as a contiguous PyTorch tensor.
 */

/** out = W·x + b, with W of shape [rows, cols]. */
export function matVecAdd(
  W: Float64Array,
  rows: number,
  cols: number,
  x: ArrayLike<number>,
  b: ArrayLike<number> | null,
  out: Float64Array,
): Float64Array {
  for (let r = 0; r < rows; r++) {
    let sum = b ? b[r] : 0;
    const offset = r * cols;
    for (let c = 0; c < cols; c++) sum += W[offset + c] * x[c];
    out[r] = sum;
  }
  return out;
}

/** Converts a row-major matrix to nested arrays (like `tensor.tolist()` in PyTorch). */
export function toNested(data: ArrayLike<number>, rows: number, cols: number): number[][] {
  const result: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = new Array(cols);
    for (let c = 0; c < cols; c++) row[c] = data[r * cols + c];
    result.push(row);
  }
  return result;
}
