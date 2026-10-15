/** Number formatting for the UI (true minus sign, no "−0.000", signed variants). */

export const MINUS = '−';

const isZeroString = (s: string) => /^-?0(\.0*)?$/.test(s);

export function formatValue(value: number, digits = 3): string {
  if (Number.isNaN(value)) return 'NaN';
  if (!Number.isFinite(value)) return value > 0 ? '+∞' : `${MINUS}∞`;
  // Tiny non-zero values would round to "0.000": show them in scientific notation instead.
  const tiny = value !== 0 && Math.abs(value) < 0.5 * Math.pow(10, -digits) && digits > 0;
  let s = Math.abs(value) >= 1e6 || tiny ? value.toExponential(2) : value.toFixed(digits);
  if (isZeroString(s)) s = s.replace('-', '');
  return s.replace('-', MINUS);
}

/** Like formatValue but always shows the sign of non-zero values: "+0.420", "−0.170". */
export function formatSigned(value: number, digits = 3): string {
  const s = formatValue(value, digits);
  return value > 0 && !isZeroString(s) ? `+${s}` : s;
}

export function formatPercent(value: number, digits = 1): string {
  return `${formatValue(value * 100, digits)}%`;
}

/** A probability as a percentage, with "<0.1%" instead of tiny numbers in scientific notation. */
export function formatProbability(p: number): string {
  if (p > 0 && p < 0.001) return '<0.1%';
  if (p < 1 && p > 0.999) return '>99.9%';
  return formatPercent(p, p < 0.1 || p > 0.99 ? 1 : 0);
}

export function formatCount(value: number): string {
  return value.toLocaleString('en-US');
}

/** Number for a TeX expression (TeX typesets "-" as a proper minus). */
export function texNumber(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return value > 0 ? '+\\infty' : value < 0 ? '-\\infty' : '\\text{NaN}';
  if (Math.abs(value) >= 1e6) {
    const [mantissa, exponent] = value.toExponential(2).split('e');
    return `${mantissa}\\times 10^{${Number(exponent)}}`;
  }
  const s = value.toFixed(digits);
  return isZeroString(s) ? s.replace('-', '') : s;
}

/** TeX number wrapped in parentheses when negative: (−0.170). */
export function texParen(value: number, digits = 3): string {
  const s = texNumber(value, digits);
  return s.startsWith('-') ? `\\left(${s}\\right)` : s;
}
