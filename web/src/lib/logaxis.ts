// Position of a positive value on a log axis, 0 at `lo` and 1 at `hi`, clamped.
export function logPos(v: number, lo = 1e-7, hi = 10): number {
  if (!(v > 0)) return 0;
  const t = (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  return Math.min(1, Math.max(0, t));
}
