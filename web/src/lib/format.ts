export const int = (n: number): string => Math.round(n).toLocaleString("en-US");

export function pct(p: number, digits = 0): string {
  if (p > 0 && p < 0.5 / 10 ** (digits + 2)) return `<${(0.5 / 10 ** digits).toFixed(digits)}%`;
  return `${(p * 100).toFixed(digits)}%`;
}

export const sci = (x: number): string => x.toExponential(1);

// Plain hyphen on purpose: the typographic minus is easy to mistake for a dash.
export function signed(x: number, digits = 2): string {
  const s = Math.abs(x).toFixed(digits);
  return x < 0 && Number(s) !== 0 ? `-${s}` : `+${s}`;
}

export const mb = (bytes: number): string => (bytes / 1e6).toFixed(1);

export function duration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
