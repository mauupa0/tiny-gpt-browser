// Colours for the canvases. The CSS tokens in styles/tokens.css carry the same values;
// a unit test keeps the two in step.
export type Lch = readonly [number, number, number];

export const GROUND: Lch = [0.26, 0.085, 262];
export const GROUND_DEEP: Lch = [0.2, 0.07, 262];
export const PANEL: Lch = [0.295, 0.088, 262];
export const MARKER: Lch = [0.78, 0.14, 55];
export const COOL: Lch = [0.82, 0.085, 215];
export const INK: Lch = [0.95, 0.014, 95];
export const INK2: Lch = [0.82, 0.035, 250];
export const INK3: Lch = [0.72, 0.045, 250];

// Single-hue exposure ramp: lightness rises monotonically from the ground to paper.
export const RAMP: readonly Lch[] = [
  [0.34, 0.1, 264],
  [0.44, 0.13, 263],
  [0.55, 0.14, 260],
  [0.66, 0.12, 256],
  [0.76, 0.09, 250],
  [0.87, 0.05, 245],
  [0.96, 0.015, 240],
];

type Lab = [number, number, number];

function toLab([l, c, h]: Lch): Lab {
  const r = (h * Math.PI) / 180;
  return [l, c * Math.cos(r), c * Math.sin(r)];
}

function encode(u: number): number {
  const v = Math.min(1, Math.max(0, u));
  return v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055;
}

export function labToRgb([l, a, b]: Lab): [number, number, number] {
  const l_ = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m_ = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s_ = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_;
  const g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_;
  const bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_;
  return [Math.round(encode(r) * 255), Math.round(encode(g) * 255), Math.round(encode(bl) * 255)];
}

export function inGamut([l, a, b]: Lab): boolean {
  const l_ = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m_ = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s_ = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const ch = [
    4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
    -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
    -0.0041960863 * l_ - 0.7034186147 * m_ + 1.707614701 * s_,
  ];
  return ch.every((v) => v >= -0.0005 && v <= 1.0005);
}

export const lchInGamut = (c: Lch): boolean => inGamut(toLab(c));

export function lchToRgb(c: Lch): [number, number, number] {
  return labToRgb(toLab(c));
}

export function lchToCss(c: Lch): string {
  return `oklch(${c[0]} ${c[1]} ${c[2]})`;
}

function mix(a: Lab, b: Lab, t: number): Lab {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function sample(stops: readonly Lab[], t: number): Lab {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  return mix(stops[i], stops[i + 1], x - i);
}

// Little-endian ABGR words, ready for a Uint32Array view over canvas ImageData.
function pack(rgb: [number, number, number]): number {
  return ((255 << 24) | (rgb[2] << 16) | (rgb[1] << 8) | rgb[0]) >>> 0;
}

export function rampLut(size = 256): Uint32Array {
  const stops = RAMP.map(toLab);
  return Uint32Array.from({ length: size }, (_, i) => pack(labToRgb(sample(stops, i / (size - 1)))));
}

export function rampRgb(t: number): [number, number, number] {
  return labToRgb(sample(RAMP.map(toLab), t));
}

// Signed values: the cool pole for negatives, the marker for positives, the panel colour at zero.
export function divergingLut(size = 257): Uint32Array {
  const mid = toLab(PANEL);
  const neg = [mid, toLab([0.55, 0.06, 225]), toLab(COOL)];
  const pos = [mid, toLab([0.58, 0.1, 50]), toLab(MARKER)];
  return Uint32Array.from({ length: size }, (_, i) => {
    const v = (i / (size - 1)) * 2 - 1;
    return pack(labToRgb(v < 0 ? sample(neg, -v) : sample(pos, v)));
  });
}

export function rgbCss(rgb: [number, number, number], alpha = 1): string {
  return alpha === 1 ? `rgb(${rgb[0]} ${rgb[1]} ${rgb[2]})` : `rgb(${rgb[0]} ${rgb[1]} ${rgb[2]} / ${alpha})`;
}

// WCAG relative luminance contrast of two colours.
export function contrast(a: Lch, b: Lch): number {
  const lum = (c: Lch) => {
    const [r, g, bl] = lchToRgb(c).map((v) => {
      const s = v / 255;
      return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
  };
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

// CSS colour strings for a LUT built by rampLut or divergingLut, for canvas fillStyle.
export function lutToCss(lut: Uint32Array): string[] {
  return Array.from(lut, (w) => `rgb(${w & 255} ${(w >>> 8) & 255} ${(w >>> 16) & 255})`);
}

export const RAMP_LUT = rampLut(256);
export const RAMP_CSS = lutToCss(RAMP_LUT);
export const DIV_LUT = divergingLut(257);
export const DIV_CSS = lutToCss(DIV_LUT);

// Attention and activations are skewed to zero, so cells are coloured on a square-root scale.
export const rampIndex = (v: number): number => Math.round(Math.sqrt(Math.min(1, Math.max(0, v))) * 255);
// Signed value in [-1, 1] to an index of DIV_LUT.
export const divIndex = (v: number): number => Math.round((Math.min(1, Math.max(-1, v)) + 1) * 128);
