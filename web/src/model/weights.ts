import type { BlockWeights, Manifest, Weights } from "./types";

export function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  let v: number;
  if (e === 0) v = f * 2 ** -24;
  else if (e === 31) v = f ? NaN : Infinity;
  else v = (1 + f / 1024) * 2 ** (e - 15);
  return s ? -v : v;
}

function decode(buf: ArrayBuffer, dtype: Manifest["dtype"], offset: number, length: number): Float32Array {
  if (dtype === "float32") return new Float32Array(buf.slice(offset, offset + length * 4));
  const src = new Uint16Array(buf.slice(offset, offset + length * 2));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = halfToFloat(src[i]);
  return out;
}

export function parseWeights(manifest: Manifest, buf: ArrayBuffer): Weights {
  const map = new Map<string, Float32Array>();
  for (const t of manifest.tensors) {
    const expected = t.shape.reduce((a, b) => a * b, 1);
    if (expected !== t.length) throw new Error(`Tensor ${t.name}: shape and length disagree`);
    map.set(t.name, decode(buf, manifest.dtype, t.offset, t.length));
  }
  const get = (n: string): Float32Array => {
    const v = map.get(n);
    if (!v) throw new Error(`Missing tensor ${n}`);
    return v;
  };
  const blocks: BlockWeights[] = [];
  for (let i = 0; i < manifest.config.n_layer; i++) {
    const p = `blocks.${i}.`;
    blocks.push({
      ln1w: get(p + "ln1.weight"),
      ln1b: get(p + "ln1.bias"),
      qkvw: get(p + "qkv.weight"),
      qkvb: get(p + "qkv.bias"),
      projw: get(p + "proj.weight"),
      projb: get(p + "proj.bias"),
      ln2w: get(p + "ln2.weight"),
      ln2b: get(p + "ln2.bias"),
      fcw: get(p + "fc.weight"),
      fcb: get(p + "fc.bias"),
      outw: get(p + "out.weight"),
      outb: get(p + "out.bias"),
    });
  }
  return {
    tok: get("tok.weight"),
    pos: get("pos.weight"),
    blocks,
    lnfw: get("ln_f.weight"),
    lnfb: get("ln_f.bias"),
  };
}
