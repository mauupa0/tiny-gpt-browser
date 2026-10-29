export interface ParamLedger {
  tokenEmbedding: number;
  positionEmbedding: number;
  attention: number;
  mlp: number;
  layerNorm: number;
  total: number;
}

// Parameter count of the architecture, from its shape alone. The output layer shares the
// token embedding table, so it adds nothing.
export function paramLedger(cfg: { vocab: number; ctx: number; d: number; layers: number }): ParamLedger {
  const { vocab, ctx, d, layers } = cfg;
  const tokenEmbedding = vocab * d;
  const positionEmbedding = ctx * d;
  const attention = layers * (3 * d * d + 3 * d + d * d + d);
  const mlp = layers * (4 * d * d + 4 * d + 4 * d * d + d);
  const layerNorm = layers * 4 * d + 2 * d;
  return { tokenEmbedding, positionEmbedding, attention, mlp, layerNorm, total: tokenEmbedding + positionEmbedding + attention + mlp + layerNorm };
}
