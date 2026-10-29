export interface ModelConfig {
  vocab: number;
  ctx: number;
  d_model: number;
  n_layer: number;
  n_head: number;
}

export interface TensorEntry {
  name: string;
  shape: number[];
  offset: number;
  length: number;
}

export interface Manifest {
  config: ModelConfig;
  chars: string[];
  dtype: "float16" | "float32";
  tensors: TensorEntry[];
  bytes: number;
}

export interface BlockWeights {
  ln1w: Float32Array;
  ln1b: Float32Array;
  qkvw: Float32Array;
  qkvb: Float32Array;
  projw: Float32Array;
  projb: Float32Array;
  ln2w: Float32Array;
  ln2b: Float32Array;
  fcw: Float32Array;
  fcb: Float32Array;
  outw: Float32Array;
  outb: Float32Array;
}

export interface Weights {
  tok: Float32Array;
  pos: Float32Array;
  blocks: BlockWeights[];
  lnfw: Float32Array;
  lnfb: Float32Array;
}

export type Fault = "none" | "scale" | "mask";
