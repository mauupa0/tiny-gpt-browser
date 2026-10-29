"""Exports a checkpoint to flat binary weights (fp16 for the page, fp32 for tests)
plus reference logits and attention maps from PyTorch for the parity test."""
import json
import pathlib
import sys

import numpy as np
import torch

from model import GPT

root = pathlib.Path(__file__).resolve().parent.parent
out = root / "train" / "out"
ckpt = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
cfg = meta["cfg"]
chars = meta["chars"]
stoi = {c: i for i, c in enumerate(chars)}

model = GPT(cfg["vocab"], cfg["ctx"], cfg["d_model"], cfg["n_layer"], cfg["n_head"], 0.0)
model.load_state_dict(torch.load(out / ckpt, map_location="cpu"))
model.eval()

order = ["tok.weight", "pos.weight"]
for i in range(cfg["n_layer"]):
    for n in ["ln1", "qkv", "proj", "ln2", "fc", "out"]:
        order += [f"blocks.{i}.{n}.weight", f"blocks.{i}.{n}.bias"]
order += ["ln_f.weight", "ln_f.bias"]
sd = model.state_dict()
assert set(order) == set(sd.keys()), set(sd.keys()) ^ set(order)


def write(dtype, path_bin, path_json):
    np_dtype = np.float16 if dtype == "float16" else np.float32
    size = 2 if dtype == "float16" else 4
    tensors, chunks, offset = [], [], 0
    for name in order:
        arr = sd[name].detach().cpu().numpy().astype(np_dtype)
        tensors.append(dict(name=name, shape=list(arr.shape), offset=offset, length=int(arr.size)))
        chunks.append(arr.astype("<" + arr.dtype.str[1:]).tobytes())
        offset += arr.size * size
    path_bin.write_bytes(b"".join(chunks))
    manifest = dict(config=cfg, chars=chars, dtype=dtype, tensors=tensors, bytes=offset)
    path_json.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return offset


pub = root / "web" / "public" / "model"
fix = root / "web" / "tests" / "fixtures"
pub.mkdir(parents=True, exist_ok=True)
fix.mkdir(parents=True, exist_ok=True)
b16 = write("float16", pub / "weights.f16.bin", pub / "manifest.json")
b32 = write("float32", fix / "weights.f32.bin", fix / "manifest.f32.json")
print("fp16 bytes", b16, "fp32 bytes", b32)

corpus = (root / "data" / "corpus.txt").read_text(encoding="utf-8")
cases = {
    "short": "Litwo! Ojczyzno moja!",
    "verse": "Gdzie, przestępując miedzę, nie poznasz, że cudza!",
    "single": "A",
    "full_ctx": corpus[1000:1000 + cfg["ctx"]],
}


@torch.no_grad()
def run(m, text):
    ids = torch.tensor([[stoi[c] for c in text]])
    logits, _, attns = m(ids, want_attn=True)
    att = torch.stack([a[0] for a in attns])  # [L, H, T, T]
    return ids[0].tolist(), logits[0].numpy(), att.numpy()


def half_rounded():
    m = GPT(cfg["vocab"], cfg["ctx"], cfg["d_model"], cfg["n_layer"], cfg["n_head"], 0.0)
    m.load_state_dict({k: v.half().float() for k, v in sd.items()})
    m.eval()
    return m


def pack(m):
    res = {}
    for name, text in cases.items():
        ids, logits, att = run(m, text)
        entry = dict(text=text, ids=ids, logits=np.round(logits, 6).tolist())
        if name == "verse":
            entry["attn"] = np.round(att, 6).tolist()
        res[name] = entry
    return res


ref = dict(float32=pack(model), float16=pack(half_rounded()))
(fix / "reference.json").write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
print("reference written", (fix / "reference.json").stat().st_size, "bytes")

drift = 0.0
for name in cases:
    a = np.array(ref["float32"][name]["logits"])
    b = np.array(ref["float16"][name]["logits"])
    drift = max(drift, float(np.abs(a - b).max()))
curve = json.loads((out / "curve.json").read_text(encoding="utf-8"))
best = min(curve, key=lambda c: c["val"])
bench_path = root / "web" / "src" / "generated" / "bench.json"
stats = dict(
    params=meta["params"], steps=meta["steps"], batch=meta["batch"], ctx=cfg["ctx"],
    trainChars=meta["train_chars"], valChars=meta["val_chars"], vocab=cfg["vocab"],
    seconds=meta["seconds"], finalTrain=curve[-1]["train"], finalVal=curve[-1]["val"],
    bestVal=best["val"], bestTrain=best["train"], bestStep=best["step"], bestSec=best["sec"], curve=curve,
    fp16Bytes=b16, fp32Bytes=b32, fp16Drift=drift,
)
(root / "web" / "src" / "generated").mkdir(parents=True, exist_ok=True)
(root / "web" / "src" / "generated" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
print("stats written; fp16 vs fp32 max logit diff", drift)
