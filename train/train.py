import json, math, os, sys, time, pathlib
import numpy as np
import torch
from model import GPT

root = pathlib.Path(__file__).resolve().parent.parent
out = root / "train" / "out"
out.mkdir(exist_ok=True)
CFG = dict(ctx=128, d=192, n_layer=4, n_head=4, dropout=0.2)
BATCH = 64
STEPS = int(os.environ.get("STEPS", 3200))
LR = 2e-3
WARM = 200
EVAL_EVERY = 200
torch.manual_seed(1337)
np.random.seed(1337)
torch.set_num_threads(int(os.environ.get("THREADS", 10)))

text = (root / "data" / "corpus.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_val = len(data) // 10
train_d, val_d = data[:-n_val], data[-n_val:]
print("chars", len(text), "vocab", len(chars), "train", len(train_d), "val", len(val_d), flush=True)

model = GPT(len(chars), CFG["ctx"], CFG["d"], CFG["n_layer"], CFG["n_head"], CFG["dropout"])
n_params = sum(p.numel() for p in model.parameters())
print("params", n_params, flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)


def batch(d):
    ix = torch.randint(len(d) - CFG["ctx"] - 1, (BATCH,))
    x = torch.stack([d[i:i + CFG["ctx"]] for i in ix])
    y = torch.stack([d[i + 1:i + CFG["ctx"] + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(d, iters=20):
    model.eval()
    g = torch.Generator().manual_seed(7)
    tot = 0.0
    for _ in range(iters):
        ix = torch.randint(len(d) - CFG["ctx"] - 1, (BATCH,), generator=g)
        x = torch.stack([d[i:i + CFG["ctx"]] for i in ix])
        y = torch.stack([d[i + 1:i + CFG["ctx"] + 1] for i in ix])
        tot += model(x, y)[1].item()
    model.train()
    return tot / iters


def lr_at(s):
    if s < WARM:
        return LR * (s + 1) / WARM
    p = (s - WARM) / max(1, STEPS - WARM)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


curve = []
t0 = time.time()
run = 0.0
for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0 or step == STEPS:
        tr, va = evaluate(train_d), evaluate(val_d)
        curve.append(dict(step=step, train=round(tr, 4), val=round(va, 4), sec=round(time.time() - t0, 1)))
        print(f"step {step} train {tr:.4f} val {va:.4f} t {time.time() - t0:.0f}s", flush=True)
        (out / "curve.json").write_text(json.dumps(curve))
        torch.save(model.state_dict(), out / "ckpt.pt")
        if va <= min(c["val"] for c in curve):
            torch.save(model.state_dict(), out / "best.pt")
    if step == STEPS:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = batch(train_d)
    _, loss, _ = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

total = time.time() - t0
meta = dict(params=n_params, steps=STEPS, batch=BATCH, lr=LR, seconds=round(total, 1), chars=chars,
            cfg=model.cfg, final_train=curve[-1]["train"], final_val=curve[-1]["val"],
            best_val=min(c["val"] for c in curve), train_chars=len(train_d), val_chars=len(val_d))
(out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
print("done", total, flush=True)
