import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, want_attn=False):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))
        return (x, att) if want_attn else (x, None)


class GPT(nn.Module):
    def __init__(self, vocab, ctx, d, n_layer, n_head, dropout=0.0):
        super().__init__()
        self.cfg = dict(vocab=vocab, ctx=ctx, d_model=d, n_layer=n_layer, n_head=n_head)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d, eps=1e-5)
        self.drop = nn.Dropout(dropout)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("out.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, want_attn=False):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        attns = []
        for b in self.blocks:
            x, a = b(x, want_attn)
            attns.append(a)
        x = self.ln_f(x)
        logits = x @ self.tok.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, attns
