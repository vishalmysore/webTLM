"""Reference PyTorch implementation matching model.safetensors exactly.
No custom ops, no trust_remote_code needed beyond this file — copy it
next to the weights and load with load_webltm() below.

    from modeling_webltm import load_webltm, tokenize, decode
    model, cfg = load_webltm("model.safetensors", "config.json")
    ids = tokenize(cfg, 4821, 367)
    digits = model(torch.tensor([ids]), r=8).argmax(-1)[0, cfg["ans_start"]:]
    print(decode(cfg, digits.tolist()))   # -> "005188"
"""
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class RecurrentDepthModel(nn.Module):
    """prelude -> core (looped r times, runtime-supplied) -> coda. r is not
    part of the architecture: it's an inference-time argument, which is
    what lets a single checkpoint answer at any thinking depth."""

    def __init__(self, vocab_size, seq_len, d_model=64, n_heads=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.prelude = Block(d_model, n_heads)
        self.core = Block(d_model, n_heads)
        self.coda = Block(d_model, n_heads)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, r):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        e0 = self.prelude(self.tok_emb(x) + self.pos_emb(pos))
        h = e0
        for _ in range(r):
            h = self.core(h + e0)   # nothing here is ever decoded to a token
        h = self.coda(h)
        return self.head(self.ln_f(h))


def load_webltm(weights_path="model.safetensors", config_path="config.json", device="cpu"):
    with open(config_path) as f:
        cfg = json.load(f)
    model = RecurrentDepthModel(len(cfg["vocab"]), cfg["seq_len"], cfg["d_model"], cfg["n_heads"])
    state = load_file(weights_path, device=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def tokenize(cfg, a, b):
    tok2id = {t: i for i, t in enumerate(cfg["vocab"])}
    d = cfg["digits_per_operand"]
    a_s, b_s = str(a).zfill(d), str(b).zfill(d)
    ids = [tok2id[c] for c in a_s] + [tok2id["+"]] + [tok2id[c] for c in b_s] + [tok2id["="]]
    ids += [tok2id["ANS"]] * (d + 1)
    return ids


def decode(cfg, digit_ids):
    return "".join(cfg["vocab"][i] for i in digit_ids)


if __name__ == "__main__":
    model, cfg = load_webltm()
    with torch.no_grad():
        for a, b, r in [(5, 3, 8), (4821, 367, 8), (99999, 99999, 8)]:
            ids = torch.tensor([tokenize(cfg, a, b)])
            logits = model(ids, r=r)
            digits = logits.argmax(-1)[0, cfg["ans_start"]:].tolist()
            print(f"{a} + {b} (r={r}) = {decode(cfg, digits)}  [true: {a+b}]")
