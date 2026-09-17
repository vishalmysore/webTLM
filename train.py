"""
webLTM -- a tiny recurrent-depth ("opaque recurrence") transformer.

Architecture: prelude -> core (looped r times in latent space, no decoded
intermediate tokens) -> coda -> output head.

Task: multi-digit addition, predicted NON-autoregressively (all answer
digits at once). This is deliberate: there is no chain-of-thought text
channel at all. The only thing that changes with more "thinking" is how
many times the core block iterates internally before the coda reads out
an answer. Nothing about those iterations is ever decoded to tokens --
which is the whole point of the demo (mirrors the "opaque recurrence"
framing used for OpenAI's Astra: reasoning that happens in latent space,
not as visible text).

Trained with random recurrence depth per step (Huginn-style), so a single
set of weights works at any test-time depth r >= 1, and harder instances
benefit from larger r without any extra visible tokens.
"""
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# Task / vocab
# ---------------------------------------------------------------------------
D = 5  # digits per operand -> answer has D+1 digits
DIGIT_TOKENS = [str(i) for i in range(10)]
SPECIAL = ["+", "=", "ANS"]
VOCAB = DIGIT_TOKENS + SPECIAL
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PLUS, EQ, ANS = TOK2ID["+"], TOK2ID["="], TOK2ID["ANS"]

SEQ_LEN = D + 1 + D + 1 + (D + 1)  # A + '+' + B + '=' + answer slots
ANS_START = D + 1 + D + 1  # index of first answer slot


def make_batch(batch_size, max_digits=D, device="cpu"):
    """Sample random addition problems with operand length in [1, max_digits]."""
    xs = torch.full((batch_size, SEQ_LEN), 0, dtype=torch.long)
    ys = torch.full((batch_size, D + 1), 0, dtype=torch.long)
    digit_lens = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        nd = random.randint(1, max_digits)
        a = random.randint(0, 10 ** nd - 1)
        b = random.randint(0, 10 ** nd - 1)
        s = a + b
        a_str = str(a).zfill(D)
        b_str = str(b).zfill(D)
        s_str = str(s).zfill(D + 1)
        seq = [TOK2ID[c] for c in a_str] + [PLUS] + [TOK2ID[c] for c in b_str] + [EQ] + [ANS] * (D + 1)
        xs[i] = torch.tensor(seq, dtype=torch.long)
        ys[i] = torch.tensor([TOK2ID[c] for c in s_str], dtype=torch.long)
        digit_lens[i] = nd
    return xs.to(device), ys.to(device), digit_lens.to(device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
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
    """prelude -> core (looped) -> coda. The loop count r is a *runtime*
    argument, not part of the architecture -- the same weights are reused
    for every iteration, and the initial embedding is re-injected at every
    step so the state doesn't drift as r grows (this is what lets the model
    generalize to more loops at test time than it ever saw in training)."""

    def __init__(self, vocab_size, seq_len, d_model=64, n_heads=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.prelude = Block(d_model, n_heads)
        self.core = Block(d_model, n_heads)
        self.coda = Block(d_model, n_heads)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, r):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        e = self.tok_emb(x) + self.pos_emb(pos)
        e0 = self.prelude(e)          # prelude runs once
        h = e0
        for _ in range(r):            # core loops r times -- this is the
            h = self.core(h + e0)     # "thinking" step; nothing here is ever
        h = self.coda(h)              # decoded to a token
        h = self.ln_f(h)
        return self.head(h)           # logits, read out only at the coda


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(steps=4000, batch_size=128, r_max=8, lr=3e-4, device="cpu"):
    model = RecurrentDepthModel(VOCAB_SIZE, SEQ_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    t0 = time.time()
    for step in range(steps):
        xs, ys, _ = make_batch(batch_size, device=device)
        r = random.randint(1, r_max)  # random recurrence depth per step
        logits = model(xs, r)
        ans_logits = logits[:, ANS_START:, :]
        loss = F.cross_entropy(ans_logits.reshape(-1, VOCAB_SIZE), ys.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            acc = eval_exact_match(model, r=r_max, n=200, max_digits=D, device=device)
            print(f"step {step:5d}  loss {loss.item():.4f}  r={r}  "
                  f"acc@r{r_max}={acc:.3f}  ({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def eval_exact_match(model, r, n=200, max_digits=D, device="cpu"):
    model.eval()
    xs, ys, _ = make_batch(n, max_digits=max_digits, device=device)
    logits = model(xs, r)[:, ANS_START:, :]
    pred = logits.argmax(-1)
    exact = (pred == ys).all(dim=1).float().mean().item()
    model.train()
    return exact


@torch.no_grad()
def depth_hardness_grid(model, r_values, digit_values, n=200, device="cpu"):
    model.eval()
    grid = {}
    for nd in digit_values:
        row = {}
        for r in r_values:
            row[r] = eval_exact_match(model, r=r, n=n, max_digits=nd, device=device)
        grid[nd] = row
    model.train()
    return grid


def export_weights(model, path):
    sd = model.state_dict()
    out = {"vocab": VOCAB, "seq_len": SEQ_LEN, "ans_start": ANS_START, "d": D,
           "d_model": model.d_model, "n_heads": 4, "weights": {}}
    for k, v in sd.items():
        out["weights"][k] = v.detach().cpu().numpy().astype("float32").tolist()
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}  ({sum(v.numel() for v in sd.values())} params)")


if __name__ == "__main__":
    device = "cpu"
    model = train(steps=4000, device=device)

    r_values = [1, 2, 4, 8, 16, 24]
    digit_values = [1, 2, 3, 4, 5]
    grid = depth_hardness_grid(model, r_values, digit_values, n=300, device=device)
    with open("/home/claude/webLTM/eval/depth_hardness_grid.json", "w") as f:
        json.dump(grid, f, indent=2)
    print(json.dumps(grid, indent=2))

    export_weights(model, "/home/claude/webLTM/eval/weights.json")
