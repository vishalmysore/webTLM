"""Reference forward pass (numpy, batch=1) used ONLY to check that model.js
reproduces the trained model correctly. Reads the same weights.json the
binary weights.bin/manifest.json were packed from."""
import json
import math
import numpy as np

with open("/home/claude/webLTM/eval/weights.json") as f:
    data = json.load(f)

W = {k: np.array(v, dtype=np.float32) for k, v in data["weights"].items()}
VOCAB = data["vocab"]
D = data["d"]
SEQ_LEN = data["seq_len"]
ANS_START = data["ans_start"]
D_MODEL = data["d_model"]
N_HEADS = data["n_heads"]
TOK2ID = {t: i for i, t in enumerate(VOCAB)}


def erf(x):
    return np.vectorize(math.erf)(x)


def gelu(x):
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))


def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def linear(x, w, b):
    return x @ w.T + b


def self_attention(x, prefix, n_heads):
    T, Dm = x.shape
    head_dim = Dm // n_heads
    qkv = linear(x, W[prefix + ".in_proj_weight"], W[prefix + ".in_proj_bias"])
    q, k, v = qkv[:, :Dm], qkv[:, Dm:2 * Dm], qkv[:, 2 * Dm:]
    out = np.zeros((T, Dm), dtype=np.float32)
    scale = 1 / math.sqrt(head_dim)
    for h in range(n_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
        scores = (qh @ kh.T) * scale
        scores = scores - scores.max(-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(-1, keepdims=True)
        out[:, sl] = p @ vh
    return linear(out, W[prefix + ".out_proj.weight"], W[prefix + ".out_proj.bias"])


def block(x, prefix, n_heads):
    h1 = layer_norm(x, W[prefix + ".ln1.weight"], W[prefix + ".ln1.bias"])
    a = self_attention(h1, prefix + ".attn", n_heads)
    x1 = x + a
    h2 = layer_norm(x1, W[prefix + ".ln2.weight"], W[prefix + ".ln2.bias"])
    m1 = gelu(linear(h2, W[prefix + ".mlp.0.weight"], W[prefix + ".mlp.0.bias"]))
    m2 = linear(m1, W[prefix + ".mlp.2.weight"], W[prefix + ".mlp.2.bias"])
    return x1 + m2


def tokenize(a, b):
    a_str = str(a).zfill(D)
    b_str = str(b).zfill(D)
    ids = [TOK2ID[c] for c in a_str] + [TOK2ID["+"]] + [TOK2ID[c] for c in b_str] + [TOK2ID["="]]
    ids += [TOK2ID["ANS"]] * (D + 1)
    return ids


def run(a, b, r):
    ids = tokenize(a, b)
    pos = np.arange(SEQ_LEN)
    e = W["tok_emb.weight"][ids] + W["pos_emb.weight"][pos]
    e0 = block(e, "prelude", N_HEADS)
    h = e0
    for _ in range(r):
        h = block(h + e0, "core", N_HEADS)
    h = block(h, "coda", N_HEADS)
    h = layer_norm(h, W["ln_f.weight"], W["ln_f.bias"])
    logits = linear(h[ANS_START:], W["head.weight"], W["head.bias"])
    digit_ids = logits.argmax(-1).tolist()
    s = "".join(VOCAB[i] for i in digit_ids)
    return s, logits.tolist()


if __name__ == "__main__":
    test_cases = [
        (5, 3), (12, 88), (999, 1), (4321, 5678), (99999, 99999),
        (10203, 4), (7, 70000), (55555, 44444), (100, 900), (30303, 7),
    ]
    r_values = [1, 4, 8]
    out = {}
    for a, b in test_cases:
        key = f"{a}+{b}"
        out[key] = {"true": a + b, "preds": {}}
        for r in r_values:
            s, _ = run(a, b, r)
            out[key]["preds"][str(r)] = s
    with open("/home/claude/webLTM/eval/verify_numpy.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
