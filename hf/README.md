---
license: mit
tags:
  - recurrent-depth
  - looped-transformer
  - latent-reasoning
  - pytorch
  - from-scratch
language: en
library_name: pytorch
---

# webLTM — a 153K-parameter recurrent-depth transformer

Answers addition problems by looping a single transformer block over its
own latent state `r` times, then reading out every output digit at once.
There is no chain-of-thought text channel in this architecture — nothing
that happens inside the loop is ever decoded into a token. `r` is supplied
at inference time, not fixed by training, so one checkpoint answers at any
thinking depth.

This is a small, fully open, from-scratch build of the mechanism behind
what OpenAI called "opaque recurrence" in Astra (Sept 2026) and behind the
recurrent-depth / looped-transformer line of research — most directly
Geiping et al., ["Scaling up Test-Time Compute with Latent Reasoning: A
Recurrent Depth Approach"](https://arxiv.org/abs/2502.05171) (arXiv:2502.05171),
whose random-recurrence-depth training trick this model uses.

**Code + browser demo + full write-up:** see the `webLTM` GitHub repo
(train.py, eval scripts, and a live in-browser demo with an interactive
thinking-depth slider).

## Architecture

```
tokens → embed → prelude (1×) → core (×r, looped) → coda (1×) → head → digits
                                  ▲        │
                                  └────────┘  prelude output re-injected every iteration
```

- `prelude`, `core`, `coda` are pre-LN transformer blocks (multi-head
  attention + GELU MLP). `core` is applied `r` times with the prelude's
  output re-injected at every step (`h = core(h + e0)`), which keeps the
  latent state from drifting as `r` grows.
- `d_model=64`, 4 heads, 152,909 parameters total, 615KB in `float32`
  safetensors.
- Task: 5-digit + 5-digit addition, all output digits predicted
  **simultaneously** (non-autoregressive) — no scratchpad, no per-token
  decode loop.
- Trained with `r` sampled uniformly from `[1, 8]` on every step, so the
  weights aren't specialized to one fixed depth.

## Usage

```python
# pip install torch safetensors
from modeling_webltm import load_webltm, tokenize, decode

model, cfg = load_webltm("model.safetensors", "config.json")
ids = tokenize(cfg, 4821, 367)

import torch
with torch.no_grad():
    logits = model(torch.tensor([ids]), r=8)   # r = thinking depth
digits = logits.argmax(-1)[0, cfg["ans_start"]:].tolist()
print(decode(cfg, digits))   # -> "005188"
```

`modeling_webltm.py` (included in this repo) is the entire model
definition — no `trust_remote_code` framework dependency, just plain
`torch.nn`.

## Results: accuracy by loop depth and problem difficulty

Exact-match accuracy, evaluated on held-out problems (300 per cell):

| digits | r=1 | r=2 | r=4 | r=8 (max trained) | r=16 | r=24 |
|---|---|---|---|---|---|---|
| 1 | 100% | 100% | 100% | 100% | 97% | 74% |
| 2 | 85% | 92% | 93% | 92% | 79% | 62% |
| 3 | 75% | 81% | 84% | 82% | 67% | 51% |
| 4 | 63% | 67% | 69% | 70% | 55% | 29% |
| 5 | 56% | 59% | 62% | 61% | 45% | 26% |

Harder (more-digit) problems keep improving with more loops through the
trained range; pushed well past it (r=16, r=24) accuracy collapses — the
re-injection trick stabilizes the recurrence within the trained range, it
doesn't make the coda's read-out valid for latent states it never saw
during training. That ceiling is disclosed here deliberately, not hidden.

## Scope

Not a reproduction of OpenAI's Astra — Astra is an undisclosed,
frontier-scale system, and the actual controversy around it is about
chain-of-thought monitorability disappearing at a scale where a lot could
go wrong unseen. This is 153K parameters doing arithmetic, built at a
scale small enough to read every line of the loop — the opposite move
from "opaque," on purpose.

## Files

- `model.safetensors` — trained weights (float32, no pickle)
- `config.json` — architecture + tokenizer config
- `modeling_webltm.py` — full model definition (copy-paste runnable)
