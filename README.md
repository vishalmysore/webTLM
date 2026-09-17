# webLTM

A tiny (153K-parameter, 597KB) **recurrent-depth transformer**: instead of
writing out a chain of reasoning tokens, it re-runs a single transformer
block on its own latent state `r` times, then reads out an answer. Nothing
that happens inside those `r` iterations is ever decoded into text. It's a
small, fully open, from-scratch build of the mechanism behind what OpenAI
has called "opaque recurrence" in Astra (launched Sept 2026) — and behind
a growing line of academic work on recurrent-depth / looped-transformer
latent reasoning (Geiping et al.'s depth-recurrent transformers among them).

This sits in the same "small model, runs entirely on-device" line as
[webSLM](../webSLM), [webForecast](../webForecast) and
[recursiveMASWebLLM](../recursiveMASWebLLM) — same instinct (don't ship a
multi-GB model to prove a point about architecture), applied to a
different question: not *"can a small model run in the browser,"* but
*"what does it look like when a model's reasoning has no text
representation at all."*

**Live demos:** [vishalmysore.github.io/webTLM](https://vishalmysore.github.io/webTLM/)
— the webLTM demo above, plus a **comparison demo** ("Visible Thinking")
running a real reasoning model (DeepSeek-R1-Distill-Qwen-7B via WebLLM)
whose full chain-of-thought streams to the screen, unhidden, as the
opposite case study. Same task, opposite transparency properties, back to
back.
**Repo:** [github.com/vishalmysore/webTLM](https://github.com/vishalmysore/webTLM)
**Weights:** [huggingface.co/VishalMysore/webLTM](https://huggingface.co/VishalMysore/webLTM)
(`model.safetensors` + `config.json` + a standalone `modeling_webltm.py` —
see `hf/` in this repo for the export/upload scripts).

## Why addition

Multi-digit addition is the standard toy task in the recurrent-depth
literature for a reason: getting it right requires propagating carries
across positions, a genuinely iterative computation, so *some* problems
mechanically need more computational depth than others (a 5-digit sum can
need a carry chain up to 5 steps long; a 1-digit sum never does). That
makes it a clean way to ask "does giving the model more loops actually
buy it anything, and does it need more of them exactly when the problem
is harder?" — rather than a question you have to take on faith.

The model predicts every output digit **in parallel, in one shot** — there
is no autoregressive decoding loop, no scratchpad, and nothing resembling
a chain-of-thought string at any point. The *only* channel available to
the model for "thinking more" is looping the core block more times before
that single, simultaneous read-out.

## Architecture

```
tokens ──▶ embed ──▶ prelude (1×) ──▶ core (×r, looped) ──▶ coda (1×) ──▶ head ──▶ digits
                                        ▲        │
                                        └────────┘   (prelude output re-injected
                                                       every iteration)
```

- **prelude** — a standard pre-LN transformer block (multi-head attention +
  GELU MLP), run once, embeds the problem.
- **core** — the *same* block (shared weights), re-applied `r` times. At
  each step the original prelude output is added back in before the block
  runs (`h = core(h + e0)`), which is what keeps the recurrence stable and
  lets a single set of weights generalize to loop counts it never saw
  during training.
- **coda** — one more block, run once, whose output is the only thing ever
  projected to logits.

`r` is a runtime argument, not an architectural choice — it's supplied at
inference time, which is what lets `web/index.html`'s "thinking depth"
slider work with *no retraining*.

Trained with **random recurrence depth per step** (`r ~ Uniform(1, 8)`,
resampled every batch — the trick used in Geiping et al.'s Huginn), so the
weights aren't specialized to one fixed depth.

- vocab: `0-9, +, =, ANS` (13 tokens)
- `d_model=64`, 4 attention heads, 3 transformer blocks total (prelude,
  core, coda — core is the one that loops)
- sequence layout: 5-digit `A` + `+` + 5-digit `B` + `=` + 6 answer slots
  (an `ANS` placeholder token per slot; sum can have up to 6 digits)

## Results: does looping help, and does it help more on harder problems?

Exact-match accuracy, evaluated at various loop counts `r`, split out by
how many digits the operands have (see `eval/depth_hardness_grid.json`,
plotted interactively in the live demo):

| digits | r=1 | r=2 | r=4 | r=8 (max trained) | r=16 | r=24 |
|---|---|---|---|---|---|---|
| 1 | 100% | 100% | 100% | 100% | 97% | 74% |
| 2 | 85% | 92% | 93% | 92% | 79% | 62% |
| 3 | 75% | 81% | 84% | 82% | 67% | 51% |
| 4 | 63% | 67% | 69% | 70% | 55% | 29% |
| 5 | 56% | 59% | 62% | 61% | 45% | 26% |

Two things worth noting, both visible in the demo's chart:

1. **Harder problems need more loops to saturate.** 1–2 digit problems are
   already near their ceiling at `r=2`; 4–5 digit problems keep climbing
   through `r=8`. More digits, more carry chain, more benefit from extra
   recurrence — exactly the mechanistic story the architecture predicts.
2. **It doesn't extrapolate indefinitely.** Push `r` well past the trained
   range (16, 24) and accuracy falls off a cliff. This model was trained
   on `r ∈ [1, 8]`; asking it to loop 3× further than anything it saw
   drifts the latent state somewhere the coda was never trained to read.
   That's an honest limitation, not a hidden one — real recurrent-depth
   research (and presumably Astra, at a scale where nobody outside OpenAI
   can check) faces the same ceiling, just further out.

## Relationship to Astra

To be precise about scope: this is **not** a reproduction of OpenAI's
Astra. Astra is an undisclosed, frontier-scale system, and the actual
controversy around it — reported by TechCrunch and others in September
2026 — is that "opaque recurrence" removes chain-of-thought monitoring
*as a safety mechanism* at a scale where a lot could go wrong unseen.
webLTM is 153K parameters doing arithmetic. What it *does* share with
Astra is the mechanism, not the stakes: reasoning compute that lives in
a loop over latent state rather than in emitted tokens. Building it at a
scale small enough to read every line was the point — the opposite move
from "opaque," on purpose.

## Repo layout

```
webLTM/
├── train.py                    # model + training loop (PyTorch, CPU, ~8 min)
├── pack_weights.py             # trained weights → compact binary for the browser
├── verify_numpy.py             # numpy reference forward pass (for parity checking)
├── verify_node.js              # runs web/model.js under Node against the same cases
├── eval/
│   ├── weights.json            # full-precision exported weights (source of truth)
│   ├── depth_hardness_grid.json
│   ├── verify_numpy.json
│   └── verify_node.json        # identical to verify_numpy.json — confirms JS parity
├── web/                         # source for the webLTM demo — no build step
│   ├── index.html
│   ├── model.js                 # ~200 lines: embed/attention/layernorm/GELU/loop, from scratch
│   ├── manifest.json            # tensor shapes/offsets
│   ├── weights.bin               # 597KB binary weights (primary, any static host)
│   └── weights.b64.txt          # base64 fallback (hosts that only serve fixed content-types)
├── hf/                          # Hugging Face model repo export (safetensors + model card)
│   ├── export_hf.py
│   ├── push_to_hf.py
│   ├── modeling_webltm.py
│   ├── config.json
│   └── model.safetensors
└── docs/                        # what GitHub Pages actually serves (vishalmysore.github.io/webTLM)
    ├── index.html                # hub linking both demos
    ├── webltm/                   # copy of web/ — the opaque-recurrence demo
    └── monitor/                  # "Visible Thinking" — DeepSeek-R1-Distill-Qwen-7B via WebLLM,
                                   # full chain-of-thought streamed live, as the contrasting case
```

## Running it

**Retrain:**
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 train.py        # ~8 min on CPU, writes eval/weights.json + eval/depth_hardness_grid.json
python3 pack_weights.py # writes web/weights.bin + web/manifest.json
```

**Verify the JS port matches the trained weights exactly:**
```
python3 verify_numpy.py     # reference predictions, numpy
node verify_node.js         # same problems, run through web/model.js
diff eval/verify_numpy.json eval/verify_node.json   # → identical
```

**Run the demo locally:**
```
cd web && python3 -m http.server 8000
# open http://localhost:8000
```

**Deploy:** GitHub Pages is already configured to serve `docs/` on the
`main` branch (Settings → Pages → Source → Deploy from a branch → `main` /
`docs`) — live at [vishalmysore.github.io/webTLM](https://vishalmysore.github.io/webTLM/).
`docs/webltm/` is a synced copy of `web/`; if you change `web/`, copy it
back into `docs/webltm/` before pushing (both are plain static files, no
build step either way).

## The comparison demo ("Visible Thinking")

`docs/monitor/` runs **DeepSeek-R1-Distill-Qwen-7B** client-side via
[WebLLM](https://github.com/mlc-ai/web-llm) — a real, much larger reasoning
model that's trained to always emit a `<think>...</think>` block before
its answer. Unlike webLTM, nothing about its reasoning is architecturally
hidden: the demo streams every token, thinking included, straight to the
screen as it's generated. Point it at the same kind of arithmetic problem
webLTM solves silently and the contrast is direct: one model's "thinking
longer" is an invisible loop counter, the other's is a visibly growing
paragraph. It also makes the actual cost of that visibility concrete — the
auditable model is roughly 7,700× larger on disk than the opaque one.

This demo needs WebGPU and downloads ~4.5GB on first load (cached after);
it degrades to a clear inline message on unsupported browsers/hardware
rather than failing silently.
