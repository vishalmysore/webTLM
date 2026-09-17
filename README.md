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
— the webLTM demo above, a **comparison demo** ("Visible Thinking") running
a real reasoning model (DeepSeek-R1-Distill-Qwen-7B via WebLLM) whose full
chain-of-thought streams to the screen, unhidden, as the opposite case
study, and a third demo ("Real Retrofit") that bolts the same
prelude/core/coda mechanism onto an actual pretrained language model
(SmolLM2-135M) instead of a from-scratch toy. Same mechanism, three angles.
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

## The retrofit demo ("Real Retrofit")

`docs/retrofit/` answers the obvious objection to webLTM: *"sure, but that's
153K parameters that add numbers — does this mechanism actually work on a
real language model?"* It retrofits the same prelude/core/coda pattern onto
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
a real 30-layer pretrained Llama-architecture model:

- layers 0–9 (**prelude**) and the last 10 layers (**coda**) are the base
  model's own weights, frozen, run once each, exactly as SmolLM2 shipped;
- one layer (a deep-copy of the original layer 15) becomes the **core**,
  looped `r` times per forward pass — the only part with gradients;
- fine-tuned for 500 steps on CPU, batch 8, sequence length 128, on
  [wikitext-2-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext),
  with `r` sampled uniformly from 1–12 every step (same randomized-depth
  trick as webLTM);
- **3.54M of the model's 102.65M parameters (3.4%) were ever updated.**

Training loss went from 12.85 to 4.85 in ~26 minutes; held-out generations
went from `"The history of the city began when"` (nothing, base weights) to
`"...the area was a planned " 33st " and a city " block in the square and
in 1800 it was built"` — real English syntax and plausible vocabulary, not
fluent prose. Stated plainly, one finding did **not** replicate: unlike
webLTM's addition task, held-out loss here is *lowest* at `r=2` and gets
worse through `r=10`/`r=12` — 500 steps fine-tuning one layer wasn't enough
budget to teach it to make good use of the deeper loops it was trained
across. Full loss table and sample progression in `docs/retrofit/index.html`.

No KV-cache (same reasoning as webLTM: the same core-layer module gets
called `r` times per forward pass, which would corrupt a cache keyed by a
fixed `layer_idx`), so every generated token recomputes the full sequence —
this is inherently a research/demo checkpoint, not a chatbot, and the demo
says so.

**Export:** `retrofit/export_onnx.py` splits the fine-tuned wrapper into
three ONNX graphs — `trunk_pre.onnx` (embed output → after prelude),
`core.onnx` (one layer, called `r` times by the browser's JS loop —
mirrors webLTM's own JS-orchestrated loop, just delegating the matmuls to
onnxruntime-web/WASM), and `trunk_post.onnx` (after coda + final norm) —
plus a raw `embed_fp16.bin` (the tied embedding/lm_head weight, used
directly by JS for the embedding gather and the final logits projection,
so that tied 28M-parameter matrix is never duplicated inside an ONNX
graph) and a precomputed `rope_table.json` (RoPE cos/sin for positions
0–255, since RoPE isn't retraced into the graphs — the browser just slices
rows). `retrofit/verify_onnx.py` confirms the ONNX pipeline's output
matches the PyTorch reference forward pass exactly (max abs diff ≈ 7e-5,
float32 rounding noise).

Because `trunk_pre.onnx` and `trunk_post.onnx` are ~142MB each — over
GitHub's 100MB hard per-file push limit — the `onnx_export/` assets are
**not** committed to this repo; they're pushed to the `onnx/` folder of the
[HF weights repo](https://huggingface.co/VishalMysore/webLTM) instead, and
`docs/retrofit/model.js` fetches them from there at runtime (same pattern
`@huggingface/transformers` and `onnxruntime-web` use generally). See
`retrofit/README_UPLOAD.md` for the upload command.

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
├── retrofit/                     # SmolLM2-135M recurrent-depth retrofit (training + export)
│   ├── model.py                  # RecurrentDepthWrapper: prelude/core/coda split of a real Llama-arch model
│   ├── prepare_data.py           # tokenizes wikitext-2-raw-v1 into fixed-length chunks (data.pt)
│   ├── train_retrofit.py         # fine-tunes only the core layer, random depth r∈[1,12], 500 steps CPU
│   ├── export_onnx.py            # exports trunk_pre/core/trunk_post ONNX graphs + fp16 embed + RoPE table
│   ├── quantize_fp16.py          # (best-effort) fp16 weight conversion for the ONNX graphs
│   ├── verify_onnx.py            # confirms ONNX pipeline output matches the PyTorch reference exactly
│   ├── core_final.pt             # fine-tuned core layer weights (14MB)
│   └── onnx_export/              # trunk_pre.onnx, core.onnx, trunk_post.onnx, embed_fp16.bin,
│                                  # rope_table.json, meta.json — pushed to the HF repo's onnx/ folder,
│                                  # NOT committed to git (141MB+141MB single files exceed GitHub's limit)
└── docs/                        # what GitHub Pages actually serves (vishalmysore.github.io/webTLM)
    ├── index.html                # hub linking all three demos
    ├── webltm/                   # copy of web/ — the opaque-recurrence demo
    ├── monitor/                  # "Visible Thinking" — DeepSeek-R1-Distill-Qwen-7B via WebLLM,
    │                              # full chain-of-thought streamed live, as the contrasting case
    └── retrofit/                 # "Real Retrofit" — SmolLM2-135M recurrent-depth retrofit,
                                   # runs client-side via onnxruntime-web (WASM), weights fetched from HF
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

**Deploy:** `.github/workflows/deploy-pages.yml` builds and deploys `docs/`
via GitHub Actions (Settings → Pages → Source → GitHub Actions) on every
push that touches `docs/**`, or on manual `workflow_dispatch` — live at
[vishalmysore.github.io/webTLM](https://vishalmysore.github.io/webTLM/).
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
