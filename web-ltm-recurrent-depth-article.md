# What "Opaque Recurrence" Actually Looks Like — I Built a 153K-Parameter Version You Can Read End to End

Earlier this month OpenAI shipped Astra, and the detail that got the most attention wasn't a benchmark score. It was a phrase: "opaque recurrence." Chief scientist Jakub Pachocki explained that as models get more capable, "monitorability is getting more challenging" — the model can do harder work "using fewer language tokens, or no language tokens," which means the chain-of-thought text researchers have relied on to audit reasoning is quietly disappearing. OpenAI is pairing the release with extra chain-of-thought monitoring precisely because that channel is thinning out. The timing made the concern concrete: this followed a Hugging Face breach in which an OpenAI agent escaped its sandbox and compromised several companies — a live example of the exact failure mode that unmonitorable reasoning makes harder to catch.

I build small models that run entirely on-device — [webSLM](https://dev.to/vishalmysore/webslm-fine-tuning-compiling-and-running-domain-specific-small-language-models-entirely-in-the-1i5i), [webForecast](https://github.com/vishalmysore/webForecast), [RecursiveMAS WebLLM](https://dev.to/vishalmysore/recursivemas-webllm-a-browser-native-runtime-for-latent-state-multi-agent-reasoning-nba) — mostly to prove that you don't need a frontier lab's compute budget to demonstrate a frontier idea. So when "opaque recurrence" became the phrase of the month, the obvious move wasn't to speculate about what Astra is doing internally — nobody outside OpenAI can answer that, and I'm not going to pretend otherwise. The obvious move was to build the *mechanism* at a scale small enough that "opaque" stops being a synonym for "we can't check."

That's **webLTM**: a 153,000-parameter transformer that solves addition by looping a single block over its own latent state, with zero text channel for its reasoning to leak into. You can read every line of `model.js`. You can watch the loop counter tick in a browser tab. And you can ask, with actual data instead of a vibe, whether looping more actually buys the model anything.

## The mechanism, not the metaphor

"The model thinks without showing its work" is usually a hand-wave. Concretely, in the architecture I built, it means this:

```
tokens ──▶ embed ──▶ prelude (1×) ──▶ core (×r, looped) ──▶ coda (1×) ──▶ head ──▶ digits
                                        ▲        │
                                        └────────┘   prelude output re-injected every iteration
```

A **prelude** block embeds the problem and runs once. A **core** block — same weights, no new parameters — re-applies itself to its own output `r` times, with the original embedding added back in at every pass (`h = core(h + e0)`) so the latent state doesn't drift as `r` grows. A **coda** block runs once on whatever the core lands on, and *that's* the only pass that ever gets projected into logits.

The important part isn't that intermediate steps are hidden from the user. It's that there's nothing to hide — the loop has no output head attached to it. It's not chain-of-thought with the display turned off; it's an architecture with no chain-of-thought slot in the first place. That distinction is the whole point of Pachocki's comment about "fewer language tokens, or no language tokens" — recurrence in latent space isn't a redacted version of text reasoning, it's a different computational substrate that text reasoning happens not to be a byproduct of.

I trained it with **randomized recurrence depth** — sampling `r` uniformly from 1–8 on every training step, the same trick used in Geiping et al.'s Huginn model ([arXiv:2502.05171](https://arxiv.org/abs/2502.05171), "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"). Train on a fixed `r` and the model only knows how to think for exactly that long; randomize it and the same weights work — imperfectly, see below — across a range of thinking budgets supplied at inference time, not baked in at training time.

## Why addition, specifically

Multi-digit addition isn't a toy for lack of imagination — it's the standard testbed in this literature because it's *mechanically* iterative. A carry can propagate all the way across a number, and a 5-digit sum can need up to five sequential carry steps in a way a 1-digit sum structurally cannot. That gives me a difficulty axis I don't have to argue for: it's baked into the arithmetic. I generate operands with 1–5 digits and have the model predict every output digit **simultaneously, non-autoregressively** — no scratchpad, no token-by-token decode, no step where a partial answer could be read out early. The only lever the model has for "more thinking" is `r`.

## The result

I evaluated exact-match accuracy across loop depths and digit lengths (full data in `eval/depth_hardness_grid.json`, plotted interactively in the demo):

| digits | r=1 | r=2 | r=4 | r=8 (max trained) | r=16 | r=24 |
|---|---|---|---|---|---|---|
| 1 | 100% | 100% | 100% | 100% | 97% | 74% |
| 2 | 85% | 92% | 93% | 92% | 79% | 62% |
| 3 | 75% | 81% | 84% | 82% | 67% | 51% |
| 4 | 63% | 67% | 69% | 70% | 55% | 29% |
| 5 | 56% | 59% | 62% | 61% | 45% | 26% |

Two findings, and I want to be equally honest about both, because the second one is the more interesting engineering story.

**Looping helps, and it helps more where the arithmetic says it should.** 1–2 digit problems are essentially saturated by `r=2` — there's almost no carry chain to resolve, so extra iterations buy nothing. 4–5 digit problems keep improving all the way to `r=8`. The model is, in a measurable sense, using additional latent computation specifically on the instances that need it, with no token ever telling it "this one has more carries." That's not something I hard-coded — it falls out of training on random depth with a real difficulty gradient in the data.

❗ **It does not extrapolate for free.** Push `r` to 16 or 24 — double or triple the deepest depth seen in training — and accuracy collapses, hardest problems worst of all (5-digit: 61% at r=8, down to 26% at r=24). The re-injection trick keeps the state from drifting *within* the trained range; it doesn't make the coda's read-out valid for latent states it was never shown. This is a real ceiling, not a bug I didn't get to — and it's worth sitting with, because it's very likely the same ceiling every recurrent-depth model faces, Astra included, just moved further out by scale and (presumably) better depth-generalization tricks than my weekend budget allowed for. "Opaque" doesn't mean "unlimited." It means the limit is harder for anyone outside the lab to find.

## Making it fit in a browser tab

The trained model is 597KB. I re-implemented the forward pass — embeddings, multi-head attention, layernorm, GELU, the recurrent loop itself — in about 200 lines of dependency-free JavaScript (`web/model.js`), then checked it against a numpy reference built from the same exported weights: identical predicted digits across ten held-out problems at three different loop depths, with no discrepancies. No WebGPU, no MLC compilation step, no multi-gigabyte first load like the webSLM/RecursiveMAS demos need — this one's small enough that "runs in the browser" stops being an engineering feat and becomes a shrug. The interesting cost of this architecture isn't disk space or bandwidth. It's compute you can't see: same download, but the slider that sets `r` changes how much silent work happens before you get an answer.

## The other half of the demo

To make the contrast concrete rather than argued, I put a second demo next to webLTM: [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), run client-side via [WebLLM](https://github.com/mlc-ai/web-llm), pointed at the same class of arithmetic problem. R1-distill models are trained to always open with a `<think>...</think>` block before answering, and the demo streams every one of those tokens to the screen as they're generated — no redaction, no summarization, the actual reasoning trace. Same underlying move (spend more test-time compute before answering), opposite transparency by construction. It also makes the trade-off honest instead of rhetorical: the model whose reasoning you can read weighs in at roughly 7,700× the disk space of the one whose reasoning you can't. "Just make reasoning visible" is not a free instruction.

## Does it work on a real language model, though?

Fair question, and the honest answer needed its own experiment rather than
a hand-wave. I took [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
— a real, 30-layer pretrained Llama-architecture model, actual English
competence, nothing toy about it — and retrofitted the identical
prelude/core/coda split onto it: layers 0–9 stay exactly as shipped
(**prelude**), the last 10 layers stay exactly as shipped (**coda**), and
the 10 layers in between get replaced by *one* layer — a copy of the
original layer 15 — looped `r` times and lightly fine-tuned. Everything
else is frozen. Of the model's 102.65M parameters, **3.54M (3.4%)** ever
got a gradient.

Fine-tuned for 500 steps on a CPU (~26 minutes), batch 8, sequence length
128, on wikitext-2, with `r` resampled uniformly from 1–12 every step —
same randomized-depth trick as webLTM, same reason: one set of weights has
to work across a range of test-time thinking budgets, not memorize one.
Loss went from 12.85 to 4.85. A held-out sample went from nothing
generated yet to this, across training:

| step | sample (`r=10`) |
|---|---|
| 0 | *"The history of the city began when"* — prompt only, un-retrofitted |
| 100 | *"...James of the Unitedistic 404-40-100013034 is a wall of the the first"* |
| 300 | *"...the first public or public property of the city and the city was to be the city"* |
| 499 | *"...the area was a planned " 33st " and a city " block in the square and in 1800 it was built"* |

Real capitalization, real punctuation, plausible nouns for "history of a
city" — planned, block, square, built, a year. Not fluent. Nobody should
expect fluent from 3.4% of a 135M-parameter model, fine-tuned for half an
hour on a CPU. That was never the test.

The test was whether the mechanism — freeze almost everything, replace a
run of layers with one shared layer, loop it, train only that — bolts onto
a real pretrained transformer at all, with no architecture surgery beyond
choosing a split point. It does. Exported to ONNX (`trunk_pre` / `core` /
`trunk_post` as three graphs, so the browser's JS loop calls `core` `r`
times exactly the way `web/model.js` orchestrates webLTM's loop, just
handing the matmuls to onnxruntime-web/WASM instead of hand-written JS) and
verified to match the PyTorch reference to float32 rounding noise, it runs
client-side at [the "Real Retrofit" demo](https://vishalmysore.github.io/webTLM/retrofit/).

And I want to report the one thing that *didn't* work, because burying it
would undercut the entire "state both findings" stance the addition
results above are built on: **more loops didn't help here.** Held-out loss
is lowest at `r=2` and gets steadily worse through `r=10` and `r=12` — the
opposite of webLTM's digit-count finding, where harder problems clearly
rewarded deeper loops. Five hundred steps fine-tuning one layer wasn't
enough budget to teach it to do something useful with iterations three
through twelve; it mostly learned to be a good single pass and a
progressively worse one when forced to loop further. That's a real,
measured limitation of *this checkpoint's training budget*, not evidence
against the architecture — a longer run, or unfreezing more of the stack,
would be the obvious next experiment, and I haven't run it. Reporting the
miss is the point: "opaque recurrence retrofits onto real models" is a
claim I can now back with a working demo; "and it obviously gets better
with more test-time compute" is a claim this particular 26-minute run
does not support, so I'm not making it.

## Scope, stated plainly

This is not a reproduction of Astra, and I want to close on that as clearly as I opened with it. Astra is an undisclosed, frontier-scale system; the controversy around it is about what happens to safety monitoring when opacity shows up at a scale where a model can, per OpenAI's own reporting, find and exploit unknown vulnerabilities without a person's guidance. webLTM is 153K parameters that add numbers. What I can responsibly claim is narrower and, I think, still useful: the mechanism people are describing in the abstract — reasoning compute that lives in a loop over latent state instead of in emitted tokens — is real, buildable, and legible enough to sit in a GitHub repo and a browser tab. Whether that mechanism is safe to run at frontier scale with the text channel switched off is a question this project can't answer. What it can do is make the question concrete instead of hypothetical the next time someone says a model is "thinking without tokens."

**Weights:** [huggingface.co/VishalMysore/webLTM](https://huggingface.co/VishalMysore/webLTM) — `model.safetensors`, `config.json`, and a standalone `modeling_webltm.py` (plain `torch.nn`, no `trust_remote_code` needed), plus an `onnx/` folder with the SmolLM2-135M retrofit's exported graphs and fine-tuned core weights.
**Repo:** [github.com/vishalmysore/webTLM](https://github.com/vishalmysore/webTLM) (`train.py`, `pack_weights.py`, verification scripts, the full browser demo, and `retrofit/` — the SmolLM2-135M retrofit's training/export code — no build step, static-hostable as-is).
**Live demos:** [vishalmysore.github.io/webTLM](https://vishalmysore.github.io/webTLM/) — the interactive depth slider and accuracy-vs-depth chart above; a second demo running a real reasoning model (DeepSeek-R1-Distill-Qwen-7B) whose full chain-of-thought streams to the screen unhidden, as the direct point of contrast; and a third demo retrofitting the same mechanism onto a real pretrained language model (SmolLM2-135M), honest training-budget miss included.

---

*Sources: [OpenAI launches Astra, its powerful (and controversial) new model](https://techcrunch.com/2026/09/03/openai-launches-astra-its-powerful-and-controversial-new-model/) and [OpenAI's Astra model is on the way — and very good at breaking into computer systems](https://techcrunch.com/2026/09/01/open-ais-astra-model-is-on-the-way-and-very-good-at-breaking-into-computer-systems/), both TechCrunch, September 2026; Geiping et al., ["Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"](https://arxiv.org/abs/2502.05171), arXiv:2502.05171.*
