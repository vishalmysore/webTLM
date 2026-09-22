# What Happens When a Model Stops Showing Its Work

A few weeks ago I read a piece about Astra, a new reasoning technique reportedly in the works at OpenAI, and safety researchers weren't happy about it. The reporting called the underlying idea "opaque recurrence." The short version: instead of a model writing out its reasoning as text before answering, the extra thinking happens inside the network itself, in a loop, with nothing readable coming out of it. No transcript. No chain of thought to check. Just more compute spent somewhere you can't see.

It's worth asking why anyone would build this in the first place, because it's not just an opacity trick — there's a real efficiency case behind it. Writing reasoning out as plain English is slow and expensive: every token of chain-of-thought costs compute to generate, adds latency the user sits through, and forces the model's internal computation through the bottleneck of human vocabulary, even when the underlying "thinking" would naturally be continuous rather than word-shaped. Looping a block internally instead lets a model spend extra compute on a hard problem without growing its parameter count, its context window, or the number of tokens it has to generate and bill for. That's a genuine advantage — more reasoning per dollar and per millisecond, not a bigger model — and it's presumably a large part of why labs are exploring it. The catch, and the reason this piece exists, is that the same design choice that makes it efficient is what makes it unreadable.

That idea stuck with me, mostly because I couldn't picture it. "The model thinks in latent space instead of tokens" is one of those sentences that sounds precise and explains nothing. So I built three small demos to see it for myself — not to reproduce what OpenAI is doing (their model is undisclosed and enormous, mine run in a browser tab), but to get the actual mechanism in front of me, at a scale small enough to poke at.

This is a walkthrough of what I built and what it taught me about why "the model is reasoning, you just can't see it" is a bigger deal than it sounds.

## The reasoning arms race, quickly

Over the last couple of years, most of the progress in AI reasoning has come from giving models more time to think before they answer, rather than making them bigger. DeepSeek's R1 models, OpenAI's o-series, and a handful of others all do some version of the same thing: write out a long scratchpad — try this, no wait, check that, actually here's a cleaner way — and only then give a final answer. That scratchpad is usually visible to you, sometimes collapsed behind a "show thinking" toggle, but it's there as plain text if you want to read it.

That visibility turned out to be useful for more than curiosity. Safety and alignment teams lean on it. If a model is quietly planning to lie to you, or misreads the task, or takes a shortcut that technically satisfies the prompt but violates the spirit of it, a lot of the time you can catch that by reading its chain of thought before the final answer ever gets produced. It's not a perfect safeguard — a model could in principle reason one way and say another — but it's a real one, and it's cheap: you're not doing anything special, you're just reading what the model already wrote.

Opaque recurrence removes that scratchpad. The model still spends extra computation before answering, it just spends it in a loop of internal states rather than in words. Nothing is hidden from you on purpose in the sense of being deleted or filtered — it's that there was never anything in text form to hide. The architecture itself has no place to put a chain of thought.

That distinction — nothing to read vs. something being withheld — is subtle but it's the whole point, and it's exactly why I wanted a version of it I could actually click through.

None of this is exotic or secret research, either, which is worth saying plainly. Depth-recurrent transformers — looping a shared block instead of stacking new layers — have been public work for a while now. Jonas Geiping and colleagues published Huginn, a 3.5-billion-parameter depth-recurrent model, showing the trick scales well past toy size, months before any of this made headlines. And the worry about monitorability isn't just a vibe — a group at Redwood Research, including Ryan Greenblatt (one of the researchers quoted raising alarms about Astra), put an actual number on it. They call it "opaque serial depth": roughly, the longest chain of computation a model can run without passing through anything a person could read. Ordinary chain-of-thought models keep that number low no matter how big they get, because the readable bottleneck is built into how they work. Recurrent-depth architectures don't have that constraint, and the Redwood team's estimate is that this kind of architecture could push it up by orders of magnitude. That's the rigorous version of the thing these demos are trying to make you feel in your gut.

## Three demos, three points on the same line

### 1. webLTM — the part where nothing shows

The first demo is a 153,000-parameter model trained from scratch to do one thing: add two numbers together. Nothing fancy — small transformer, a "core" block in the middle, and instead of running that block once, it runs it a configurable number of times, r, before producing an answer. Turning up r gives the model more computation per problem, the same way a longer chain of thought would.

![Three demos landing page](images/01-hub.jpg)

Ask it to add 4821 and 367 and set the loop count to 8, and here's what you see: a status line that says "looping… 4 / 8," ticking up, and then a final number.

![webLTM mid-loop, showing only a counter](images/02-webltm-looping.jpg)

That's it. That's the whole visible surface of eight extra passes of computation. Whatever the model is doing with those loops — and it is doing something, accuracy actually depends on getting the loop count right for the problem size — none of it comes out as anything you could read. There's no channel for it. Then the answer lands.

![webLTM's finished answer, 5188, with no visible reasoning](images/03-webltm-answer.jpg)

This is the "opaque" half of opaque recurrence, made as literal as I could make it. It's not that the model is hiding something from you. It genuinely has nothing to show. The loop counter is the only artifact of eight passes of thinking.

To make that internal progression tangible, webLTM includes a step-by-step thinking enhancement: an intermediate latent probe that decodes the recurrent state (`coda -> layerNorm -> head`) at each loop iteration without interrupting the recurrent computation.

The clearest illustration is the cascade preset **4999 + 1 = 5000** (run at loop depth r = 4), which reveals how the model ripples carries across columns step by step:

- **Loop 0 (Prelude)**: Decodes to `018110` — the raw prelude embedding before any recurrent core iterations take place.
- **Loop 1**: Decodes to `005990` (`5990`) — the units addition resolves (9 + 1 = 10, leaving 0 and a carry), but the ripple hasn't propagated across the upper decimal places yet.
- **Loop 2**: Decodes to `005000` (`5000`) — the carry cascade propagates across three positions simultaneously (tens, hundreds, and thousands), resolving `5990` into the final sum `5000`.
- **Loop 3**: Decodes to `005000` (`5000`) — stable solved state.
- **Loop 4**: Decodes to `005090` (`5090`) — demonstrates the drift when recurrent depth extends past the problem's required depth.

![webLTM intermediate latent probe readout revealing carry propagation for 4999 + 1](images/03b-webltm-cascade.png)

Seeing that intermediate readout makes the mechanism concrete: the model isn't spinning empty loops, but executing a multi-step carry resolution depth by depth in latent space.


### 2. Visible Thinking — trained to show it, versus asked to show it

The second demo is where I ended up spending the most time, because a single comparison — opaque model vs. transparent model — undersells what's actually going on. There isn't just "shows reasoning" and "doesn't." There's a middle case that turns out to matter a lot: a model that shows reasoning only because you asked it to.

This page runs two real language models entirely in the browser via WebGPU, no server involved, and lets you pick between them. The first is DeepSeek-R1-Distill-Qwen-7B, which was trained to always wrap its reasoning in a block before answering. You don't prompt it to think step by step — it just does, every time, because that behavior is baked into the weights from training.

![Comparison table: webLTM, Phi-4-mini prompted, and R1-distill trained](images/04-monitor-table-deepseek.jpg)

The second option is Phi-4-mini-instruct, an ordinary instruction-tuned model with no reasoning training at all. Left alone, it would just answer. The only reason it shows any step-by-step reasoning here is one line in a system prompt asking it to think out loud before responding.

![Model picker switched to Phi-4-mini-instruct, badge explaining prompted-only reasoning](images/05-monitor-phi-badge.jpg)

Switch to it, ask the same arithmetic question, and it works — you get real, legible reasoning, arriving token by token, same as the trained model:

![Phi-4-mini-instruct reasoning live, digit by digit, in the browser](images/06-monitor-phi-reasoning.jpg)

But it's worth sitting with what's actually holding that visibility in place: one sentence in a system prompt. Nothing architectural stops that sentence from being dropped in some future version, some cost-saving change, some product decision that trims the system prompt to save a few hundred tokens per request. Take that line out and this exact model — same weights, same capability — goes straight to an answer with no visible reasoning at all, and you'd have no way to tell from the outside that anything had changed.

That's the part I think gets lost when people talk about model transparency as if it were a fixed property, like a spec sheet. For most of today's reasoning models, it's closer to a setting. A trained reasoner like R1-distill is the closer thing to a guarantee — the behavior is in the weights, not the prompt. But most of the models people actually deploy day to day are the Phi kind: ordinary models, made to look thoughtful with the right instructions, and just as capable of quietly not doing that anymore.

OpenAI's own Jakub Pachocki has said the company remains "committed to legible chains of thought" — and to be fair to him, the same reporting and OpenAI's own public statements don't hide the harder part either: monitorability tends to get worse as models get more capable and rely less on language tokens to do their thinking. Pachocki isn't claiming otherwise. So I don't read the Phi-4-mini toggle here as evidence that the commitment is insincere. What it shows is that the commitment is structurally fragile in a way trained-in behavior isn't: a policy, re-made every time a prompt is written or a product tradeoff gets made, not a property baked into the weights the way it is for a trained reasoner. Policies can hold for a long time. They can also erode one reasonable-sounding change at a time, without anyone deciding to break the promise — and that quieter failure mode is what the people doing the actual research on this are worried about.

### 3. Real Retrofit — proving the mechanism isn't a toy trick

The webLTM demo makes the opaque side of this easy to see, but it's a 153K-parameter model trained only to add numbers, which invites a fair objection: sure, but does the "loop instead of write" mechanism actually work on a real language model, or is that just a toy that happens to behave itself?

So the third demo takes an actual pretrained model — SmolLM2, 135 million parameters, trained on real text by Hugging Face — freezes almost all of it, and performs the same surgery: one layer in the middle gets replaced with a version that loops, and only that one layer gets fine-tuned, for about 26 minutes on a single CPU.

![Real Retrofit hero section describing the architecture surgery](images/07-retrofit-hero.jpg)

It's not a good model after that treatment, and I'd rather give you the actual numbers than wave at "rough English." After 500 steps, training loss went from 12.85 down to 4.85. Held-out loss on new text came in lowest at two loops — 4.42 — not at ten or twelve loops, which scored 4.88 and 5.05. So I want to be upfront about what that does and doesn't prove. It doesn't show that looping makes a model reason better, and it doesn't show this checkpoint would hold up against a same-size model trained the ordinary way on the same compute budget — I haven't run that comparison, and until someone does, the honest reading is that the loop worked mechanically without breaking the model, not that it paid off.

What it does show is narrower, and it's still the thing I actually went in looking for: a real pretrained transformer accepted the surgery, kept its language ability intact, loops a shared layer instead of running distinct ones, and generates actual, if rough, English:

![Real Retrofit generating text from a real, if rough, pretrained model](images/08-retrofit-generation.jpg)

That's the mechanical proof I was after, and it's worth being precise about what kind of proof it is. The opaque-loop pattern isn't a party trick that only works because the model was trained for it from scratch on a narrow task — it bolts onto a real, general-purpose language model without touching most of its weights. It says nothing about whether recurrence is a *good* way to spend compute at scale; that's a separate, much bigger question that a 26-minute CPU run on 135 million parameters can't answer either way. Scale the training budget and the base model up — which is exactly what a frontier lab would do — and I don't see an obvious architectural reason this stops working. Whether it stays *worth doing* at that scale is a different claim, and not one I can make from here.

## Why the middle case is the one to watch

Put the three demos side by side and the picture that emerges isn't "transparent AI vs. opaque AI." It's a spectrum, and most of what's actually shipping today sits in the middle, not at either end.

webLTM can't show its reasoning — there's no channel for it, full stop. R1-distill always does — it's trained in, not optional. Almost everything else, including a lot of what people currently trust for having "visible reasoning," is closer to Phi: capable of showing its work, doing so because of a prompt, and just as capable of not doing that the moment the prompt changes. If the industry's direction of travel is toward more opaque recurrence and less trained-in chain of thought — which is what prompted this whole detour for me — then the honest middle ground, the "we'll show you if we choose to," starts to look a lot less reassuring than it does today.

None of this is a claim about what OpenAI's actual system does internally — that's undisclosed, and it operates at a scale none of these demos come close to. What I wanted was something small enough to run in a browser tab and stare at directly, so that "the model reasons in a hidden loop instead of writing it out" stopped being an abstract sentence and became something I could watch happen, sum by sum, token by token.

## What would auditable recurrence even look like?

None of these three demos answer this, and I don't think anyone has a clean answer yet — but it's worth naming the actual open question instead of stopping at the worry. If a model is going to think in loops instead of words, the alternative to "give up on watching it" isn't necessarily "force it back into words." There's a middle path people are starting to poke at: exposing the model's intermediate state at fixed checkpoints inside the loop rather than the whole hidden trajectory; training a small, separate readout model whose only job is translating a slice of that latent state into something a person can sanity-check; or defining narrow, automatable verification conditions on what a loop produced, that don't require reading a transcript at all.

None of that is chain-of-thought monitoring in the form people are used to, and it probably catches less than a transcript would. But "catches less than a full transcript" and "catches nothing because there was never anything to read" are very different failure modes, and the gap between them is where I'd want to see more work go.

All three demos run client-side, no signup, no server — you can try them yourself, switch the model picker back and forth, and watch the badge change from "trained-in reasoning" to "prompted reasoning only" in real time.

- Live demos: [vishalmysore.github.io/webTLM](https://vishalmysore.github.io/webTLM/)
- Code and training notes, honest misses included: [github.com/vishalmysore/webTLM](https://github.com/vishalmysore/webTLM)
- The reporting that started this: [TechCrunch, September 2026](https://techcrunch.com/2026/09/02/openais-new-reasoning-technique-alarms-ai-safety-experts/)
- Prior research this builds on: [Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"](https://huggingface.co/papers/2502.05171) (the Huginn model), and Redwood Research's [operationalization of opaque serial depth](https://www.redwoodresearch.org/blog/an-operationalization-of-opaque-serial-depth)

---

**Disclaimer:** This is a personal piece, not a research paper and not a statement from my employer or anyone I work with. I'm an engineer who got curious about a headline, not an ML researcher or an AI safety specialist, and I built these three small demos myself, on my own time, to understand a concept I couldn't otherwise picture. The opinions above — what opaque recurrence means, why the "prompted vs. trained" distinction matters, what I think auditable recurrence might look like — are my own, shaped by that hands-on experimentation and by a lot of reading and arguing in public spaces: Reddit threads, Hacker News comment sections, LessWrong, and AI-safety discussions on X, where people who actually do this research for a living go back and forth on it in far more depth than I have here. I've tried to be precise about what my demos do and don't prove, and to credit the real research rather than imply I arrived at any of this independently — but this should be read as one practitioner's attempt to make an abstract worry concrete and legible, not as an authoritative or final word on where AI reasoning transparency is headed. Corrections and disagreement are welcome.
