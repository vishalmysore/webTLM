"""
Retrofit a real pretrained LLM (SmolLM2-135M) with a recurrent-depth core,
the same prelude/core/coda pattern as webLTM but on an actual English
language model instead of a from-scratch addition toy.

30 original decoder layers -> prelude (layers 0-9, frozen, unchanged) ->
core (ONE layer, initialized from a copy of the original layer 15, looped
r times, this is the only trainable part) -> coda (layers 20-29 + final
norm + lm_head, frozen, unchanged).

No KV cache: every generation step recomputes the full sequence. That's
deliberately simple and, crucially, correct -- HF's cache is keyed by a
fixed layer_idx per module, and calling the SAME core-layer module r times
in one forward pass would corrupt a cache built for "one call per layer".
Full recompute sidesteps that entirely, and at 135M params / short prompts
it's still fast enough for an interactive demo.
"""
import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
N_PRELUDE = 10
N_CODA = 10
CORE_INIT_FROM = 15  # which original layer's weights seed the shared core


class RecurrentDepthWrapper(nn.Module):
    def __init__(self, base=None):
        super().__init__()
        if base is None:
            base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        self.config = base.config
        base_layers = base.model.layers

        self.tok_emb = base.model.embed_tokens
        self.rotary_emb = base.model.rotary_emb
        self.norm = base.model.norm
        self.lm_head = base.lm_head

        self.prelude = nn.ModuleList([base_layers[i] for i in range(N_PRELUDE)])
        self.coda = nn.ModuleList(
            [base_layers[i] for i in range(len(base_layers) - N_CODA, len(base_layers))]
        )
        self.core = copy.deepcopy(base_layers[CORE_INIT_FROM])

        # freeze everything except the recurrent core
        for p in self.tok_emb.parameters():
            p.requires_grad_(False)
        for p in self.prelude.parameters():
            p.requires_grad_(False)
        for p in self.coda.parameters():
            p.requires_grad_(False)
        for p in self.norm.parameters():
            p.requires_grad_(False)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)
        for p in self.core.parameters():
            p.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.core.parameters() if p.requires_grad]

    def forward(self, input_ids, r):
        B, T = input_ids.shape
        device = input_ids.device
        inputs_embeds = self.tok_emb(input_ids)
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        h = inputs_embeds
        kwargs = dict(
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            use_cache=False,
        )
        for layer in self.prelude:
            h = layer(h, **kwargs)
        for _ in range(r):
            h = self.core(h, **kwargs)   # same weights, r times -- nothing here is a token
        for layer in self.coda:
            h = layer(h, **kwargs)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


@torch.no_grad()
def generate(model, tok, prompt, r, max_new_tokens=40, temperature=0.7, top_k=40, device="cpu"):
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(max_new_tokens):
        logits = model(ids, r=r)
        next_logits = logits[0, -1] / max(temperature, 1e-5)
        if top_k:
            v, ix = torch.topk(next_logits, top_k)
            probs = torch.softmax(v, dim=-1)
            next_id = ix[torch.multinomial(probs, 1)]
        else:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    import time
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    wrapper = RecurrentDepthWrapper()
    n_trainable = sum(p.numel() for p in wrapper.trainable_parameters())
    n_total = sum(p.numel() for p in wrapper.parameters())
    print(f"trainable params (core only): {n_trainable:,} / total {n_total:,}")

    t0 = time.time()
    text = generate(wrapper, tok, "The capital of France is", r=10, max_new_tokens=20)
    print(f"gen time {time.time()-t0:.2f}s")
    print(repr(text))
