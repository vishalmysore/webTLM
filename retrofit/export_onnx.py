"""
Export the fine-tuned recurrent-depth retrofit to a browser-runnable form:
three ONNX graphs (trunk_pre / core / trunk_post) that operate purely on
hidden_states + a precomputed causal mask + precomputed RoPE cos/sin, plus
one raw fp16 weight file for the tied embedding/lm_head (used directly by
JS for the embedding gather and the final logits projection, so that tied
weight is never duplicated inside an ONNX graph).

Split rationale (mirrors webLTM's original prelude/core/coda JS orchestration,
just delegating the matmuls to onnxruntime-web instead of hand-written JS):
  - trunk_pre.onnx : embed_output -> after 10 prelude layers
  - core.onnx      : one shared layer, the JS caller loops it r times
  - trunk_post.onnx: after 10 coda layers + final RMSNorm (no lm_head --
                     that's a JS-side matmul against the same tied weight
                     used for embedding, done only for the last position)

No KV-cache anywhere -- consistent with model.py, every generated token
recomputes the full sequence. position_ids are never needed by the decoder
layers themselves when past_key_values=None (verified against HF source),
so the exported graphs take only (hidden_states, attention_mask, cos, sin).
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import json
import numpy as np
import torch
import torch.nn as nn

from model import RecurrentDepthWrapper, MODEL_ID

OUT_DIR = "onnx_export"
MAX_LEN = 256  # rope table + causal-mask precompute horizon for the demo

os.makedirs(OUT_DIR, exist_ok=True)


class TrunkPre(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask, cos, sin):
        h = hidden_states
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, position_embeddings=(cos, sin), use_cache=False)
        return h


class CoreStep(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, attention_mask, cos, sin):
        return self.layer(hidden_states, attention_mask=attention_mask, position_embeddings=(cos, sin), use_cache=False)


class TrunkPost(nn.Module):
    def __init__(self, layers, norm):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm

    def forward(self, hidden_states, attention_mask, cos, sin):
        h = hidden_states
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, position_embeddings=(cos, sin), use_cache=False)
        return self.norm(h)


def build_causal_mask(T, dtype=torch.float32):
    mask = torch.full((T, T), torch.finfo(dtype).min, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, T, T)


def main():
    print("loading base model + fine-tuned core...")
    wrapper = RecurrentDepthWrapper()
    core_sd = torch.load("core_final.pt", map_location="cpu")
    wrapper.core.load_state_dict(core_sd)
    wrapper.eval()

    trunk_pre = TrunkPre(list(wrapper.prelude)).eval()
    core_step = CoreStep(wrapper.core).eval()
    trunk_post = TrunkPost(list(wrapper.coda), wrapper.norm).eval()

    # ---- sanity: decomposed pipeline must equal wrapper.forward exactly ----
    torch.manual_seed(0)
    T = 12
    input_ids = torch.randint(0, wrapper.config.vocab_size, (1, T))
    r_test = 5
    with torch.no_grad():
        ref_logits = wrapper(input_ids, r=r_test)

        inputs_embeds = wrapper.tok_emb(input_ids)
        position_ids = torch.arange(T).unsqueeze(0)
        cos, sin = wrapper.rotary_emb(inputs_embeds, position_ids=position_ids)
        mask = build_causal_mask(T)

        h = trunk_pre(inputs_embeds, mask, cos, sin)
        for _ in range(r_test):
            h = core_step(h, mask, cos, sin)
        h = trunk_post(h, mask, cos, sin)
        logits_manual = h @ wrapper.tok_emb.weight.T

    err = (ref_logits - logits_manual).abs().max().item()
    print(f"decomposition max abs diff vs wrapper.forward: {err:.6f}")
    assert err < 1e-3, "decomposed pipeline does not match reference forward!"
    print("decomposition verified OK")

    # ---- export the 3 ONNX graphs (dynamic seq_len) ----
    dummy_h = torch.zeros(1, T, wrapper.config.hidden_size)
    dummy_mask = build_causal_mask(T)
    dummy_cos = cos
    dummy_sin = sin
    dyn_axes = {
        "hidden_states": {1: "T"},
        "attention_mask": {2: "T", 3: "T"},
        "cos": {1: "T"},
        "sin": {1: "T"},
        "out": {1: "T"},
    }

    def export(mod, name):
        path = os.path.join(OUT_DIR, name)
        torch.onnx.export(
            mod,
            (dummy_h, dummy_mask, dummy_cos, dummy_sin),
            path,
            input_names=["hidden_states", "attention_mask", "cos", "sin"],
            output_names=["out"],
            dynamic_axes=dyn_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"exported {name}: {size_mb:.1f} MB")

    print("exporting trunk_pre.onnx ...")
    export(trunk_pre, "trunk_pre.onnx")
    print("exporting core.onnx ...")
    export(core_step, "core.onnx")
    print("exporting trunk_post.onnx ...")
    export(trunk_post, "trunk_post.onnx")

    # ---- raw fp16 tied embedding weight, for JS-side gather + lm_head matmul ----
    emb = wrapper.tok_emb.weight.detach().numpy().astype(np.float16)  # [vocab, hidden]
    emb.tofile(os.path.join(OUT_DIR, "embed_fp16.bin"))
    print(f"exported embed_fp16.bin: {emb.nbytes/1e6:.1f} MB  shape={emb.shape}")

    # ---- precomputed RoPE cos/sin table for positions 0..MAX_LEN-1 ----
    with torch.no_grad():
        dummy = torch.zeros(1, MAX_LEN, wrapper.config.hidden_size)
        pos = torch.arange(MAX_LEN).unsqueeze(0)
        table_cos, table_sin = wrapper.rotary_emb(dummy, position_ids=pos)
    rope = {
        "max_len": MAX_LEN,
        "head_dim": wrapper.config.head_dim,
        "cos": table_cos[0].numpy().astype(np.float32).tolist(),
        "sin": table_sin[0].numpy().astype(np.float32).tolist(),
    }
    with open(os.path.join(OUT_DIR, "rope_table.json"), "w") as f:
        json.dump(rope, f)
    print(f"exported rope_table.json: max_len={MAX_LEN} head_dim={wrapper.config.head_dim}")

    meta = {
        "model_id": MODEL_ID,
        "n_prelude": len(wrapper.prelude),
        "n_coda": len(wrapper.coda),
        "hidden_size": wrapper.config.hidden_size,
        "num_attention_heads": wrapper.config.num_attention_heads,
        "num_key_value_heads": wrapper.config.num_key_value_heads,
        "head_dim": wrapper.config.head_dim,
        "vocab_size": wrapper.config.vocab_size,
        "max_len": MAX_LEN,
        "trained_r_max": 12,
        "tied_embeddings": True,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("exported meta.json:", meta)
    print("done.")


if __name__ == "__main__":
    main()
