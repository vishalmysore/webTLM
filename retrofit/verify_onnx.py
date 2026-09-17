import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np
import torch
import onnxruntime as ort

from model import RecurrentDepthWrapper
from export_onnx import build_causal_mask

OUT_DIR = "onnx_export"

wrapper = RecurrentDepthWrapper()
core_sd = torch.load("core_final.pt", map_location="cpu")
wrapper.core.load_state_dict(core_sd)
wrapper.eval()

torch.manual_seed(1)
T = 17
r = 7
input_ids = torch.randint(0, wrapper.config.vocab_size, (1, T))

with torch.no_grad():
    ref_logits = wrapper(input_ids, r=r).numpy()
    inputs_embeds = wrapper.tok_emb(input_ids)
    position_ids = torch.arange(T).unsqueeze(0)
    cos, sin = wrapper.rotary_emb(inputs_embeds, position_ids=position_ids)
    mask = build_causal_mask(T)

sess_pre = ort.InferenceSession(os.path.join(OUT_DIR, "trunk_pre.onnx"))
sess_core = ort.InferenceSession(os.path.join(OUT_DIR, "core.onnx"))
sess_post = ort.InferenceSession(os.path.join(OUT_DIR, "trunk_post.onnx"))

def run(sess, h, mask, cos, sin):
    out = sess.run(["out"], {
        "hidden_states": h.astype(np.float32),
        "attention_mask": mask.astype(np.float32),
        "cos": cos.astype(np.float32),
        "sin": sin.astype(np.float32),
    })[0]
    return out

h = inputs_embeds.numpy()
mask_np = mask.numpy()
cos_np = cos.numpy()
sin_np = sin.numpy()

h = run(sess_pre, h, mask_np, cos_np, sin_np)
for _ in range(r):
    h = run(sess_core, h, mask_np, cos_np, sin_np)
h = run(sess_post, h, mask_np, cos_np, sin_np)

emb = np.fromfile(os.path.join(OUT_DIR, "embed_fp16.bin"), dtype=np.float16).reshape(wrapper.config.vocab_size, wrapper.config.hidden_size).astype(np.float32)
onnx_logits = h @ emb.T

err = np.abs(ref_logits - onnx_logits).max()
rel = err / (np.abs(ref_logits).max() + 1e-8)
print(f"max abs diff: {err:.6f}  (relative: {rel:.6f})")
print("ref argmax last-token:", ref_logits[0, -1].argmax(), " onnx argmax last-token:", onnx_logits[0, -1].argmax())
assert err < 0.05, "ONNX pipeline diverges from PyTorch reference!"
print("ONNX pipeline verified OK")

# also sanity-check the rope table matches on-the-fly rotary_emb for these positions
import json
with open(os.path.join(OUT_DIR, "rope_table.json")) as f:
    rope = json.load(f)
table_cos = np.array(rope["cos"])[:T]
table_sin = np.array(rope["sin"])[:T]
cos_err = np.abs(table_cos - cos_np[0]).max()
sin_err = np.abs(table_sin - sin_np[0]).max()
print(f"rope table max diff: cos={cos_err:.8f} sin={sin_err:.8f}")
assert cos_err < 1e-4 and sin_err < 1e-4
print("rope table verified OK")
