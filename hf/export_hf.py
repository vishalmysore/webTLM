"""Package the trained webLTM weights for a Hugging Face model repo:
config.json + model.safetensors (no pickle), ready for
`huggingface-cli upload`."""
import json
import numpy as np
from safetensors.numpy import save_file

with open("/home/claude/webLTM/eval/weights.json") as f:
    data = json.load(f)

tensors = {k: np.array(v, dtype=np.float32) for k, v in data["weights"].items()}
save_file(tensors, "/home/claude/webLTM/hf/model.safetensors")

config = {
    "architecture": "webLTM-recurrent-depth",
    "vocab": data["vocab"],
    "seq_len": data["seq_len"],
    "ans_start": data["ans_start"],
    "digits_per_operand": data["d"],
    "d_model": data["d_model"],
    "n_heads": data["n_heads"],
    "n_blocks": 3,
    "block_order": ["prelude", "core", "coda"],
    "recurrent_block": "core",
    "trained_recurrence_range": [1, 8],
    "task": "non-autoregressive multi-digit addition",
    "params": sum(t.size for t in tensors.values()),
}
with open("/home/claude/webLTM/hf/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("params:", config["params"])
print("tensors:", len(tensors))
