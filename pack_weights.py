"""Repack the verbose JSON weight dump into a compact binary blob for the
browser: one flat Float32Array (weights.bin) plus a small manifest.json
recording tensor names, shapes and byte offsets. Keeps the JS loader tiny
(fetch + DataView, no giant JSON parse)."""
import json
import numpy as np

with open("/home/claude/webLTM/eval/weights.json") as f:
    data = json.load(f)

manifest = {
    "vocab": data["vocab"],
    "seq_len": data["seq_len"],
    "ans_start": data["ans_start"],
    "d": data["d"],
    "d_model": data["d_model"],
    "n_heads": data["n_heads"],
    "tensors": {},
}

blob = bytearray()
offset = 0
# deterministic order
for name in sorted(data["weights"].keys()):
    arr = np.array(data["weights"][name], dtype=np.float32)
    manifest["tensors"][name] = {"shape": list(arr.shape), "offset": offset, "length": int(arr.size)}
    blob += arr.tobytes()
    offset += arr.size * 4

with open("/home/claude/webLTM/web/weights.bin", "wb") as f:
    f.write(bytes(blob))
with open("/home/claude/webLTM/web/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"weights.bin: {len(blob)/1024:.1f} KB, {len(manifest['tensors'])} tensors")
for k, v in manifest["tensors"].items():
    print(f"  {k:35s} {v['shape']}")
