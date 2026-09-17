"""Convert the three exported ONNX graphs to fp16 weights to roughly halve
browser download size. Inputs/outputs stay fp32 (keep_io_types=True) so the
JS side doesn't need to juggle two numeric formats for activations -- only
the stored weights shrink."""
import os
import onnx
from onnxconverter_common import float16

OUT_DIR = "onnx_export"

for name in ["trunk_pre.onnx", "core.onnx", "trunk_post.onnx"]:
    path = os.path.join(OUT_DIR, name)
    model = onnx.load(path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    fp16_path = os.path.join(OUT_DIR, name.replace(".onnx", "_fp16.onnx"))
    onnx.save(model_fp16, fp16_path)
    before = os.path.getsize(path) / 1e6
    after = os.path.getsize(fp16_path) / 1e6
    print(f"{name}: {before:.1f}MB -> {after:.1f}MB")
