# Uploading the retrofit ONNX assets to Hugging Face

`onnx_export/` (~354MB: two ~142MB `.onnx` files, one ~14MB `.onnx` file, a
~57MB `embed_fp16.bin`, plus tiny JSON configs) is too large for a normal
`git push` — GitHub hard-blocks any single file over 100MB, and two of these
files exceed that. It's pushed straight to the existing HF weights repo
instead, the same way `hf/model.safetensors` was uploaded earlier in this
project.

From `C:\work\webLTM\retrofit` (or wherever this checkout lives), with the
`hf` CLI already authenticated (same login used for the original
`VishalMysore/webLTM` upload):

```
set HF_HUB_DISABLE_XET=1
hf upload VishalMysore/webLTM onnx_export onnx
```

That uploads the local `onnx_export/` folder's contents into the `onnx/`
subfolder of the `VishalMysore/webLTM` repo on the Hub, so the files end up
at:

- `https://huggingface.co/VishalMysore/webLTM/resolve/main/onnx/trunk_pre.onnx`
- `https://huggingface.co/VishalMysore/webLTM/resolve/main/onnx/core.onnx`
- `https://huggingface.co/VishalMysore/webLTM/resolve/main/onnx/trunk_post.onnx`
- `https://huggingface.co/VishalMysore/webLTM/resolve/main/onnx/embed_fp16.bin`
- `https://huggingface.co/VishalMysore/webLTM/resolve/main/onnx/rope_table.json`
- `https://huggingface.co/VishalMysore/webLTM/resolve/main/onnx/meta.json`

which is exactly what `docs/retrofit/model.js`'s `HF_BASE` constant points
at. Until this upload runs, the "Real Retrofit" demo page will load but
fail at the "fetching config…" step (404 on `meta.json`) — that's expected
and the error message says as much.

`HF_HUB_DISABLE_XET=1` avoids the same Xet-storage-backend issue worked
around earlier when downloading SmolLM2-135M — it forces the plain HTTP
upload/download path instead.
