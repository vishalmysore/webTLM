/*
 * Browser inference for the SmolLM2-135M recurrent-depth retrofit.
 *
 * Same prelude -> core(xr) -> coda pattern as webLTM's toy model, but here
 * the weights are a real 30-layer pretrained language model: layers 0-9
 * (prelude) and the last 10 layers (coda) run once, frozen, exactly as
 * HuggingFaceTB/SmolLM2-135M shipped them. A single copy of one original
 * layer (the "core") was fine-tuned and is looped r times. Everything else
 * -- 99M of the model's 102.65M parameters -- was never touched.
 *
 * No KV-cache: every generated token recomputes the full sequence through
 * prelude + core*r + coda from scratch. That's a deliberate simplification
 * (see the article), and it's the reason generation is slow -- this is
 * real transformer compute happening in your tab via WASM, not a trick.
 */

const HF_BASE = "https://huggingface.co/VishalMysore/webLTM/resolve/main/onnx";
const TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M";

let ort = null;
let AutoTokenizer = null;
let sessPre = null, sessCore = null, sessPost = null;
let embedTable = null;   // Float32Array [vocab * hidden], decoded from fp16
let ropeCos = null, ropeSin = null; // Float32Array [maxLen * headDim]
let meta = null;
let tokenizer = null;

export async function loadLibraries(onStatus) {
  onStatus("loading onnxruntime-web…");
  const ortMod = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/+esm");
  ort = ortMod.default || ortMod;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
  ort.env.wasm.numThreads = 1;   // GitHub Pages doesn't send COOP/COEP, so no SharedArrayBuffer
  ort.env.wasm.simd = true;

  onStatus("loading tokenizer library…");
  const txMod = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/+esm");
  AutoTokenizer = txMod.AutoTokenizer;
}

function fp16ToFp32(uint16) {
  // Standard IEEE-754 half -> single conversion.
  const out = new Float32Array(uint16.length);
  for (let i = 0; i < uint16.length; i++) {
    const h = uint16[i];
    const sign = (h & 0x8000) >> 15;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;
    let val;
    if (exp === 0) {
      val = (frac / 1024) * Math.pow(2, -14);
    } else if (exp === 0x1f) {
      val = frac ? NaN : Infinity;
    } else {
      val = (1 + frac / 1024) * Math.pow(2, exp - 15);
    }
    out[i] = sign ? -val : val;
  }
  return out;
}

export async function loadModel(onStatus, onProgress) {
  onStatus("fetching config…");
  const metaResp = await fetch(`${HF_BASE}/meta.json`);
  if (!metaResp.ok) throw new Error(`meta.json fetch failed (${metaResp.status}) — has the onnx/ folder been uploaded to the HF repo yet?`);
  meta = await metaResp.json();

  onStatus("fetching RoPE table…");
  const ropeResp = await fetch(`${HF_BASE}/rope_table.json`);
  const rope = await ropeResp.json();
  const maxLen = rope.max_len, headDim = rope.head_dim;
  ropeCos = new Float32Array(maxLen * headDim);
  ropeSin = new Float32Array(maxLen * headDim);
  for (let i = 0; i < maxLen; i++) {
    ropeCos.set(rope.cos[i], i * headDim);
    ropeSin.set(rope.sin[i], i * headDim);
  }

  onStatus("fetching tied embedding weight (~57MB)…");
  const embResp = await fetch(`${HF_BASE}/embed_fp16.bin`);
  const embBuf = await embResp.arrayBuffer();
  embedTable = fp16ToFp32(new Uint16Array(embBuf));

  onStatus("loading tokenizer…");
  tokenizer = await AutoTokenizer.from_pretrained(TOKENIZER_ID);

  onStatus("loading trunk_pre.onnx (~142MB)…");
  sessPre = await ort.InferenceSession.create(`${HF_BASE}/trunk_pre.onnx`, { executionProviders: ["wasm"] });
  onProgress(33);

  onStatus("loading core.onnx (~14MB)…");
  sessCore = await ort.InferenceSession.create(`${HF_BASE}/core.onnx`, { executionProviders: ["wasm"] });
  onProgress(66);

  onStatus("loading trunk_post.onnx (~142MB)…");
  sessPost = await ort.InferenceSession.create(`${HF_BASE}/trunk_post.onnx`, { executionProviders: ["wasm"] });
  onProgress(100);

  onStatus("ready");
}

function buildCausalMask(T) {
  const data = new Float32Array(T * T);
  const NEG = -1e9;
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      data[i * T + j] = j > i ? NEG : 0;
    }
  }
  return new ort.Tensor("float32", data, [1, 1, T, T]);
}

function sliceRope(T) {
  const headDim = meta.head_dim;
  const cos = new ort.Tensor("float32", ropeCos.slice(0, T * headDim), [1, T, headDim]);
  const sin = new ort.Tensor("float32", ropeSin.slice(0, T * headDim), [1, T, headDim]);
  return { cos, sin };
}

function embed(ids) {
  const H = meta.hidden_size;
  const data = new Float32Array(ids.length * H);
  for (let i = 0; i < ids.length; i++) {
    const row = embedTable.subarray(ids[i] * H, ids[i] * H + H);
    data.set(row, i * H);
  }
  return new ort.Tensor("float32", data, [1, ids.length, H]);
}

async function runTrunk(sess, hidden, mask, cos, sin) {
  const feeds = { hidden_states: hidden, attention_mask: mask, cos, sin };
  const out = await sess.run(feeds);
  return out.out;
}

function lastTokenLogits(hiddenTensor) {
  const H = meta.hidden_size, V = meta.vocab_size;
  const T = hiddenTensor.dims[1];
  const data = hiddenTensor.data;
  const lastOffset = (T - 1) * H;
  const logits = new Float32Array(V);
  for (let v = 0; v < V; v++) {
    let acc = 0;
    const rowOff = v * H;
    for (let h = 0; h < H; h++) {
      acc += embedTable[rowOff + h] * data[lastOffset + h];
    }
    logits[v] = acc;
  }
  return logits;
}

function sample(logits, temperature, topK) {
  const T = Math.max(temperature, 1e-5);
  const scaled = Array.from(logits, (x) => x / T);
  const idx = Array.from(scaled.keys()).sort((a, b) => scaled[b] - scaled[a]).slice(0, topK);
  const vals = idx.map((i) => scaled[i]);
  const maxV = Math.max(...vals);
  const exps = vals.map((v) => Math.exp(v - maxV));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return idx[i];
  }
  return idx[idx.length - 1];
}

export async function* generate(promptText, { r = 6, maxNewTokens = 20, temperature = 0.7, topK = 40 } = {}) {
  const encoded = await tokenizer(promptText, { return_tensor: false });
  let ids = Array.from(encoded.input_ids);
  const eosId = tokenizer.eos_token_id ?? tokenizer.model?.eos_token_id;

  for (let step = 0; step < maxNewTokens; step++) {
    const T = ids.length;
    if (T >= meta.max_len - 1) break;
    const t0 = performance.now();

    const mask = buildCausalMask(T);
    const { cos, sin } = sliceRope(T);
    let h = embed(ids);

    h = await runTrunk(sessPre, h, mask, cos, sin);
    for (let i = 0; i < r; i++) {
      h = await runTrunk(sessCore, h, mask, cos, sin);
    }
    h = await runTrunk(sessPost, h, mask, cos, sin);

    const logits = lastTokenLogits(h);
    const nextId = sample(logits, temperature, topK);
    ids.push(nextId);

    const ms = performance.now() - t0;
    const text = tokenizer.decode(ids, { skip_special_tokens: true });
    yield { text, tokenMs: ms, step, ids: ids.slice() };

    if (eosId !== undefined && nextId === eosId) break;
  }
}

export function getMeta() {
  return meta;
}
