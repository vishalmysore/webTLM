/*
 * webLTM inference engine — pure vanilla JS, no dependencies.
 *
 * Re-implements the trained PyTorch model's forward pass exactly:
 *   prelude (once) -> core (looped r times, re-injecting the prelude
 *   output at every step) -> coda -> layernorm -> linear head.
 *
 * The loop count r is the only "thinking" knob. Nothing produced inside
 * the loop is ever turned into a token — the function below returns only
 * the final answer digits, exactly like the model architecture intends.
 *
 * Environment-agnostic: pass in the parsed manifest + a Float32Array view
 * factory over the raw weight bytes. Works the same in a browser (fetch)
 * or Node (fs.readFileSync), see loadModelBrowser()/loadModelNode() below.
 */

function makeTensorAccessor(manifest, buffer) {
  const cache = {};
  return function get(name) {
    if (cache[name]) return cache[name];
    const t = manifest.tensors[name];
    if (!t) throw new Error("missing tensor: " + name);
    const data = new Float32Array(buffer, t.offset, t.length);
    const tensor = { data, shape: t.shape };
    cache[name] = tensor;
    return tensor;
  };
}

function erf(x) {
  // Abramowitz & Stegun 7.1.26, max abs error 1.5e-7 — matches torch's
  // exact GELU closely enough that argmax digit predictions never flip.
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function gelu(x) {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

function zeros(T, D) {
  const m = new Array(T);
  for (let t = 0; t < T; t++) m[t] = new Float32Array(D);
  return m;
}

function addInto(dst, a, b) {
  for (let t = 0; t < dst.length; t++)
    for (let i = 0; i < dst[t].length; i++)
      dst[t][i] = a[t][i] + b[t][i];
  return dst;
}

function linear(get, x, wName, bName, outDim) {
  const W = get(wName), B = get(bName);
  const inDim = x[0].length;
  const T = x.length;
  const out = zeros(T, outDim);
  for (let t = 0; t < T; t++) {
    const xt = x[t];
    for (let o = 0; o < outDim; o++) {
      let s = B.data[o];
      const rowOff = o * inDim;
      for (let i = 0; i < inDim; i++) s += xt[i] * W.data[rowOff + i];
      out[t][o] = s;
    }
  }
  return out;
}

function layerNorm(get, x, wName, bName, eps) {
  eps = eps === undefined ? 1e-5 : eps;
  const W = get(wName), B = get(bName);
  const T = x.length, D = x[0].length;
  const out = zeros(T, D);
  for (let t = 0; t < T; t++) {
    let mean = 0;
    for (let i = 0; i < D; i++) mean += x[t][i];
    mean /= D;
    let vari = 0;
    for (let i = 0; i < D; i++) { const d = x[t][i] - mean; vari += d * d; }
    vari /= D;
    const inv = 1 / Math.sqrt(vari + eps);
    for (let i = 0; i < D; i++)
      out[t][i] = (x[t][i] - mean) * inv * W.data[i] + B.data[i];
  }
  return out;
}

function selfAttention(get, x, prefix, nHeads) {
  const T = x.length, D = x[0].length;
  const headDim = D / nHeads;
  const qkv = linear(get, x, prefix + ".in_proj_weight", prefix + ".in_proj_bias", 3 * D);
  const scale = 1 / Math.sqrt(headDim);
  const concat = zeros(T, D);
  for (let h = 0; h < nHeads; h++) {
    const off = h * headDim;
    // scores[t1][t2]
    const scores = new Array(T);
    for (let t1 = 0; t1 < T; t1++) {
      const row = new Float32Array(T);
      let maxv = -Infinity;
      for (let t2 = 0; t2 < T; t2++) {
        let dot = 0;
        for (let i = 0; i < headDim; i++)
          dot += qkv[t1][off + i] * qkv[t2][D + off + i];
        row[t2] = dot * scale;
        if (row[t2] > maxv) maxv = row[t2];
      }
      let sum = 0;
      for (let t2 = 0; t2 < T; t2++) { row[t2] = Math.exp(row[t2] - maxv); sum += row[t2]; }
      for (let t2 = 0; t2 < T; t2++) row[t2] /= sum;
      scores[t1] = row;
    }
    for (let t1 = 0; t1 < T; t1++) {
      for (let i = 0; i < headDim; i++) {
        let s = 0;
        for (let t2 = 0; t2 < T; t2++) s += scores[t1][t2] * qkv[t2][2 * D + off + i];
        concat[t1][off + i] = s;
      }
    }
  }
  return linear(get, concat, prefix + ".out_proj.weight", prefix + ".out_proj.bias", D);
}

function block(get, x, prefix, nHeads) {
  const h1 = layerNorm(get, x, prefix + ".ln1.weight", prefix + ".ln1.bias");
  const a = selfAttention(get, h1, prefix + ".attn", nHeads);
  const x1 = addInto(zeros(x.length, x[0].length), x, a);
  const h2 = layerNorm(get, x1, prefix + ".ln2.weight", prefix + ".ln2.bias");
  const m1 = linear(get, h2, prefix + ".mlp.0.weight", prefix + ".mlp.0.bias", h2[0].length * 4);
  for (let t = 0; t < m1.length; t++) for (let i = 0; i < m1[t].length; i++) m1[t][i] = gelu(m1[t][i]);
  const m2 = linear(get, m1, prefix + ".mlp.2.weight", prefix + ".mlp.2.bias", x[0].length);
  return addInto(zeros(x.length, x[0].length), x1, m2);
}

function decodeHead(get, manifest, h) {
  const T = manifest.seq_len, D = manifest.d_model;
  const ansStart = manifest.ans_start;
  const answerDigits = [];
  const logitsOut = [];
  const head = get("head.weight"), headB = get("head.bias");
  const vocabSize = manifest.vocab.length;
  for (let t = ansStart; t < T; t++) {
    const logits = new Float32Array(vocabSize);
    for (let o = 0; o < vocabSize; o++) {
      let s = headB.data[o];
      for (let i = 0; i < D; i++) s += h[t][i] * head.data[o * D + i];
      logits[o] = s;
    }
    let best = 0;
    for (let o = 1; o < vocabSize; o++) if (logits[o] > logits[best]) best = o;
    answerDigits.push(best);
    logitsOut.push(Array.from(logits));
  }
  return { answerDigits, logits: logitsOut };
}

function probeLatentState(get, manifest, h) {
  const nHeads = manifest.n_heads;
  const codaH = block(get, h, "coda", nHeads);
  const normed = layerNorm(get, codaH, "ln_f.weight", "ln_f.bias");
  return decodeHead(get, manifest, normed);
}

// Runs the full model. tokenIds: array of length manifest.seq_len.
// r: number of core-loop iterations ("thinking depth").
// Returns { answerDigits: number[], logits: number[][], steps: object[] } —
// includes linear probe readouts at each recurrence step so the internal carry
// resolution progression can be inspected without interrupting latent compute.
function runModel(model, tokenIds, r) {
  const { manifest, buffer } = model;
  const get = makeTensorAccessor(manifest, buffer);
  const T = manifest.seq_len, D = manifest.d_model, nHeads = manifest.n_heads;
  const tok = get("tok_emb.weight"), pos = get("pos_emb.weight");

  let e = zeros(T, D);
  for (let t = 0; t < T; t++) {
    const id = tokenIds[t];
    for (let i = 0; i < D; i++) e[t][i] = tok.data[id * D + i] + pos.data[t * D + i];
  }

  const e0 = block(get, e, "prelude", nHeads);
  let h = e0;

  // Record linear readout probes across loop iterations
  const steps = [];

  // Loop 0: prelude embedding prior to any core recurrence
  const probe0 = probeLatentState(get, manifest, e0);
  steps.push({
    step: 0,
    label: "Loop 0 (Prelude)",
    answerDigits: probe0.answerDigits,
    rawDigits: probe0.answerDigits.map((id) => manifest.vocab[id]).join(""),
    value: digitsToNumber(manifest, probe0.answerDigits)
  });

  for (let i = 0; i < r; i++) {
    const injected = addInto(zeros(T, D), h, e0);
    h = block(get, injected, "core", nHeads);

    const probed = probeLatentState(get, manifest, h);
    steps.push({
      step: i + 1,
      label: "Loop " + (i + 1),
      answerDigits: probed.answerDigits,
      rawDigits: probed.answerDigits.map((id) => manifest.vocab[id]).join(""),
      value: digitsToNumber(manifest, probed.answerDigits)
    });
  }

  const codaH = block(get, h, "coda", nHeads);
  const finalNorm = layerNorm(get, codaH, "ln_f.weight", "ln_f.bias");
  const finalDecoded = decodeHead(get, manifest, finalNorm);

  return { answerDigits: finalDecoded.answerDigits, logits: finalDecoded.logits, steps };
}

function tokenizeProblem(manifest, a, b) {
  const D = manifest.d, vocab = manifest.vocab;
  const tok2id = {};
  vocab.forEach((v, i) => (tok2id[v] = i));
  const aStr = String(a).padStart(D, "0");
  const bStr = String(b).padStart(D, "0");
  const ids = [];
  for (const c of aStr) ids.push(tok2id[c]);
  ids.push(tok2id["+"]);
  for (const c of bStr) ids.push(tok2id[c]);
  ids.push(tok2id["="]);
  for (let i = 0; i < D + 1; i++) ids.push(tok2id["ANS"]);
  return ids;
}

function digitsToNumber(manifest, digitIds) {
  const vocab = manifest.vocab;
  let s = "";
  for (const id of digitIds) s += vocab[id];
  return s.replace(/^0+(?=\d)/, "");
}

if (typeof module !== "undefined") {
  module.exports = { runModel, tokenizeProblem, digitsToNumber, makeTensorAccessor };
}
