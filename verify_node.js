const fs = require("fs");
const path = require("path");
const { runModel, tokenizeProblem, digitsToNumber } = require("./web/model.js");

const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, "web/manifest.json"), "utf8"));
const bin = fs.readFileSync(path.join(__dirname, "web/weights.bin"));
const buffer = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const model = { manifest, buffer };

const testCases = [
  [5, 3], [12, 88], [999, 1], [4321, 5678], [99999, 99999],
  [10203, 4], [7, 70000], [55555, 44444], [100, 900], [30303, 7],
];
const rValues = [1, 4, 8];

const out = {};
for (const [a, b] of testCases) {
  const key = `${a}+${b}`;
  out[key] = { true: a + b, preds: {} };
  const ids = tokenizeProblem(manifest, a, b);
  for (const r of rValues) {
    const { answerDigits } = runModel(model, ids, r);
    const vocab = manifest.vocab;
    const s = answerDigits.map((id) => vocab[id]).join("");
    out[key].preds[String(r)] = s;
  }
}
fs.writeFileSync(path.join(__dirname, "eval/verify_node.json"), JSON.stringify(out, null, 2));
console.log(JSON.stringify(out, null, 2));
