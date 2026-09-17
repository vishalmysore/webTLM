"""Tokenize wikitext-2-raw into fixed-length chunks for causal LM training.
Cached to disk so train_retrofit.py doesn't need network access to rerun."""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from model import MODEL_ID

SEQ_LEN = 128

def build(seq_len=SEQ_LEN):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def tokenize_split(split):
        texts = [t for t in ds[split]["text"] if len(t.strip()) > 200]
        ids = []
        for t in texts:
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
            ids.append(tok.eos_token_id)
        chunks = [ids[i:i + seq_len + 1] for i in range(0, len(ids) - seq_len - 1, seq_len)]
        return torch.tensor(chunks, dtype=torch.long)

    train = tokenize_split("train")
    val = tokenize_split("validation")
    print("train chunks:", train.shape, "val chunks:", val.shape)
    torch.save({"train": train, "val": val, "seq_len": seq_len}, "data.pt")


if __name__ == "__main__":
    build()
