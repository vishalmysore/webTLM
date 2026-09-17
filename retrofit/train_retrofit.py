import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import random
import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import RecurrentDepthWrapper, MODEL_ID, generate

torch.manual_seed(0)
random.seed(0)

R_MAX = 12
STEPS = 500
BATCH = 8
LR = 2e-4
EVAL_EVERY = 50
CKPT_EVERY = 100

def get_batch(data, batch_size):
    idx = torch.randint(0, data.size(0), (batch_size,))
    chunk = data[idx]
    return chunk[:, :-1], chunk[:, 1:]

@torch.no_grad()
def eval_loss(wrapper, val, r, n_batches=5, batch_size=8):
    wrapper.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val, batch_size)
        logits = wrapper(x, r=r)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    wrapper.train()
    return sum(losses) / len(losses)


def main():
    data = torch.load("data.pt")
    train, val = data["train"], data["val"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    wrapper = RecurrentDepthWrapper()
    wrapper.train()
    opt = torch.optim.AdamW(wrapper.trainable_parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    log = []
    t0 = time.time()
    for step in range(STEPS):
        r = random.randint(1, R_MAX)
        x, y = get_batch(train, BATCH)
        logits = wrapper(x, r=r)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.trainable_parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 10 == 0:
            elapsed = time.time() - t0
            print(f"step {step:4d}  r={r:2d}  loss {loss.item():.3f}  ({elapsed:.0f}s)", flush=True)

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            vloss_low = eval_loss(wrapper, val, r=2)
            vloss_mid = eval_loss(wrapper, val, r=10)
            vloss_hi = eval_loss(wrapper, val, r=R_MAX)
            print(f"  eval @ step {step}: val_loss r=2:{vloss_low:.3f}  r=10:{vloss_mid:.3f}  r={R_MAX}:{vloss_hi:.3f}", flush=True)
            log.append({"step": step, "train_loss": loss.item(), "r": r,
                        "val_loss_r2": vloss_low, "val_loss_r10": vloss_mid, "val_loss_rmax": vloss_hi})
            sample = generate(wrapper, tok, "The history of the city began when", r=10, max_new_tokens=30)
            print(f"  sample(r=10): {sample!r}", flush=True)

        if step % CKPT_EVERY == 0 or step == STEPS - 1:
            torch.save(wrapper.core.state_dict(), "core_checkpoint.pt")
            with open("train_log.json", "w") as f:
                json.dump(log, f, indent=2)

    print("done", time.time() - t0)
    torch.save(wrapper.core.state_dict(), "core_final.pt")
    with open("train_log.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
