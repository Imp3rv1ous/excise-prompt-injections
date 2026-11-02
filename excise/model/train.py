"""Training (report Section 3.2 and Appendix A).

Loss = 1.5 x tag-weighted cross-entropy of the token head + 0.3 x binary cross-entropy of the window head.
AdamW (heads at 10x the backbone learning rate), 5% linear warmup then linear decay, three epochs; the epoch with
the lowest combined dev loss is kept.
Input: data/cache/{train,dev}.pt. Output: model.safetensors, config and tokenizer in --out, plus train_log.jsonl
(one line per optimizer step and one per epoch with the dev losses).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from excise.config import BACKBONE, TRAIN
from excise.data.windows import O, tokenizer
from excise.model.model import ExcisionModel, device


def collate(batch):
    ids = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=BACKBONE["pad_id"])
    tags = pad_sequence([b[1].long() for b in batch], batch_first=True, padding_value=-100)
    lengths = torch.tensor([len(b[0]) for b in batch])
    mask = (torch.arange(ids.shape[1])[None] < lengths[:, None]).long()
    return ids, mask, tags, torch.tensor([float(b[2]) for b in batch])


def tag_weights(windows):
    """w_c = (f_c / f_O)^(-1/2), clipped to [1, tag_weight_max]."""
    f = np.bincount(np.concatenate([w[1].numpy() for w in windows]), minlength=5)
    return torch.tensor(np.clip((f / f[O]) ** -0.5, 1, TRAIN["tag_weight_max"]), dtype=torch.float)


def loss_fn(tag_logits, win_logits, tags, win, weights):
    """(total loss, token-head loss, window-head loss)."""
    ce = F.cross_entropy(tag_logits.flatten(0, 1), tags.flatten(), weight=weights, ignore_index=-100)
    bce = F.binary_cross_entropy_with_logits(win_logits, win)
    return TRAIN["token_loss_weight"] * ce + TRAIN["window_loss_weight"] * bce, ce, bce


def batch_loss(model, batch, weights, dev):
    ids, mask, tags, win = (x.to(dev) for x in batch)
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        tag_logits, win_logits = model(ids, mask)
    return loss_fn(tag_logits.float(), win_logits.float(), tags, win, weights)


@torch.no_grad()
def dev_loss(model, loader, weights, dev):
    model.eval()
    losses = [[x.item() for x in batch_loss(model, batch, weights, dev)] for batch in loader]
    model.train()
    return np.mean(losses, axis=0).tolist()


def train(args):
    dev = device()
    train_set, dev_set = torch.load(args.train), torch.load(args.dev)
    weights = tag_weights(train_set).to(dev)
    model = ExcisionModel.from_backbone(args.backbone).to(dev).train()

    heads = list(model.token_head.parameters()) + list(model.window_head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr},
            {"params": heads, "lr": TRAIN["head_lr_scale"] * args.lr},
        ]
    )
    steps = args.epochs * math.ceil(len(train_set) / (args.batch * args.accum))
    sched = get_linear_schedule_with_warmup(opt, int(TRAIN["warmup"] * steps), steps)
    generator = torch.Generator().manual_seed(TRAIN["seed"])
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=collate, generator=generator)
    dev_loader = DataLoader(dev_set, batch_size=args.batch, collate_fn=collate)

    os.makedirs(args.out, exist_ok=True)
    log = open(f"{args.out}/train_log.jsonl", "a")
    best, step, t0, acc = float("inf"), 0, time.time(), np.zeros(3)
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            loss, ce, bce = batch_loss(model, batch, weights, dev)
            (loss / args.accum).backward()
            acc += [loss.item(), ce.item(), bce.item()]
            if (i + 1) % args.accum:
                continue

            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            total, tok, win = (acc / args.accum).tolist()
            acc[:] = 0
            row = dict(step=step, epoch=epoch, lr=sched.get_last_lr()[0], loss_token=tok, loss_window=win, loss=total)
            log.write(json.dumps(dict(row, sec=time.time() - t0)) + "\n")
            log.flush()

        held, held_tok, held_win = dev_loss(model, dev_loader, weights, dev)
        row = dict(epoch=epoch, step=step, dev_loss=held, dev_loss_token=held_tok, dev_loss_window=held_win)
        log.write(json.dumps(dict(row, sec=time.time() - t0)) + "\n")
        log.flush()
        print(f"epoch {epoch} step {step} dev_loss {held:.4f}", flush=True)

        if held < best:
            best = held
            model.save(args.out)
            tokenizer(args.backbone).save_pretrained(args.out)


def main():
    p = argparse.ArgumentParser(description="Train the excision model.")
    p.add_argument("--train", default="data/cache/train.pt")
    p.add_argument("--dev", default="data/cache/dev.pt")
    p.add_argument("--out", default="weights/run1")
    p.add_argument("--backbone", default=BACKBONE["repo"])
    p.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    p.add_argument("--batch", type=int, default=TRAIN["batch"])
    p.add_argument("--accum", type=int, default=TRAIN["accum"])
    p.add_argument("--lr", type=float, default=TRAIN["lr"])
    train(p.parse_args())


if __name__ == "__main__":
    main()
