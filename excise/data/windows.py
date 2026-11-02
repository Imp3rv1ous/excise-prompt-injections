"""Tokenization, BIOES token tags and windows for training (report Sections 3.1 and 3.2).

Every window that contains an injection is kept with two or three windows without one: the same position on the
pair's original page, the nearest injection-free window of the same attack page, and the same position on a random
original page of the split.
Input: the dataset, a split and its labels. Output: data/cache/<split>.pt, a list of
(token ids, token tags, window label, view).
"""

import argparse
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from excise.config import BACKBONE, STRIDE, WINDOW
from excise.data.labels import load_labels
from excise.data.source import VIEWS, load_pages, normalize, original_id, read_split

TAGS = ["O", "B", "I", "E", "S"]
O, B, I, E, S = range(5)  # noqa: E741 (standard BIOES tag names)


def tokenizer(path=BACKBONE["repo"]):
    revision = BACKBONE["revision"] if path == BACKBONE["repo"] else None
    return AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=True)


def encode(tok, text):
    """(token ids int32 [T], character offsets int64 [T, 2]), no special tokens."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    return np.array(enc.input_ids, dtype=np.int32), np.array(enc.offset_mapping, dtype=np.int64)


def window_starts(n):
    """Windows every STRIDE tokens; the last one ends at the page end."""
    if n <= WINDOW:
        return [0]
    return list(range(0, n - WINDOW, STRIDE)) + [n - WINDOW]


def token_tags(offsets, spans):
    """Character spans to BIOES tags: a token overlapping a span is inside it."""
    inside = np.zeros(len(offsets), dtype=bool)
    for s, e in spans:
        inside |= (offsets[:, 0] < e) & (offsets[:, 1] > s)

    tags = np.full(len(offsets), O, dtype=np.int8)
    edges = np.diff(np.r_[0, inside.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a == 1:
            tags[a] = S
        else:
            tags[a], tags[a + 1 : b - 1], tags[b - 1] = B, I, E
    return tags


def window_at(ids, start):
    s = min(start, max(0, len(ids) - WINDOW))
    return ids[s : s + WINDOW]


def attack_windows(ids, tags, original, pool, rng):
    """(ids, tags or None, window label) for the windows of one attack page."""
    starts = window_starts(len(ids))
    wins = [(ids[s : s + WINDOW], tags[s : s + WINDOW]) for s in starts]
    positive = [k for k, (_, t) in enumerate(wins) if (t != O).any()]
    negative = [k for k in range(len(wins)) if k not in positive]

    out = []
    for k in positive:
        out.append((wins[k][0], wins[k][1], 1))
        out.append((window_at(original, starts[k]), None, 0))
        if negative:
            out.append((wins[min(negative, key=lambda n: abs(n - k))][0], None, 0))
        out.append((window_at(rng.choice(pool), starts[k]), None, 0))
    return out


def build_windows(df, entry_ids, labels, tok, seed=0):
    """labels: {(entry_id, view): spans} of accepted pairs. Returns [(ids, tags, window label, view)] tensors."""
    rng = random.Random(seed)
    rows = df.set_index("entry_id")
    out = []
    for view, column in VIEWS.items():
        originals = {e: encode(tok, normalize(rows.loc[e][column]))[0] for e in entry_ids if e.endswith("_benign")}
        pool = list(originals.values())
        for e in entry_ids:
            if (e, view) not in labels:
                continue
            ids, offsets = encode(tok, normalize(rows.loc[e][column]))
            tags = token_tags(offsets, labels[(e, view)])
            for w_ids, w_tags, label in attack_windows(ids, tags, originals[original_id(e)], pool, rng):
                w_tags = w_tags if w_tags is not None else np.full(len(w_ids), O, np.int8)
                out.append((torch.from_numpy(w_ids.astype(np.int64)), torch.from_numpy(w_tags), label, view))
    return out


def main():
    p = argparse.ArgumentParser(description="Tokenize and cut training windows.")
    p.add_argument("--splits", nargs="+", default=["train", "dev"])
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--out", default="data/cache")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_pages()
    tok = tokenizer()
    for name in args.splits:
        windows = build_windows(df, read_split(name), load_labels(name, args.labels), tok)
        torch.save(windows, f"{args.out}/{name}.pt")
        print(name, len(windows), "windows,", sum(w[2] for w in windows), "with an injection")


if __name__ == "__main__":
    main()
