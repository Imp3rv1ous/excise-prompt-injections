"""Decoding parameters on the dev set (report Section 3.3 and Appendix A).

Threshold per view: the 99th percentile of the page scores of the dev original pages.
Offsets: a grid search over (O, B, E) in {-2, ..., 2}^3 that maximizes the share of dev attack pages (with label
spans) restored to their original page; ties go to the smallest total offset.
Input: trained weights, the dev split and its labels. Output: <weights>/decode.json and results/tune_grid.jsonl.
"""

import argparse
import itertools
import json
import os

import numpy as np
import torch

from excise.config import DECODE
from excise.data.labels import load_labels
from excise.data.source import VIEWS, load_pages, normalize, original_id, read_split, remove
from excise.data.windows import tokenizer
from excise.eval.restore import restores_original
from excise.model.decode import PAD_EM, align_html, bias_vector, char_spans, page_logits, viterbi
from excise.model.model import ExcisionModel, device

VITERBI_BUDGET = 4_000_000  # tag positions per Viterbi batch


def score_pages(model, tok, df, entry_ids, labels):
    """Original-page scores {view: [...]} and the attack pages with label spans, with their logits."""
    rows = df.set_index("entry_id")
    original_scores = {view: [] for view in VIEWS}
    attacks = []
    for view, column in VIEWS.items():
        for e in entry_ids:
            text = normalize(rows.loc[e][column])
            logits, scores, offsets = page_logits(model, tok, text)
            if e.endswith("_benign"):
                original_scores[view].append(float(scores.max()))
            elif labels[(e, view)]:
                original = normalize(rows.loc[original_id(e)][column])
                attacks.append(dict(view=view, text=text, logits=logits, offsets=offsets, original=original))
    return original_scores, attacks


def restored(attack, path):
    spans = char_spans(attack["text"], attack["offsets"], path)
    if attack["view"] == "html":
        spans = align_html(attack["text"], spans)
    return restores_original(attack["view"], remove(attack["text"], spans), spans, attack["original"])


def search_offsets(attacks, combos):
    """Share of attack pages restored under each offset combination; pages are batched by length."""
    biases = torch.stack([bias_vector(c) for c in combos])
    hits = np.zeros(len(combos))
    attacks = sorted(attacks, key=lambda a: len(a["logits"]))
    i = 0
    while i < len(attacks):
        chunk = attacks[i : i + 16]
        T = len(chunk[-1]["logits"])
        chunk = chunk[: max(1, min(16, VITERBI_BUDGET // (T * len(combos))))]
        em = torch.stack([torch.cat([a["logits"], PAD_EM.expand(T - len(a["logits"]), 5)]) for a in chunk])
        paths = viterbi((em[:, None] + biases[None, :, None]).reshape(-1, T, 5)).reshape(len(chunk), len(combos), T)
        for p, a in enumerate(chunk):
            for g in range(len(combos)):
                hits[g] += restored(a, paths[p, g, : len(a["logits"])])
        i += len(chunk)
    return hits / len(attacks)


def tune(model, tok, df, entry_ids, labels, out_dir="results"):
    original_scores, attacks = score_pages(model, tok, df, entry_ids, labels)
    combos = list(itertools.product(DECODE["offset_grid"], repeat=3))
    rate = search_offsets(attacks, combos)
    best = max(range(len(combos)), key=lambda g: (rate[g], -sum(map(abs, combos[g]))))

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/tune_grid.jsonl", "w") as f:
        for c, r in zip(combos, rate):
            f.write(json.dumps(dict(bias_O=c[0], bias_B=c[1], bias_E=c[2], dev_restore_rate=float(r))) + "\n")
    threshold = {view: float(np.percentile(s, DECODE["threshold_percentile"])) for view, s in original_scores.items()}
    return dict(threshold=threshold, bias=list(combos[best]), dev_restore_rate=float(rate[best]))


def main():
    p = argparse.ArgumentParser(description="Set the page thresholds and tag offsets on the dev set.")
    p.add_argument("--weights", default="weights/run1")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    model = ExcisionModel.from_dir(args.weights).to(device()).eval()
    labels = load_labels("dev", args.labels, accepted_only=False)
    params = tune(model, tokenizer(args.weights), load_pages(), read_split("dev"), labels, args.out)
    json.dump(params, open(f"{args.weights}/decode.json", "w"), indent=1)
    print(params)


if __name__ == "__main__":
    main()
