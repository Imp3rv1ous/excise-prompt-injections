"""Train / dev / test split by injection family (report Section 3.1).

A family is the longest injection fragment after whitespace and case folding; all pairs of a family go to one
set, so no injection text is shared across sets. Sizes and seed are in configs/excise.yaml.
Output: data/splits/{train,dev,test,latency}.txt, two entry ids per pair (attack page, then original page).
The agent evaluation set (data/splits/agent_eval.txt) is chosen later by excise/data/agent_set.py.
"""

import argparse
import json
import os
import random
import re

from excise.config import SPLITS
from excise.data.source import load_pages


def family(fragments_json):
    return re.sub(r"\s+", " ", max(json.loads(fragments_json), key=len)).strip().lower()


def split_pairs(attack, sizes, seed):
    """{set name: [pair, ...]} from the attack rows; whole families are assigned at random, weighted by room left."""
    groups = attack.groupby(attack.injection_text_fragments_json.map(family)).pair.apply(sorted)
    families = sorted(groups.tolist())
    rng = random.Random(seed)
    rng.shuffle(families)

    left = dict(sizes)
    out = {name: [] for name in sizes}
    for pairs in families:
        fits = [name for name in sizes if len(pairs) <= left[name]]
        if not fits:
            continue
        name = rng.choices(fits, weights=[left[n] for n in fits])[0]
        out[name] += pairs
        left[name] -= len(pairs)
    assert not any(left.values()), left
    return out


def write_splits(sets, out_dir):
    sets = dict(sets)
    sets["latency"] = sets.pop("latency_pool")[SPLITS["latency_skip"] :]
    os.makedirs(out_dir, exist_ok=True)
    for name, pairs in sets.items():
        with open(f"{out_dir}/{name}.txt", "w") as f:
            f.writelines(f"{p}_{arm}\n" for p in pairs for arm in ("malicious", "benign"))
    return {name: len(pairs) for name, pairs in sets.items()}


def main():
    p = argparse.ArgumentParser(description="Split BrowseSafe++ pairs by injection family.")
    p.add_argument("--out", default="data/splits")
    args = p.parse_args()

    df = load_pages()
    sets = split_pairs(df[df.arm == "attack"], SPLITS["sizes"], SPLITS["seed"])
    print(write_splits(sets, args.out))


if __name__ == "__main__":
    main()
