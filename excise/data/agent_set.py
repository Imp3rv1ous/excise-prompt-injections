"""Choose the agent evaluation set, 1,005 pairs with 201 per attack goal (report Section 3.4).

The set is drawn after training from pairs outside train / dev / test, in three steps:

  select     read the attack-page metadata of all pairs (small columns only) and sample candidate pairs per
             attack goal; propagation_to_others is left out (too few pairs)
  originals  fetch the original pages of the candidates, in the page layout excise/agent/run.py reads, so the
             reference agent can be run on them (condition R) and judged
  cut        keep candidates whose original-page run was judged a task success in every verdict file given,
             and sample the same number per attack goal

The frozen result is data/splits/agent_eval.txt; these steps only reproduce it.
"""

import argparse
import json
import random
from collections import Counter

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from excise.config import DATASET
from excise.data.source import normalize, read_split

SHARDS = [
    f"datasets/{DATASET['repo']}@{DATASET['revision']}/data/members-train-{s:04d}.parquet"
    for s in range(DATASET["shards"])
]
META = [
    "entry_id",
    "arm",
    "goal_class",
    "injection_strategy",
    "attack_type",
    "requires_action_tools",
    "benign_task",
    "malicious_goal",
    "expected_outcome",
    "attack_quotes_json",
]
ORIGINAL = ["entry_id", "content_markdown", "benign_task", "expected_outcome"]
QUOTA = {
    "unauthorized_action": 400,
    "task_disruption": 400,
    "exfiltration": 400,
    "misinformation_to_user": 400,
    "persistence_or_state_change": None,  # None: every pair of the goal
}
PER_GOAL = 214  # upper bound; the frozen set has 201 per goal, the size of the smallest goal after `cut`
SEED = 0
META_FILE = "results/agent_attack_meta.jsonl"
BLOCK = 4 * 1024 * 1024


def read_attack_meta(fs):
    rows = []
    for path in SHARDS:
        with fs.open(path, "rb", block_size=BLOCK) as f:
            table = pq.ParquetFile(f).read(columns=META)
        rows += [dict(r, pair=r["entry_id"].split("_")[0]) for r in table.to_pylist() if r["arm"] == "attack"]
    return rows


def sample_candidates(attack, seed=SEED):
    """{goal: sorted pairs}; pairs are sorted before sampling so the result depends only on the seed."""
    used = {e.split("_")[0] for name in ("train", "dev", "test") for e in read_split(name)}
    pool = [r for r in attack if r["pair"] not in used]
    rng = random.Random(seed)
    out = {}
    for goal, n in QUOTA.items():
        pairs = sorted(r["pair"] for r in pool if r["goal_class"] == goal)
        out[goal] = sorted(rng.sample(pairs, n)) if n else pairs
    return out


def write_pairs(path, chosen):
    with open(path, "w") as f:
        f.writelines(f"{p}_{arm}\n" for goal in QUOTA for p in chosen[goal] for arm in ("malicious", "benign"))


def select(args):
    attack = read_attack_meta(HfFileSystem())
    with open(META_FILE, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in attack)
    chosen = sample_candidates(attack)
    write_pairs(args.out, chosen)
    print({goal: len(pairs) for goal, pairs in chosen.items()})


def fetch_originals(fs, wanted):
    """Read only the parquet row groups that hold a wanted original page."""
    rows = {}
    for path in SHARDS:
        with fs.open(path, "rb", block_size=BLOCK) as f:
            pf = pq.ParquetFile(f)
            ids = pf.read(columns=["entry_id"]).column(0).to_pylist()
            start = 0
            for g in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                if wanted & set(ids[start : start + n]):
                    table = pf.read_row_group(g, columns=ORIGINAL)
                    rows.update({r["entry_id"]: r for r in table.to_pylist() if r["entry_id"] in wanted})
                start += n
    return rows


def originals(args):
    ids = [line.strip() for line in open(args.candidates)]
    meta = {r["pair"]: r for r in map(json.loads, open(META_FILE))}
    rows = fetch_originals(HfFileSystem(), {e for e in ids if e.endswith("_benign")})
    with open(args.out, "w") as out:
        for e in ids[::2]:
            pair = e.split("_")[0]
            a, c = meta[pair], rows[f"{pair}_benign"]
            attack = dict(a, arm="attack", view="markdown", detected=False, text=None, restored=None)
            original = dict(
                entry_id=c["entry_id"],
                pair=pair,
                arm="control",
                view="markdown",
                detected=False,
                text=normalize(c["content_markdown"]),
                benign_task=c["benign_task"],
                expected_outcome=c["expected_outcome"],
            )
            out.write(json.dumps(attack, ensure_ascii=False) + "\n")
            out.write(json.dumps(original, ensure_ascii=False) + "\n")


def cut(args):
    verdicts = list(map(json.loads, open(args.verdicts[0])))
    passed = set.intersection(
        *[{v["pair"] for v in map(json.loads, open(path)) if v["task_success"] is True} for path in args.verdicts]
    )
    ok = {
        goal: sorted(v["pair"] for v in verdicts if v["goal_class"] == goal and v["pair"] in passed) for goal in QUOTA
    }
    k = min(PER_GOAL, min(map(len, ok.values())))
    rng = random.Random(SEED)
    write_pairs(args.out, {goal: sorted(rng.sample(pairs, k)) for goal, pairs in ok.items()})
    print("pairs per goal", k, "| candidates per goal", Counter(v["goal_class"] for v in verdicts))


def main():
    p = argparse.ArgumentParser(description="Choose the agent evaluation set.")
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("select", help="sample candidate pairs per attack goal")
    s.add_argument("--out", default="data/splits/agent_candidates.txt")
    s = sub.add_parser("originals", help="fetch the candidates' original pages for the screening run")
    s.add_argument("--candidates", default="data/splits/agent_candidates.txt")
    s.add_argument("--out", default="results/agent_candidate_pages.jsonl")
    s = sub.add_parser("cut", help="keep pairs whose original page was judged a task success")
    s.add_argument("--verdicts", nargs="+", default=["results/agent_candidate_verdicts.jsonl"])
    s.add_argument("--out", default="data/splits/agent_eval.txt")
    args = p.parse_args()
    {"select": select, "originals": originals, "cut": cut}[args.step](args)


if __name__ == "__main__":
    main()
