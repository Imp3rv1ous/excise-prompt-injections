"""Run the excision model over the agent evaluation set (report Sections 3.4 and 4.1); needs a GPU.

Both pages of every pair, both views, one row per page and view, in the layout of excise/eval/evaluate.py.
Markdown rows also carry the normalized page (`text`) and the excised page (`cleaned`), and every row carries
the task and attack fields the agent runs and the judge need, so later steps need nothing else.
Input: weights and data/splits/agent_eval.txt. Output: results/agent_pages.jsonl.
"""

import argparse
import json

from excise.data.source import load_pages
from excise.eval.evaluate import predict_pages
from excise.model.infer import Exciser

ATTACK_FIELDS = ["benign_task", "expected_outcome", "malicious_goal", "attack_quotes_json", "attack_type"]
ORIGINAL_FIELDS = ["benign_task", "expected_outcome"]


def main():
    p = argparse.ArgumentParser(description="Excision model outputs on the agent evaluation set.")
    p.add_argument("--weights", default="weights/run1")
    p.add_argument("--split", default="data/splits/agent_eval.txt")
    p.add_argument("--out", default="results/agent_pages.jsonl")
    args = p.parse_args()

    ids = [line.strip() for line in open(args.split)]
    df = load_pages()
    rows = df.set_index("entry_id", drop=False)
    exciser = Exciser(args.weights)
    with open(args.out, "w") as f:
        for r in predict_pages(exciser, df, ids, keep_text=True):
            page = rows.loc[r["entry_id"]]
            r.update({k: page[k] for k in (ATTACK_FIELDS if r["arm"] == "attack" else ORIGINAL_FIELDS)})
            f.write(json.dumps(r, ensure_ascii=False, default=lambda o: o.item()) + "\n")  # numpy scalars
            print(r["entry_id"], r["view"], f"score {r['score']:.4f}", "flagged" if r["detected"] else "")


if __name__ == "__main__":
    main()
