"""Failure cases of excision; a diagnostic that the report does not use.

Per view: the attack pages with label spans the model missed (page score below threshold), and the flagged attack
pages that excision did not restore, split into four groups that compare the deleted characters with the label
characters. The page with the most over-deleted characters is reported with the character precision computed
with and without it.
Input: a pages file from excise/eval/evaluate.py or excise/agent/prepare_pages.py. Output: a Markdown report.
"""

import argparse
import json
from collections import Counter

from excise.data.source import VIEWS
from excise.eval.char_metrics import char_metrics

GROUPS = {
    (False, True): "all injected characters deleted, extra characters deleted",
    (True, False): "some injected characters kept, no extra characters deleted",
    (True, True): "some injected characters kept and extra characters deleted",
    (False, False): "same characters as the labels, recovery criterion not met",
}


def chars(spans):
    return {i for s, e in spans for i in range(s, e)}


def extra(p):
    """Characters deleted outside the label spans."""
    return len(chars(p["pred_spans"]) - chars(p["gold_spans"]))


def view_report(pages, view):
    attack = [p for p in pages if p["view"] == view and p["arm"] == "attack"]
    missed = [p for p in attack if p["positive"] and not p["detected"]]
    flagged = [p for p in attack if p["detected"] and not p["restored"]]
    groups = Counter()
    for p in flagged:
        gold, pred = chars(p["gold_spans"]), chars(p["pred_spans"])
        groups[(bool(gold - pred), bool(pred - gold))] += 1

    worst = max(attack, key=extra)
    rest = [p for p in attack if p is not worst]
    lines = [
        f"## {view}",
        "",
        f"Attack pages {len(attack)}, with label spans {sum(p['positive'] for p in attack)}, missed {len(missed)}; "
        f"flagged but not restored {len(flagged)} (of which {sum(not p['pred_spans'] for p in flagged)} "
        "with no predicted span).",
        "",
        "| Group | Pages |",
        "|---|---|",
    ]
    lines += [f"| {name} | {groups[k]} |" for k, name in GROUPS.items()]
    lines += [
        "",
        f"Most over-deleted page: {worst['entry_id']}, {extra(worst):,} extra characters. Character precision "
        f"{char_metrics(attack)['precision']:.3f} with it, {char_metrics(rest)['precision']:.3f} without it.",
        "",
    ]
    return lines


def main():
    p = argparse.ArgumentParser(description="Split unrestored attack pages into failure groups.")
    p.add_argument("--pages", default="results/test_pages.jsonl")
    p.add_argument("--out", default="results/test_failures.md")
    args = p.parse_args()

    pages = list(map(json.loads, open(args.pages)))
    lines = [f"# Excision failures: {args.pages}", ""]
    for view in VIEWS:
        if any(p["view"] == view for p in pages):
            lines += view_report(pages, view)
    open(args.out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
