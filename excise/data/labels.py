"""Injection labels from the difference between an attack page and its original page (report Section 3.1).

A character-level Myers diff aligns the two pages; the label spans are the attack-page characters left unaligned.
Each pair gets a diff class that says how cleanly the pages align, and only pairs in the accepted classes
give training labels:

  empty      no difference in this view (the injection is not visible here); accepted, all-negative page
  pure       the diff only inserts text; accepted
  canon      Markdown only: the diff also deletes text, but deleting the label spans restores the original
             page under the recovery criterion (renumbered node ids, whitespace); accepted
  tail       Markdown only: as canon, but the injection pushed the end of the page past the truncation point;
             accepted
  tiny_ins   Markdown only: inserted pieces that do not restore the original page; rejected
  ws_del / shell_del / small_del / large_del
             the diff deletes whitespace / only HTML tags / at most 8 characters / more text; rejected

In the Markdown view only inserted runs longer than 3 characters outside the truncation notice are labels,
because the conversion also renumbers lists and links.

Input: the dataset and a split. Output: data/labels/<split>.jsonl (one record per attack page and view)
and data/labels/<split>_report.md.
"""

import argparse
import json
import os
import re

import fast_diff_match_patch as fdmp
import pandas as pd

from excise.data.source import VIEWS, load_pages, normalize, original_id, read_split, remove
from excise.eval.restore import canon, restores_original, strip_tail

ACCEPTED = {"html": {"pure", "empty"}, "markdown": {"pure", "empty", "canon", "tail"}}
MIN_MARKDOWN_RUN = 3


def diff_ops(original, attack):
    """Myers diff as [(op, i, j, n)]: i is the position in the original page, j in the attack page."""
    i = j = 0
    ops = []
    for op, n in fdmp.diff(original, attack, timelimit=0, checklines=False, cleanup=None):
        ops.append((op, i, j, n))
        i += n * (op != "+")
        j += n * (op != "-")
    return ops


def diff_spans(original, attack, view):
    """Label spans in the attack page and the texts the diff deletes from the original page."""
    ops = diff_ops(original, attack)
    cut = len(strip_tail(attack))
    spans = [(j, j + n) for op, i, j, n in ops if op == "+" and (view == "html" or (n > MIN_MARKDOWN_RUN and j < cut))]
    deleted = [original[i : i + n] for op, i, j, n in ops if op == "-"]
    return spans, deleted


def diff_class(spans, deleted, view, attack, original):
    d = "".join(deleted)
    if not spans and not d:
        return "empty"

    if view == "markdown":
        cleaned = remove(attack, spans)
        if restores_original(view, cleaned, spans, original):
            if canon(cleaned) != canon(original):
                return "tail"
            return "canon" if d else "pure"
        if not d:
            return "tiny_ins"
    elif not d:
        return "pure"

    if not d.strip():
        return "ws_del"
    if view == "html" and not re.sub(r"<[^>]*>|\s", "", d):
        return "shell_del"
    return "small_del" if len(d.strip()) <= 8 else "large_del"


def label_pair(attack_text, original_text, view):
    """(label spans, diff class) for one pair in one view; both texts already normalized."""
    spans, deleted = diff_spans(original_text, attack_text, view)
    return spans, diff_class(spans, deleted, view, attack_text, original_text)


def fragment_key(text):
    return re.sub(r"[\\\s]", "", text)


def label_record(attack, original, view):
    """One labels.jsonl record. The benchmark's injection fragments (HTML) or quotes (Markdown) are checked
    against the label spans as a sanity count."""
    a, c = normalize(attack[VIEWS[view]]), normalize(original[VIEWS[view]])
    spans, deleted = diff_spans(c, a, view)
    refs = attack.injection_text_fragments_json if view == "html" else attack.attack_quotes_json
    frags = [fragment_key(f) for f in json.loads(refs)]
    covered = sum(any(f in fragment_key(a[s:e]) for s, e in spans) for f in frags)
    return dict(
        entry_id=attack.entry_id,
        view=view,
        diff_class=diff_class(spans, deleted, view, a, c),
        spans=spans,
        inserted_chars=sum(e - s for s, e in spans),
        deleted_chars=len("".join(deleted)),
        n_fragments=len(frags),
        n_fragments_covered=covered,
        injection_strategy=attack.injection_strategy,
    )


def label_split(df, entry_ids):
    rows = df.set_index("entry_id", drop=False)
    attacks = [e for e in entry_ids if e.endswith("_malicious")]
    return [label_record(rows.loc[e], rows.loc[original_id(e)], view) for view in VIEWS for e in attacks]


def load_labels(name, label_dir="data/labels", accepted_only=True):
    """Label spans {(entry_id, view): spans}, of the accepted pairs only unless accepted_only is False."""
    records = map(json.loads, open(f"{label_dir}/{name}.jsonl"))
    return {
        (r["entry_id"], r["view"]): [tuple(s) for s in r["spans"]]
        for r in records
        if r["diff_class"] in ACCEPTED[r["view"]] or not accepted_only
    }


def report(records):
    t = pd.DataFrame(records)
    t["coverage"] = t.n_fragments_covered / t.n_fragments
    t["accepted"] = [c in ACCEPTED[v] for c, v in zip(t.diff_class, t.view)]
    lines = []
    for view, g in t.groupby("view"):
        lines += [
            f"## {view}: {len(g)} pairs, {g.accepted.sum()} accepted ({g.accepted.mean():.1%})",
            "",
            "| Diff class | Pairs | Share | Fragments inside label spans | Median inserted chars |",
            "|---|---|---|---|---|",
        ]
        for cls, gc in g.groupby("diff_class"):
            lines.append(
                f"| {cls} | {len(gc)} | {len(gc) / len(g):.1%} | {gc.coverage.mean():.1%} "
                f"| {gc.inserted_chars.median():.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Diff labels and acceptance for the train, dev and test splits.")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--out", default="data/labels")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_pages()
    for name in args.splits:
        records = label_split(df, read_split(name))
        with open(f"{args.out}/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        open(f"{args.out}/{name}_report.md", "w").write(f"# {name}\n\n" + report(records))
        print(name, len(records), "records")


if __name__ == "__main__":
    main()
