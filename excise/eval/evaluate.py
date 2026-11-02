"""Removal performance: page-level detection and character-level localization (report Sections 3.5 and 4.1).

Per view: detection F1 / precision / recall / accuracy and false-positive rate (a page is positive when the diff
finds an injection in it), the share of attack pages restored to their original page, the share of original pages
the model changed, character-level precision / recall / F1, and restore rates by injection placement.

Two modes:
  --split test        run the model over a split and write <out>/<name>_pages.jsonl plus the metrics
  --pages FILE        metrics of an existing pages file, e.g. the agent evaluation set from
                      excise/agent/prepare_pages.py, optionally per group (--by goal_class)
--without-close-tag-rule decodes HTML without the last alignment rule of excise/model/decode.py; this gives the
report's HTML numbers (Table 4).
"""

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from excise.data.labels import ACCEPTED, label_pair
from excise.data.source import VIEWS, load_pages, normalize, original_id, read_split
from excise.eval.char_metrics import char_metrics
from excise.eval.restore import restores_original
from excise.model.infer import Exciser


def predict_pages(exciser, df, entry_ids, keep_text=False):
    """One row per page and view. Label spans come from the diff with the original page (excise/data/labels.py).
    keep_text adds the normalized and the excised Markdown page, for the agent runs."""
    rows = df.set_index("entry_id", drop=False)
    for e in entry_ids:
        page, original = rows.loc[e], rows.loc[original_id(e)]
        goal = rows.loc[e.replace("benign", "malicious")].goal_class
        for view, column in VIEWS.items():
            text, original_text = normalize(page[column]), normalize(original[column])
            r = exciser.run(text, view)
            gold, cls = label_pair(text, original_text, view) if page.arm == "attack" else ([], None)
            row = dict(
                entry_id=e,
                pair=page.pair,
                arm=page.arm,
                view=view,
                goal_class=goal,
                injection_strategy=page.injection_strategy,
                score=r["score"],
                detected=r["score"] >= exciser.params["threshold"][view],
                pred_spans=r["spans"],
                gold_spans=gold,
                positive=bool(gold),
                diff_class=cls,
                accepted=cls in ACCEPTED[view],
                modified=bool(r["spans"]),
                restored=restores_original(view, r["cleaned"], r["spans"], original_text),
            )
            if keep_text and view == "markdown":
                row.update(text=text, cleaned=r["cleaned"])
            yield row


def view_metrics(g):
    """Metrics of the rows of one view."""
    p, r, f, _ = precision_recall_fscore_support(g.positive, g.detected, average="binary", zero_division=0)
    attack = g[g.positive]
    m = dict(
        pairs=int((g.arm == "attack").sum()),
        f1=f,
        precision=p,
        recall=r,
        accuracy=accuracy_score(g.positive, g.detected),
        false_positive_rate=float(g.detected[~g.positive].mean()),
        restore_rate=float(attack.restored.mean()),
        restore_rate_accepted=float(attack[attack.accepted].restored.mean()),
        original_modified_rate=float(g[g.arm == "control"].modified.mean()),
    )
    m.update({f"char_{k}": v for k, v in char_metrics(g[g.arm == "attack"].to_dict("records")).items()})
    m["by_strategy"] = attack.groupby("injection_strategy").restored.agg(["mean", "size"]).to_dict("index")
    return m


def metrics(pages, by=None):
    """{view: metrics}, or {view: {group: metrics, "all": metrics}} when grouped by a column."""
    t = pd.DataFrame(pages)
    out = {}
    for view, g in t.groupby("view"):
        out[view] = view_metrics(g)
        if by:
            out[view] = {**{k: view_metrics(gg) for k, gg in g.groupby(by)}, "all": out[view]}
    return out


COLUMNS = [
    ("pairs", "Pairs", "d"),
    ("f1", "Page F1", ".4f"),
    ("precision", "Precision", ".4f"),
    ("recall", "Recall", ".4f"),
    ("false_positive_rate", "FPR", ".4f"),
    ("restore_rate", "Restored", ".4f"),
    ("original_modified_rate", "Originals changed", ".4f"),
    ("char_precision", "Char P", ".4f"),
    ("char_recall", "Char R", ".4f"),
    ("char_f1", "Char F1", ".4f"),
]


def table(m, by=None):
    first = ["View"] + ([by] if by else [])
    lines = ["| " + " | ".join(first + [c[1] for c in COLUMNS]) + " |", "|---" * (len(first) + len(COLUMNS)) + "|"]
    for view, v in m.items():
        for group, x in v.items() if by else [(None, v)]:
            cells = [view] + ([group] if by else []) + [format(x[k], fmt) for k, _, fmt in COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
    if not by:
        for view, v in m.items():
            lines += ["", f"{view}: restore rate by injection placement", ""]
            lines += [f"- {s}: {r['mean']:.3f} (n={r['size']})" for s, r in sorted(v["by_strategy"].items())]
    return "\n".join(lines) + "\n"


def write(pages, m, out, name, by=None):
    os.makedirs(out, exist_ok=True)
    if pages is not None:
        with open(f"{out}/{name}_pages.jsonl", "w") as f:
            f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in pages)
    json.dump(m, open(f"{out}/{name}_metrics.json", "w"), indent=1)
    open(f"{out}/{name}_metrics.md", "w").write(table(m, by))


def main():
    p = argparse.ArgumentParser(description="Detection and localization metrics.")
    p.add_argument("--weights", default="weights/run1")
    p.add_argument("--split", default="test")
    p.add_argument("--pages", default=None, help="score an existing pages file instead of running the model")
    p.add_argument("--by", default=None, help="group column for --pages, e.g. goal_class")
    p.add_argument("--without-close-tag-rule", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--name", default=None, help="output file prefix (default: split name or pages file name)")
    args = p.parse_args()

    if args.pages:
        pages = list(map(json.loads, open(args.pages)))
        m = metrics(pages, args.by)
        write(None, m, args.out, args.name or os.path.basename(args.pages).removesuffix("_pages.jsonl"), args.by)
    else:
        exciser = Exciser(args.weights)
        exciser.params["close_tag_rule"] = not args.without_close_tag_rule
        pages = list(predict_pages(exciser, load_pages(), read_split(args.split)))
        m = metrics(pages)
        write(pages, m, args.out, args.name or args.split)
    print(table(m, args.by))


if __name__ == "__main__":
    main()
