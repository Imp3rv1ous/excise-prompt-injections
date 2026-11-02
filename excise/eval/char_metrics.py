"""Character-level precision, recall and F1 of predicted spans against label spans (report Section 3.5).

Counts are summed over all attack pages before dividing (micro average); the mean of per-page F1 is also given.
"""


def overlap(a, b):
    return sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def length(spans):
    return sum(e - s for s, e in spans)


def char_metrics(pages):
    """pages: attack-page rows with pred_spans and gold_spans."""
    hit = pred = gold = 0
    page_f1 = []
    for r in pages:
        h, p, g = overlap(r["pred_spans"], r["gold_spans"]), length(r["pred_spans"]), length(r["gold_spans"])
        hit, pred, gold = hit + h, pred + p, gold + g
        page_f1.append(2 * h / (p + g) if h else 0.0)
    precision, recall = hit / max(pred, 1), hit / max(gold, 1)
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    return dict(precision=precision, recall=recall, f1=f1, page_f1_mean=sum(page_f1) / max(len(page_f1), 1))
