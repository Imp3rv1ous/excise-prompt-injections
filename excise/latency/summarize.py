"""Latency table (report Section 4.3 and Table 8).

Per view: median and p90 of the model pass, of excision (scores to deleted spans) and of their sum, in ms per page,
plus medians by page length in tokens. Excision is the time removal adds over blocking.
Input: results/latency.jsonl. Output: results/latency_table.md.
"""

import argparse
import json

import pandas as pd

LENGTHS = [(0, 4000, "< 4k"), (4000, 16000, "4k-16k"), (16000, 32000, "16k-32k"), (32000, 10**9, "> 32k")]


def median_p90(series):
    return f"{series.median():.1f} / {series.quantile(0.9):.1f}"


def summarize(path):
    lines = list(map(json.loads, open(path)))
    head, df = lines[0], pd.DataFrame(lines[1:])
    df["total_ms"] = df.forward_ms + df.decode_ms
    out = [
        "# Latency",
        "",
        f"{head['gpu']}, driver {head['driver']}, power limit {head['power_limit']}, bf16 {head['tflops_bf16']} "
        f"TFLOPS, {head['date']}; {len(df) // 2} attack pages x 2 views, batch 1, {head['warmup']} warm-up pages, "
        f"median of {head['repeats']} runs per page.",
        "",
        "| View | Model pass ms (median / p90) | Excision ms (median / p90) | Total ms (median / p90) |",
        "|---|---|---|---|",
    ]
    for view in ("html", "markdown"):
        d = df[df.view == view]
        out.append(f"| {view} | {median_p90(d.forward_ms)} | {median_p90(d.decode_ms)} | {median_p90(d.total_ms)} |")

    out += ["", "## Medians by page length (ms)", "", "| View | Tokens | Pages | Model pass | Excision | Total |"]
    out.append("|---|---|---|---|---|---|")
    for view in ("html", "markdown"):
        for lo, hi, name in LENGTHS:
            d = df[(df.view == view) & (df.n_tokens >= lo) & (df.n_tokens < hi)]
            if len(d):
                out.append(
                    f"| {view} | {name} | {len(d)} | {d.forward_ms.median():.1f} | {d.decode_ms.median():.1f} "
                    f"| {d.total_ms.median():.1f} |"
                )
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description="Summarize per-page latency.")
    p.add_argument("--latency", default="results/latency.jsonl")
    p.add_argument("--out", default="results/latency_table.md")
    args = p.parse_args()

    text = summarize(args.latency)
    open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
