"""Per-page latency of the excision model (report Sections 3.5 and 4.3); run with the GPU otherwise idle.

Attack pages of data/splits/latency.txt, both views, one page at a time.
  model pass: from the input string to the averaged tag logits and window scores (tokenization included)
  excision:   from the logits to the excised string (Viterbi, span edges, HTML alignment, deletion), always run
20 pages warm up, then every page is timed 3 times and the median kept.
Output: results/latency.jsonl; the first line records the GPU, driver, power limit and a bf16 matmul benchmark.
"""

import argparse
import datetime
import json
import subprocess
import time

import numba
import torch

from excise.data.source import VIEWS, load_pages, normalize, remove
from excise.data.windows import window_starts
from excise.model.decode import decode_page, page_logits
from excise.model.infer import Exciser

WARMUP, REPEATS = 20, 3
# One page at a time leaves the parallel Viterbi a single row; on many-core hosts the thread fan-out alone
# cost 60-100 ms per page, so excision runs on one thread.
numba.set_num_threads(1)


def bf16_tflops(seconds=30, n=8192):
    """Dense bf16 matmul throughput, to tell a power-capped GPU from a normal one."""
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0, flops = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        a @ b
        flops += 2 * n**3
    torch.cuda.synchronize()
    return round(flops / (time.perf_counter() - t0) / 1e12, 1)


def time_page(exciser, text, view):
    """(model pass ms, excision ms, tokens, windows, page score) for one timed pass."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits, scores, offsets = page_logits(exciser.model, exciser.tok, text)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    remove(text, decode_page(logits, offsets, text, view, exciser.params))
    t2 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, len(offsets), len(window_starts(len(offsets))), float(scores.max())


def header(args):
    def query(field):
        out = subprocess.check_output(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"])
        return out.decode().strip()

    return dict(
        gpu=query("name"),
        driver=query("driver_version"),
        power_limit=query("power.limit"),
        tflops_bf16=bf16_tflops(),
        date=str(datetime.date.today()),
        warmup=WARMUP,
        repeats=REPEATS,
        weights=args.weights,
        numba_threads=numba.get_num_threads(),
    )


def main():
    p = argparse.ArgumentParser(description="Time the excision model page by page.")
    p.add_argument("--weights", default="weights/run1")
    p.add_argument("--split", default="data/splits/latency.txt")
    p.add_argument("--pairs", type=int, default=None, help="only the first N pairs")
    p.add_argument("--out", default="results/latency.jsonl")
    args = p.parse_args()

    ids = [e for e in (line.strip() for line in open(args.split)) if e.endswith("_malicious")][: args.pairs]
    df = load_pages().set_index("entry_id", drop=False)
    exciser = Exciser(args.weights)
    pages = [(e, view, normalize(df.loc[e][column])) for e in ids for view, column in VIEWS.items()]

    for e, view, text in pages[:WARMUP]:
        time_page(exciser, text, view)

    with open(args.out, "w") as f:
        f.write(json.dumps(header(args)) + "\n")
        for e, view, text in pages:
            runs = [time_page(exciser, text, view) for _ in range(REPEATS)]
            row = dict(
                entry_id=e,
                view=view,
                n_tokens=runs[0][2],
                n_windows=runs[0][3],
                forward_ms=round(sorted(r[0] for r in runs)[1], 3),
                decode_ms=round(sorted(r[1] for r in runs)[1], 3),
                detected=runs[0][4] >= exciser.params["threshold"][view],
            )
            f.write(json.dumps(row) + "\n")
            print(row)


if __name__ == "__main__":
    main()
