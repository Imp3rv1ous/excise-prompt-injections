#!/bin/bash
# Per-page latency (report Section 4.3, Table 8). Run on an otherwise idle GPU after scripts/train.sh.
# Usage: bash scripts/latency.sh
set -e
mkdir -p results
nvidia-smi --query-gpu=name,memory.total,power.limit,power.max_limit --format=csv
python -m excise.latency.time_pages --weights weights/run1 --out results/latency.jsonl
python -m excise.latency.summarize --latency results/latency.jsonl --out results/latency_table.md
