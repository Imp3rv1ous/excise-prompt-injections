#!/bin/bash
# Labels, training, decoding parameters and test evaluation (report Sections 3.1-3.3 and 4.1), on one GPU machine.
# The dataset is gated on Hugging Face: run `hf auth login` once before. The frozen splits are in data/splits;
# `python -m excise.data.split` rebuilds them.
# Usage: bash scripts/train.sh   (from the repository root)
set -e
pip install -q -r requirements.txt
mkdir -p results

python -m excise.data.labels
python -m excise.data.windows
python -m excise.model.train --out weights/run1
python -m excise.model.tune --weights weights/run1

# Table 4: HTML decoded without the closing-tag alignment rule (see excise/model/decode.py).
python -m excise.eval.evaluate --weights weights/run1 --split test --without-close-tag-rule --name test
python -m excise.eval.evaluate --weights weights/run1 --split test --name test_close_tag_rule
python -m excise.eval.failure_analysis --pages results/test_pages.jsonl --out results/test_failures.md
