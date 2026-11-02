"""Inference on one page (report Section 3.3).

Input: trained weights, a view and a page file (HTML or Markdown). Output: JSON with the page score, the
predicted injection spans and the excised page.
"""

import argparse
import json

from excise.data.windows import tokenizer
from excise.model.decode import excise
from excise.model.model import ExcisionModel, device


class Exciser:
    def __init__(self, weights):
        self.model = ExcisionModel.from_dir(weights).to(device()).eval()
        self.tok = tokenizer(weights)
        self.params = json.load(open(f"{weights}/decode.json"))

    def run(self, text, view):
        return excise(self.model, self.tok, text, view, self.params)


def main():
    p = argparse.ArgumentParser(description="Locate and delete injections in one page.")
    p.add_argument("--weights", default="weights/run1")
    p.add_argument("--view", choices=["html", "markdown"], required=True)
    p.add_argument("--input", required=True, help="path of the page file")
    args = p.parse_args()
    result = Exciser(args.weights).run(open(args.input).read(), args.view)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
