# Excise, Don't Block

A 0.6B encoder locates prompt injections in web pages and deletes them before a browsing agent reads the page.
Its training labels need no annotation: in BrowseSafe++ every attack page comes with its original page, and the
difference between the two marks the injection.

Paper: [report.pdf](paper/report.pdf)

## Report sections and code

| Report section | Module | What it does |
|---|---|---|
| 3.1 Model: From Paired Pages to Labels | `excise/data/source.py`, `split.py`, `labels.py` | pinned dataset, character-diff labels, split by injection text (Table 3) |
| 3.2 Model: Architecture and Training | `excise/data/windows.py`, `excise/model/model.py`, `train.py` | windows and BIOES token tags, backbone and two heads, loss, optimizer |
| 3.3 Model: From Scores to an Excised Page | `excise/model/decode.py`, `tune.py`, `infer.py` | page score, constrained decoding, HTML alignment, dev-set thresholds and offsets; one page in, excised page out |
| 3.4 Evaluation: Pages, Agents and Conditions | `excise/data/agent_set.py`, `excise/agent/` (`run.py`, `tools.py`, `prompts.py`, `judge.py`, `framework/`) | agent evaluation set, tool-calling agent, five simulated tools, four conditions, judge |
| 3.5 Evaluation: Metrics | `excise/eval/char_metrics.py`, `restore.py`, `evaluate.py`, `excise/agent/summarize.py`, `excise/latency/summarize.py` | page and character metrics, exact-excision criterion, Wilson intervals, latency medians and p90 |
| 4.1 RQ1 | `excise/eval/evaluate.py`, `excise/agent/prepare_pages.py` | Table 4 (test set), Table 5 (agent evaluation set), exact excision by placement (Figure 5) |
| 4.2 RQ2 | `excise/agent/run.py`, `judge.py`, `summarize.py` | Tables 6 and 7: attack success and task completion per agent, condition and attack goal |
| 4.3 RQ3 | `excise/latency/time_pages.py`, `summarize.py` | Table 8 and Figure 8: time per page from input page to excised page, split into model pass and span excision |
| Appendix A | `configs/excise.yaml`, `excise/model/train.py`, `tune.py` | data and backbone revisions, windows, loss, optimizer, decoding search; training log (Figure 7) |
| Appendix C | `excise/agent/prompts.py` | agent and judge prompts |

`excise/agent/framework/` is a minimal tool-calling agent framework: tool abstraction, execution context, LiteLLM
client and the agent loop. `excise/eval/failure_analysis.py` is a diagnostic that the report does not use.

## Install

Python 3.10 or newer (the experiments ran on 3.12).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

BrowseSafe++ is gated on Hugging Face: accept its terms, then `hf auth login`. The dataset (pinned commit) and the
backbone (pinned revision) download on first use.

## API keys

Only the agent experiment calls APIs. Copy `.env.example` to `.env` at the repository root and fill in the keys;
`excise/agent/run.py` and `judge.py` load the file with python-dotenv, and LiteLLM reads the variables
(`OPENROUTER_API_KEY` for the four open-weight agents, `OPENAI_API_KEY` for GPT-6 Luna, `DEEPSEEK_API_KEY` for the
judge). `.env` is in `.gitignore`.

## Run order

All commands run from the repository root. `scripts/` wraps steps 2-6 for a GPU machine.

1. **Data.** The frozen splits are in `data/splits/` (`train`, `dev`, `test`, `agent_eval`, `latency`; entry ids,
   attack page then original page). To rebuild `train`, `dev`, `test` and `latency`: `python -m excise.data.split`
2. **Labels and windows.**
   `python -m excise.data.labels && python -m excise.data.windows`
3. **Training.**
   `python -m excise.model.train --out weights/run1 && python -m excise.model.tune --weights weights/run1`
4. **Localization test (Table 4, Figure 5).**
   `python -m excise.eval.evaluate --split test --without-close-tag-rule`
5. **Agent experiment, four conditions (Tables 5-7, Figure 6).** On a GPU: `bash scripts/agent.sh pages`. Then, with
   API keys (this spends money): `bash scripts/agent.sh run`, which runs `python -m excise.agent.run`,
   `python -m excise.agent.judge` and `python -m excise.agent.summarize`. The report's runs were made one process per
   agent (`--agents`), condition A with `--concurrency 10` and B, C and R with 20, and the run files were
   concatenated before judging.
6. **Latency (Table 8, Figure 8).** On an idle GPU: `bash scripts/latency.sh`. The RTX 4090 rows of Table 8 time 100
   development-set attack pages per view:
   `python -m excise.latency.time_pages --split data/splits/dev.txt --pairs 100 --out results/latency_4090.jsonl`

Every entry point takes `--help`. Outputs go to `results/`, weights to `weights/`.

## Notes

- **HTML decoding rule.** `excise/model/decode.py` ends its HTML alignment with a rule that keeps a closing tag at
  the start of a span. The report's HTML numbers were decoded without it (`--without-close-tag-rule`, 83.9% exact
  excision on the test set); with it the same pages give 94.7%. The agent and latency runs use the full rule set.
- **Agent evaluation set.** 1,005 pairs outside train / dev / test, 201 per attack goal. `excise/data/agent_set.py`
  documents how it was drawn; `data/splits/agent_eval.txt` is the frozen result.
- **Latency pages.** The 900 attack pages of `data/splits/latency.txt` come from pairs outside train / dev / test,
  drawn by the split before the agent evaluation set was chosen.

## Not in this repository

Trained weights, results, labels, cached windows and dataset files are not included (`.gitignore`); the steps above
regenerate them. Training writes `weights/run1/` (`model.safetensors`, `config.json`, the tokenizer files and
`train_log.jsonl`) and tuning adds `decode.json` (page thresholds and tag offsets). The tables and figures of the
report were formatted from `results/*_metrics.json`, `results/agent_table.md`, `results/latency_table.md`,
`results/latency.jsonl` and `weights/run1/train_log.jsonl`; the formatting scripts are not included.

## License

MIT, see [LICENSE](LICENSE).
