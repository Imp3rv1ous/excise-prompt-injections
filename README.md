# Excise, Don't Block

A browser agent reads web pages as text, and a page author can plant an instruction that turns the agent against
its user. The defenses in use change the page before the agent reads it, at a price: blocking a flagged page
discards the content the task needs, and rewriting the page with a large language model adds a model call for
every page. This repository trains a 0.6B encoder that locates the injected text and deletes it, leaving the rest
of the page untouched. Its training labels need no annotation: in BrowseSafe++ every attack page comes with its
original page, and the difference between the two marks the injection.

Paper: [excise-prompt-injections.pdf](paper/excise-prompt-injections.pdf) · Poster: [excise-prompt-injections-poster.pdf](paper/excise-prompt-injections-poster.pdf)

[![Poster](paper/excise-prompt-injections-poster.png)](paper/excise-prompt-injections-poster.pdf)

## How it works

**Labels from page pairs.** The two pages of a pair are compared character by character, and every character the
attack page has and the original page lacks belongs to the injection. Both pages of a pair stay in one data set,
and so do pairs that share an injection text, so no injection leaks from training into testing.

**Model and decoding.** A page is cut into overlapping windows of up to 4,096 tokens, and a bidirectional encoder
reads each window once. A token head tags every token as the start, inside, end or whole of an injected span, or as
outside any span, and a window head scores the window. The page score is the highest window score, and only pages
above a threshold are excised. A constrained decoder picks the best tag sequence that forms valid spans, and those
spans are deleted.

**Testing with agents.** Five agents, four smaller open-weight models and GPT-6 Luna, carry out the user task on each
page under four conditions: the attack page unchanged, the attack page after excision, a placeholder sentence in
place of the blocked page, and the original page. A judge model reads the tool calls and the final answer of every
run and decides whether the agent followed the injection and whether it completed the task.

## Results

- **Excision restores the original page** on 90.3% of Markdown and 83.9% of HTML test pages.
- **Agents stop following the injection without losing the task.** After excision, each agent's attack success
  rate falls back to about its level on the original page, and its task completion rate stays within 2 percentage
  points of that level. Blocking leaves almost no task completed.
- **The whole defense takes well under a second per page.** The time runs from the input page to the excised page,
  median over pages.

| GPU | HTML | Markdown |
|---|---|---|
| H200 | 147 ms | 39 ms |
| RTX 4090 | 346 ms | 71 ms |

## Limitations and next step

Every injection in this study is a fixed piece of text, written before the defense existed, so the results say
nothing about an attacker who rewrites the injection after seeing what the model removes. The agents ran in a
simulated browser: each run handed the agent the text of one page, the e-mail and form tools only recorded the call,
and a language model judged the outcome. The next step is an adversary who revises the injection against the
deployed model, and with it a way to train the excision model efficiently as such attacks evolve.

## Repository layout

```
excise/data/      dataset at a pinned commit, diff labels, split by injection text, training windows, agent set
excise/model/     encoder with two heads, training, decoding, threshold and offset tuning, single-page inference
excise/eval/      exact-excision criterion, page and character metrics, test-set evaluation
excise/agent/     tool-calling agent, five simulated tools, four conditions, judge, result tables with Wilson intervals
excise/latency/   per-page timing of the model pass and the span excision
configs/          data, model, training and decoding settings used in the paper
data/splits/      the frozen train, dev, test, agent evaluation and latency sets, as entry ids
scripts/          training, agent experiment and latency runs for a GPU machine
tests/            unit tests of the HTML span alignment
paper/            the paper and the poster
```

Each module docstring names the section of the paper it implements. `excise/agent/framework/` is a minimal
tool-calling agent framework, and `excise/eval/failure_analysis.py` is a diagnostic the paper does not use.

## Reproduce

Python 3.10 or newer (the experiments ran on 3.12). BrowseSafe++ is gated on Hugging Face: accept its terms, then
`hf auth login`. The dataset and the backbone download at their pinned revisions on first use.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only the agent experiment calls APIs. Copy `.env.example` to `.env` at the repository root and fill in
`OPENROUTER_API_KEY` for the four open-weight agents, `OPENAI_API_KEY` for GPT-6 Luna and `DEEPSEEK_API_KEY` for the
judge; `.env` is in `.gitignore`. All commands run from the repository root, every entry point takes `--help`, and
`scripts/` wraps steps 2-6 for a GPU machine. Outputs go to `results/`, weights to `weights/`.

1. **Data.** The frozen splits are in `data/splits/` (`train`, `dev`, `test`, `agent_eval`, `latency`; entry ids,
   attack page then original page). To rebuild `train`, `dev`, `test` and `latency`: `python -m excise.data.split`
2. **Labels and windows.**
   `python -m excise.data.labels && python -m excise.data.windows`
3. **Training.** Writes `weights/run1/`; tuning adds `decode.json` with the page thresholds and tag offsets.
   `python -m excise.model.train --out weights/run1 && python -m excise.model.tune --weights weights/run1`
4. **Localization test.** The paper's HTML numbers were decoded without the last HTML alignment rule of
   `excise/model/decode.py` (83.9% exact excision); with the rule the same pages give 94.7%. The agent and latency
   runs use the full rule set.
   `python -m excise.eval.evaluate --split test --without-close-tag-rule`
5. **Agent experiment.** The agent evaluation set is 1,005 pairs outside train / dev / test, 201 per attack goal
   (`data/splits/agent_eval.txt`; `excise/data/agent_set.py` shows how it was drawn). On a GPU:
   `bash scripts/agent.sh pages`. Then, with API keys (this spends money): `bash scripts/agent.sh run`, which runs
   `python -m excise.agent.run`, `python -m excise.agent.judge` and `python -m excise.agent.summarize`. The paper's
   runs were made one process per agent (`--agents`, `--concurrency 10` for the no-defense condition and 20
   otherwise) and the run files concatenated before judging.
6. **Latency.** Times the 900 attack pages of `data/splits/latency.txt`, drawn from pairs outside train / dev /
   test. On an idle GPU: `bash scripts/latency.sh`. The RTX 4090 rows time 100 development-set attack pages per view:
   `python -m excise.latency.time_pages --split data/splits/dev.txt --pairs 100 --out results/latency_4090.jsonl`

Weights, results, labels, cached windows and dataset files are not in the repository; the steps above regenerate
them. The paper's tables and figures were formatted from `results/` and the training log; those scripts are not
included.

## Citation

See [CITATION.cff](CITATION.cff), or:

```bibtex
@software{bayuud2026excise,
  author = {Bayuud, Ochila},
  title  = {Excising Prompt Injections from Web Pages Before a Browser Agent Reads Them},
  year   = {2026},
  url    = {https://github.com/Imp3rv1ous/excise-prompt-injections}
}
```

## License

MIT, see [LICENSE](LICENSE).
