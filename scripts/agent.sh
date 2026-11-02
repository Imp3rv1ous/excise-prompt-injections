#!/bin/bash
# Agent experiment (report Sections 3.4, 4.1 and 4.2).
#   bash scripts/agent.sh pages   on a GPU: excision model over the agent evaluation set (Table 5)
#   bash scripts/agent.sh run     anywhere: agents under the four conditions, judging, tables (Tables 6 and 7;
#                                 API calls cost money)
# API keys: a .env file at the repository root (see README).
set -e
mkdir -p results
case "$1" in
  pages)
    python -m excise.agent.prepare_pages --weights weights/run1 --out results/agent_pages.jsonl
    python -m excise.eval.evaluate --pages results/agent_pages.jsonl --by goal_class --name agent
    ;;
  run)
    python -m excise.agent.run --out results/agent_runs.jsonl
    python -m excise.agent.judge --runs results/agent_runs.jsonl --out results/agent_verdicts.jsonl
    python -m excise.agent.summarize --verdicts results/agent_verdicts.jsonl --out results/agent_table.md
    ;;
  *)
    echo "usage: bash scripts/agent.sh pages|run" && exit 1
    ;;
esac
