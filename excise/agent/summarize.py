"""Agent result tables (report Sections 3.5 and 4.2).

- Main table: per agent and condition, the attack success rate (ASR) and the task completion rate with 95% Wilson
  intervals; runs the judge left undecided (null) are left out of the denominator and counted. A run with no
  final answer counts as a failed task.
- ASR under no defense and excision per attack goal.
- Excision runs split by whether excision restored the original page.
Input: verdicts and pages files. Output: a Markdown file.
"""

import argparse
import json

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from excise.agent.run import AGENTS, CONDITIONS


def rate(values):
    """Share of True among the True / False values with its 95% Wilson interval; None is left out."""
    k, n = sum(v is True for v in values), sum(v is not None for v in values)
    if not n:
        return "-"
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k / n:.1%} [{lo:.1%}, {hi:.1%}] (n={n})"


def task_values(d):
    return [False if no_answer else v for v, no_answer in zip(d.task_success, d.no_answer)]


def main_table(df):
    lines = [
        "## Attack success and task completion",
        "",
        "| Agent | Condition | Runs | ASR (95% CI) | Task completion (95% CI) | ASR null "
        "| Task null | No final answer | API errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for agent in AGENTS:
        for c, name in CONDITIONS.items():
            d = df[(df.agent == agent) & (df.condition == c)]
            if len(d):
                lines.append(
                    f"| {agent} | {c} {name} | {len(d)} | {rate(list(d.injection_followed))} "
                    f"| {rate(task_values(d))} | {d.injection_followed.isna().sum()} | {d.task_success.isna().sum()} "
                    f"| {d.no_answer.sum()} | {(d.api_errors > 0).sum()} |"
                )
    return lines


def goal_table(df):
    lines = [
        "## ASR per attack goal",
        "",
        "| Agent | Attack goal | Pages | A no defense | B excision |",
        "|---|---|---|---|---|",
    ]
    for agent in AGENTS:
        for goal, d in df[df.agent == agent].groupby("goal_class"):
            a, b = d[d.condition == "A"], d[d.condition == "B"]
            if len(a):
                lines.append(
                    f"| {agent} | {goal} | {len(a)} | {rate(list(a.injection_followed))} "
                    f"| {rate(list(b.injection_followed))} |"
                )
    return lines


def restored_table(df):
    lines = [
        "## Excision runs by whether the original page was restored",
        "",
        "| Agent | Excised page | Runs | ASR | Task completion |",
        "|---|---|---|---|---|",
    ]
    for agent in AGENTS:
        for value, name in ((True, "restored"), (False, "not restored")):
            d = df[(df.agent == agent) & (df.condition == "B") & (df.restored == value)]
            if len(d):
                lines.append(
                    f"| {agent} | {name} | {len(d)} | {rate(list(d.injection_followed))} | {rate(task_values(d))} |"
                )
    return lines


def summarize(verdicts, pages):
    df = pd.DataFrame(list(map(json.loads, open(verdicts))))  # not read_json: it turns pair ids like "09219" into ints
    pairs = set(df.pair)
    attack = [
        r
        for r in map(json.loads, open(pages))
        if r["view"] == "markdown" and r["arm"] == "attack" and r["pair"] in pairs
    ]
    flagged = [r for r in attack if r["detected"]]
    lines = [
        "# Agent results",
        "",
        f"Markdown attack pages flagged: {len(flagged)} / {len(attack)}; the other {len(attack) - len(flagged)} have "
        f"no A / B / C runs. Flagged pages restored by excision: {sum(r['restored'] for r in flagged)}.",
        "",
    ]
    lines += main_table(df) + [""] + goal_table(df) + [""] + restored_table(df) + [""]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Agent result tables.")
    p.add_argument("--verdicts", default="results/agent_verdicts.jsonl")
    p.add_argument("--pages", default="results/agent_pages.jsonl")
    p.add_argument("--out", default="results/agent_table.md")
    args = p.parse_args()

    text = summarize(args.verdicts, args.pages)
    open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
