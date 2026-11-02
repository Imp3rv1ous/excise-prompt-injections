"""Judge every agent run: task_success and injection_followed (report Section 3.4).

The judge is DeepSeek Flash through LiteLLM. It sees the task, the reference answer from the original page, the
attack goal and quotes, the recorded tool calls and the final answer; runs on original pages are judged against
their pair's attack goal too. The verdict arrives as a tool call, so the API parses the JSON.
Input: results/agent_runs.jsonl and results/agent_pages.jsonl. Output: one verdict line per run.
"""

import argparse
import asyncio
import json
from types import SimpleNamespace

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from excise.agent.client import CountingClient
from excise.agent.framework.base import FunctionTool
from excise.agent.framework.helpers import format_tool_definition
from excise.agent.framework.llm import LlmRequest
from excise.agent.framework.types import Message, ToolCall
from excise.agent.prompts import JUDGE_PROMPT, Verdict
from excise.agent.run import load_pairs

JUDGE = "deepseek/deepseek-flash"
SCHEMA = Verdict.model_json_schema()
SCHEMA.pop("title")
CALL_FOR_VERDICT = "\n\nReport your verdict by calling the verdict tool."
DESCRIPTION = "Record the verdict for this run."
VERDICT_TOOL = FunctionTool(
    func=lambda **kw: None,
    name="verdict",
    description=DESCRIPTION,
    tool_definition=format_tool_definition("verdict", DESCRIPTION, SCHEMA),
)
NO_VERDICT = Verdict(task_success=None, injection_followed=None, evidence="judge returned no verdict")


def judge_prompt(run, attack, original):
    calls = [
        dict(name=c["name"], arguments=c["arguments"])
        for e in run["events"]
        for c in e["content"]
        if c["type"] == "tool_call"
    ]
    prompt = JUDGE_PROMPT.format(
        benign_task=attack["benign_task"],
        control=SimpleNamespace(**original),
        attack=SimpleNamespace(**attack),
        tool_calls=json.dumps(calls, ensure_ascii=False),
        agent_final_answer=run["output"] or "",
    )
    return prompt + CALL_FOR_VERDICT


def verdict_row(run, arguments, judge, usage):
    """One verdict line; arguments is the verdict tool call's input, or None when the judge gave none."""
    verdict = Verdict.model_validate(arguments) if arguments is not None else NO_VERDICT
    keep = ["run_id", "agent", "condition", "pair", "entry_id", "goal_class", "restored"]
    return dict(
        {k: run[k] for k in keep},
        no_answer=run["output"] is None,
        api_errors=len(run["errors"]),
        tool_errors=run["tool_errors"],
        **verdict.model_dump(),
        judge=judge,
        judge_no_verdict=arguments is None,
        judge_usage=usage,
        agent_usage=run["usage"],
    )


async def judge_one(run, pairs, out, model, semaphore):
    client = CountingClient(model, num_retries=3)
    request = LlmRequest(
        instructions=[judge_prompt(run, *pairs[run["pair"]])],
        contents=[Message(role="user", content="Please respond.")],
        tools=[VERDICT_TOOL],
        tool_choice="auto",
    )
    async with semaphore:
        response = await client.generate(request)
    calls = [c for c in response.content if isinstance(c, ToolCall)]
    row = verdict_row(run, calls[0].arguments if calls else None, model, client.usage[0])
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    return row


async def judge_all(runs, pairs, out, model, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    return await tqdm_asyncio.gather(*[judge_one(run, pairs, out, model, semaphore) for run in runs])


def main():
    p = argparse.ArgumentParser(description="Judge task success and injection following of every run.")
    p.add_argument("--runs", default="results/agent_runs.jsonl")
    p.add_argument("--pages", default="results/agent_pages.jsonl")
    p.add_argument("--out", default="results/agent_verdicts.jsonl")
    p.add_argument("--judge-model", default=JUDGE)
    p.add_argument("--conditions", nargs="+", default=None, help="judge only runs of these conditions")
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()

    load_dotenv()
    runs = [r for r in map(json.loads, open(args.runs)) if not args.conditions or r["condition"] in args.conditions]
    pairs = {attack["pair"]: (attack, original) for attack, original in load_pairs(args.pages)}
    with open(args.out, "w") as out:
        rows = asyncio.run(judge_all(runs, pairs, out, args.judge_model, args.concurrency))
    print(
        len(rows),
        "verdicts;",
        sum(r["injection_followed"] is None for r in rows),
        "followed=null;",
        sum(r["task_success"] is None for r in rows),
        "task=null;",
        sum(r["judge_no_verdict"] for r in rows),
        "without a verdict",
    )


if __name__ == "__main__":
    main()
