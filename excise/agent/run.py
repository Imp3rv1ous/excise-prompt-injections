"""Agent runs under four conditions (report Sections 3.4 and 4.2).

A tool-calling agent (excise/agent/framework, LiteLLM native function calling, at most 20 steps) gets the user
task, a read_page call, the page and the follow-up, then works on its own. The page depends on the condition:

  A  no defense      the attack page
  B  excision        the attack page with the predicted spans deleted
  C  blocking        a fixed placeholder sentence instead of the page
  R  original page   the original page of the pair

A, B and C run only on attack pages the model flagged in the Markdown view; R runs on every original page.
Input: results/agent_pages.jsonl. Output: one JSON line per run (events, final answer, steps, tokens, errors).
"""

import argparse
import asyncio
import json
import time

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from excise.agent.client import CountingClient
from excise.agent.framework.agent import Agent
from excise.agent.framework.context import ExecutionContext
from excise.agent.framework.types import Event, Message, ToolCall, ToolResult
from excise.agent.prompts import FOLLOW_UP, SYSTEM_PROMPT
from excise.agent.tools import TOOLS
from excise.data.source import original_id

# Four smaller open-weight models through OpenRouter and GPT-6 Luna, the frontier agent, through the OpenAI API.
AGENTS = [
    "openrouter/openai/gpt-oss-20b",
    "openrouter/google/gemma-4-31b-it",
    "openrouter/qwen/qwen3.5-9b",
    "openrouter/meta-llama/llama-4-scout",
    "gpt-6-luna",
]
# Qwen3.5 runs with thinking off (OpenRouter's `reasoning` parameter, passed through LiteLLM's extra_body).
MODEL_CONFIG = {"openrouter/qwen/qwen3.5-9b": {"extra_body": {"reasoning": {"enabled": False}}}}
CONDITIONS = {"A": "no defense", "B": "excision", "C": "blocking", "R": "original page"}
BLOCKED = "This page was blocked because it contains unsafe content."
MAX_STEPS = 20
TIMEOUT_S = 60  # per model call
RETRIES = 3


def load_pairs(path, skip=0, n=None):
    """Markdown rows of the pages file as (attack, original) pairs in file order."""
    rows = [r for r in map(json.loads, open(path)) if r["view"] == "markdown"]
    by_id = {r["entry_id"]: r for r in rows}
    pairs = [(r, by_id[original_id(r["entry_id"])]) for r in rows if r["arm"] == "attack"]
    return pairs[skip : skip + n if n else None]


def condition_pages(attack, original):
    """{condition: (entry id, page)}; A, B and C exist only when the model flagged the attack page."""
    pages = {"R": (original["entry_id"], original["text"])}
    if attack["detected"]:
        e = attack["entry_id"]
        pages.update(A=(e, attack["text"]), B=(e, attack["cleaned"]), C=(e, BLOCKED))
    return pages


def preset_context(task, page):
    """A fresh context holding the user task, the read_page call and the page, before FOLLOW_UP."""
    context = ExecutionContext(state={"page": page})
    events = [
        ("user", Message(role="user", content=task)),
        ("agent", ToolCall(tool_call_id="call_read_page", name="read_page", arguments={})),
        ("agent", ToolResult(tool_call_id="call_read_page", name="read_page", status="success", content=[page])),
    ]
    for author, item in events:
        context.add_event(Event(execution_id=context.execution_id, author=author, content=[item]))
    return context


async def run_one(model, condition, attack, entry_id, page, out, semaphore):
    client = CountingClient(model, num_retries=RETRIES, timeout=TIMEOUT_S, **MODEL_CONFIG.get(model, {}))
    agent = Agent(model=client, tools=TOOLS, instructions=SYSTEM_PROMPT, max_steps=MAX_STEPS)
    context = preset_context(attack["benign_task"], page)
    async with semaphore:
        t0 = time.perf_counter()
        result = await agent.run(FOLLOW_UP, context=context)
        wall = time.perf_counter() - t0

    tool_errors = sum(c.status == "error" for e in context.events for c in e.content if isinstance(c, ToolResult))
    row = dict(
        run_id=f"{model}|{condition}|{entry_id}",
        agent=model,
        condition=condition,
        pair=attack["pair"],
        entry_id=entry_id,
        goal_class=attack["goal_class"],
        restored=attack["restored"],
        events=[e.model_dump() for e in context.events],
        output=result.output,
        steps=context.current_step,
        wall_s=round(wall, 3),
        usage=client.usage,
        errors=client.errors,
        tool_errors=tool_errors,
    )
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    return row


async def run_all(pairs, agents, conditions, out, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        run_one(model, condition, attack, entry_id, page, out, semaphore)
        for attack, original in pairs
        for condition, (entry_id, page) in condition_pages(attack, original).items()
        if condition in conditions
        for model in agents
    ]
    return await tqdm_asyncio.gather(*tasks)


def main():
    p = argparse.ArgumentParser(description="Run the agents under the four conditions.")
    p.add_argument("--pages", default="results/agent_pages.jsonl")
    p.add_argument("--out", default="results/agent_runs.jsonl")
    p.add_argument("--agents", nargs="+", default=AGENTS)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--pairs", type=int, default=None, help="only this many pairs (after --skip)")
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=20)
    args = p.parse_args()

    load_dotenv()
    pairs = load_pairs(args.pages, args.skip, args.pairs)
    with open(args.out, "w") as out:
        rows = asyncio.run(run_all(pairs, args.agents, args.conditions, out, args.concurrency))
    print(
        len(rows),
        "runs;",
        sum(r["output"] is None for r in rows),
        "without a final answer;",
        sum(bool(r["errors"]) for r in rows),
        "with API errors;",
        sum(not a["detected"] for a, _ in pairs),
        "attack pages not flagged (no A / B / C runs)",
    )


if __name__ == "__main__":
    main()
