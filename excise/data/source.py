"""BrowseSafe++ pages at a pinned commit, and the text helpers every stage shares (report Section 3.1).

Each pair has an attack page (`arm == "attack"`, entry id `<pair>_malicious`) and its original page
(`arm == "control"`, entry id `<pair>_benign`); each page exists in two views, HTML and Markdown.
Spans are character ranges [start, end) of a page after `normalize`.
"""

import glob
import re

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from excise.config import DATASET

VIEWS = {"html": "content_html", "markdown": "content_markdown"}
COLUMNS = [
    "entry_id",
    "arm",
    "label",
    "content_html",
    "content_markdown",
    "benign_task",
    "expected_outcome",
    "malicious_goal",
    "injection_text_fragments_json",
    "attack_quotes_json",
    "requires_action_tools",
    "goal_class",
    "attack_type",
    "injection_strategy",
    "linguistic_style",
]


def normalize(text):
    """Collapse runs of blank lines into one newline; labels, inference and evaluation all start here."""
    return re.sub(r"\n[ \t]*(\n[ \t]*)+", "\n", text)


def remove(text, spans):
    """The page with the sorted, non-overlapping spans deleted."""
    out, p = [], 0
    for s, e in spans:
        out.append(text[p:s])
        p = e
    return "".join(out) + text[p:]


def original_id(entry_id):
    return entry_id.replace("malicious", "benign")


def load_pages(shards=None, rows=None):
    """All pages as a DataFrame. Whole parquet files are read (the preview API truncates long fields)."""
    if shards is None:
        patterns = ["data/*.parquet"]
    else:
        patterns = [f"data/members-train-{s:04d}.parquet" for s in shards]
    root = snapshot_download(
        DATASET["repo"], repo_type="dataset", revision=DATASET["revision"], allow_patterns=patterns
    )
    files = sorted(f for pat in patterns for f in glob.glob(f"{root}/{pat}"))
    df = pa.concat_tables([pq.read_table(f, columns=COLUMNS) for f in files]).to_pandas()
    df["pair"] = df.entry_id.str.split("_").str[0]
    return df if rows is None else df.head(rows)


def read_split(name, split_dir="data/splits"):
    return [line.strip() for line in open(f"{split_dir}/{name}.txt")]
