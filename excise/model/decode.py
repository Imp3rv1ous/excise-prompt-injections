"""Decoding: from a page to its score, injection spans and excised text (report Section 3.3).

1. Page logits: windows are scored and the tag logits of tokens covered by several windows are averaged.
2. Constrained Viterbi: the best BIOES tag sequence after adding the O / B / E offsets.
3. Character spans: tag runs become character ranges; an edge token that straddles `><` is trimmed to the tag
   boundary.
4. HTML alignment: spans never cut a tag in half, a deleted opening tag takes its closing tag with it, and a
   closing tag at the start of a span (its opening tag lies outside) is kept.
5. The page is excised only when its score (highest window score) reaches the view's threshold.

Decoding parameters (`weights/<run>/decode.json`, written by excise/model/tune.py):
  threshold: {view: page-score threshold}; bias: [O, B, E] offsets;
  close_tag_rule (optional, default true): the last rule of step 4; the report's HTML numbers are computed
  without it (see excise/eval/evaluate.py).
"""

import re

import numba
import numpy as np
import torch

from excise.config import WINDOW
from excise.data.source import normalize, remove
from excise.data.windows import B, E, S, encode, window_starts

NEG = float("-inf")
# Allowed tag transitions: row = previous tag, column = next tag, order O B I E S.
ALLOWED = torch.tensor(
    [[1, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1]], dtype=torch.bool
)
TRANS = torch.where(ALLOWED, 0.0, NEG).numpy()
START = torch.tensor([0, 0, NEG, NEG, 0])
END = torch.tensor([0, NEG, NEG, 0, 0])
PAD_EM = torch.tensor([0, NEG, NEG, NEG, NEG])  # padding allows only O, which enforces END at the real end

TAG = re.compile(r"<(/?)([a-zA-Z][\w-]*)[^>]*>")
CLOSE = re.compile(r"</[a-zA-Z][\w-]*\s*>")
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


@torch.no_grad()
def page_logits(model, tok, text, batch=8):
    """Averaged tag logits [T, 5], window scores [n] and token offsets of one page."""
    dev = next(model.parameters()).device
    ids, offsets = encode(tok, text)
    starts = window_starts(len(ids))
    x = torch.from_numpy(np.stack([ids[s : s + WINDOW] for s in starts]).astype(np.int64)).to(dev)

    outs = []
    for i in range(0, len(x), batch):
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            outs.append(model(x[i : i + batch], torch.ones_like(x[i : i + batch])))
    tag_logits = torch.cat([o[0].float() for o in outs])
    win_logits = torch.cat([o[1].float() for o in outs])

    total = torch.zeros(len(ids), 5, device=dev)
    count = torch.zeros(len(ids), 1, device=dev)
    for k, s in enumerate(starts):
        total[s : s + WINDOW] += tag_logits[k]
        count[s : s + WINDOW] += 1
    return (total / count).cpu(), win_logits.sigmoid().cpu(), offsets


@numba.njit(cache=True)
def _viterbi(em, trans, start, end, path):
    """Constrained Viterbi on one sequence em [T, 5]; ties go to the lowest tag index."""
    T = em.shape[0]
    score = em[0] + start
    new = np.empty(5, np.float32)
    bp = np.empty((T, 5), np.int8)
    for t in range(1, T):
        for j in range(5):
            best, bi = -np.inf, 0
            for i in range(5):
                v = score[i] + trans[i, j]
                if v > best:
                    best, bi = v, i
            new[j] = best + em[t, j]
            bp[t, j] = bi
        score = new.copy()

    last, bestv = 0, -np.inf
    for j in range(5):
        v = score[j] + end[j]
        if v > bestv:
            bestv, last = v, j
    path[T - 1] = last
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]


@numba.njit(parallel=True, cache=True)
def _viterbi_batch(em, trans, start, end):
    paths = np.empty((em.shape[0], em.shape[1]), np.int64)
    for n in numba.prange(em.shape[0]):
        _viterbi(em[n], trans, start, end, paths[n])
    return paths


def viterbi(em):
    """em [N, T, 5] with offsets added and padding filled with PAD_EM -> best valid tag paths [N, T]."""
    em = np.ascontiguousarray(em.cpu().numpy(), dtype=np.float32)
    return _viterbi_batch(em, TRANS, START.numpy(), END.numpy())


def bias_vector(bias):
    """(O, B, E) offsets -> a vector added to the five tag logits."""
    return torch.tensor([bias[0], bias[1], 0.0, bias[2], 0.0])


def char_spans(text, offsets, path):
    """Tag path -> character spans; edge tokens straddling `><` are trimmed to the tag boundary."""
    starts = np.flatnonzero((path == B) | (path == S))
    ends = np.flatnonzero((path == E) | (path == S))
    out = []
    for i, j in zip(starts, ends):
        s, e = offsets[i, 0], offsets[j, 1]
        first, last = text[s : offsets[i, 1]], text[offsets[j, 0] : e]
        if "><" in first:
            s += first.index("><") + 1
        if "><" in last:
            e = offsets[j, 0] + last.rindex("><") + 1
        if s < e:
            out.append((int(s), int(e)))
    return out


def merge(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]))
        else:
            out.append((s, e))
    return out


def unclosed_tags(text, s, e):
    """Names of the opening tags inside [s, e) whose closing tag is not inside it."""
    stack = []
    for m in TAG.finditer(text, s, e):
        name = m.group(2).lower()
        if name in VOID:
            continue
        if not m.group(1):
            stack.append(name)
        elif stack and stack[-1] == name:
            stack.pop()
    return stack


def align_html(text, spans, close_tag_rule=True):
    """Move span edges to HTML tag boundaries (step 4 above)."""
    out = []
    for s, e in spans:
        if text.rfind("<", 0, s) > text.rfind(">", 0, s):
            s = text.rfind("<", 0, s)
        lt, gt = text.find("<", e), text.find(">", e)
        if gt != -1 and (lt == -1 or gt < lt):
            e = gt + 1
        while close_tag_rule and (m := CLOSE.match(text, s)) and m.end() < e:
            s = m.end()
        for name in reversed(unclosed_tags(text, s, e)):
            m = re.compile(rf"</{name}\s*>", re.I).search(text, e)
            if m:
                e = m.end()
        out.append((s, e))
    return merge(out)


def decode_page(logits, offsets, text, view, params):
    path = viterbi((logits + bias_vector(params["bias"]))[None])[0]
    spans = char_spans(text, offsets, path)
    if view == "html":
        spans = align_html(text, spans, params.get("close_tag_rule", True))
    return spans


def excise(model, tok, text, view, params):
    """One raw page -> {"score", "spans", "cleaned"}; spans refer to the normalized page."""
    text = normalize(text)
    logits, scores, offsets = page_logits(model, tok, text)
    score = float(scores.max())
    spans = decode_page(logits, offsets, text, view, params) if score >= params["threshold"][view] else []
    return dict(score=score, spans=spans, cleaned=remove(text, spans))
