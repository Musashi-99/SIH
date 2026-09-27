"""Shared plumbing for eval/*.py runners.

Runs the exact same pipeline app.py serves (via ``app._analyse2``) so eval
scores reflect production behaviour, not a separate offline reimplementation.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import _analyse2  # noqa: E402  (reuse the real pipeline, not a copy)
from core import change as change_mod  # noqa: E402
from core import query as query_mod  # noqa: E402

SAMPLES = os.path.join(ROOT, "samples")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def analyse(name: str, gsd: float = 10.0, k: int = 7) -> dict:
    return _analyse2(os.path.join(SAMPLES, name), gsd, k)


def ask(analysis: dict, question: str, change: Optional[dict] = None) -> dict:
    plan = query_mod.parse_query(question)
    return query_mod.answer(plan, analysis, change)


def run_change(name1: str, name2: str, gsd: float = 10.0, k: int = 7) -> dict:
    a = analyse(name1, gsd, k)
    b = analyse(name2, gsd, k)
    cr = change_mod.detect(a["scene"], b["scene"], k=a["k_used"], lc1=a["landcover"])
    return {"result": cr}


def load_jsonl(path: str) -> list:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def keyword_match(pred: str, gold_keywords: list) -> bool:
    """Loose VQA-style scoring: gold is a list of acceptable keywords/phrases,
    any one matching (case-insensitive substring) counts as correct. Standard
    approach for free-form-answer VQA eval against short gold labels."""
    p = pred.lower()
    return any(kw.lower() in p for kw in gold_keywords)


def write_metrics(name: str, metrics: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path
