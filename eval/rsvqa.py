"""RSVQA-style eval: single-image yes/no VQA.

Scores core/query.py presence answers on eval/data/rsvqa_sample.jsonl — a
local stand-in split (real RSVQA-LR/HR unavailable offline). Same schema
({image, question, gold: "yes"/"no"}) so a real split drops in directly.

    python -m eval.rsvqa
"""
from __future__ import annotations

import os

from .common import analyse, ask, load_jsonl, write_metrics, DATA

SPLIT_PATH = os.path.join(DATA, "rsvqa_sample.jsonl")


def run(split_path: str = SPLIT_PATH) -> dict:
    items = load_jsonl(split_path)
    rows = []
    correct = 0
    _cache: dict = {}
    for item in items:
        img = item["image"]
        if img not in _cache:
            _cache[img] = analyse(img)
        result = ask(_cache[img], item["question"])
        pred_answer = result["answer"]
        pred_yn = "yes" if pred_answer.lower().lstrip("*").startswith("yes") else "no"
        ok = pred_yn == item["gold"]
        correct += int(ok)
        rows.append({"image": img, "question": item["question"], "pred": pred_yn,
                     "gold": item["gold"], "correct": ok})

    metrics = {
        "benchmark": "rsvqa",
        "split": os.path.basename(split_path),
        "note": "Local stand-in split (real RSVQA-LR/HR unavailable offline) — "
                "same schema, swap SPLIT_PATH for the real subset.",
        "n_items": len(items),
        "accuracy": round(correct / max(len(items), 1), 3),
        "rows": rows,
    }
    write_metrics("rsvqa", metrics)
    return metrics


if __name__ == "__main__":
    m = run()
    print(f"rsvqa: {m['n_items']} items, accuracy {m['accuracy']}")
