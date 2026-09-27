"""VRSBench-style eval: single-image VQA / captioning / grounding.

Scores the classical pipeline (core/query.py) on eval/data/vrsbench_sample.jsonl
— a small local stand-in split (NOT the real VRSBench download, unavailable
offline). Swap in the real VRSBench split by pointing SPLIT_PATH at a file
with the same {image, task, question, gold_keywords} schema.

    python -m eval.vrsbench
"""
from __future__ import annotations

import os

from .common import analyse, ask, load_jsonl, keyword_match, write_metrics, DATA

SPLIT_PATH = os.path.join(DATA, "vrsbench_sample.jsonl")


def run(split_path: str = SPLIT_PATH) -> dict:
    items = load_jsonl(split_path)
    per_task: dict = {}
    rows = []
    _cache: dict = {}
    for item in items:
        img = item["image"]
        if img not in _cache:
            _cache[img] = analyse(img)
        result = ask(_cache[img], item["question"])
        pred = result["answer"]
        correct = keyword_match(pred, item["gold_keywords"])
        task = item["task"]
        bucket = per_task.setdefault(task, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(correct)
        rows.append({"image": img, "task": task, "question": item["question"],
                     "pred": pred, "correct": correct,
                     "confidence": result.get("confidence")})

    metrics = {
        "benchmark": "vrsbench",
        "split": os.path.basename(split_path),
        "note": "Local stand-in split (real VRSBench download unavailable "
                "offline) — same schema, swap SPLIT_PATH for the real subset.",
        "n_items": len(items),
        "accuracy_by_task": {t: round(v["correct"] / v["n"], 3) for t, v in per_task.items()},
        "overall_accuracy": round(sum(v["correct"] for v in per_task.values())
                                  / max(sum(v["n"] for v in per_task.values()), 1), 3),
        "rows": rows,
    }
    write_metrics("vrsbench", metrics)
    return metrics


if __name__ == "__main__":
    m = run()
    print(f"vrsbench: {m['n_items']} items, overall accuracy {m['overall_accuracy']}")
    for t, acc in m["accuracy_by_task"].items():
        print(f"  {t}: {acc}")
