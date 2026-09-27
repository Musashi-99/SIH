"""CDVQA-style eval: bi-temporal change description / change-VQA.

Scores core/change.py + core/query.py change answers against
eval/data/cdvqa_sample.jsonl, using samples/delta_truth.json as ground
truth for changed_fraction. Local stand-in pair (real CDVQA/LEVIR-CD/WHU-CD
unavailable offline) — same schema, swap in a real pair + split file later.

    python -m eval.cdvqa
"""
from __future__ import annotations

import json
import os

from .common import ask, keyword_match, run_change, write_metrics, DATA, SAMPLES

SPLIT_PATH = os.path.join(DATA, "cdvqa_sample.jsonl")
TRUTH_PATH = os.path.join(SAMPLES, "delta_truth.json")


def run(split_path: str = SPLIT_PATH) -> dict:
    with open(split_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    with open(TRUTH_PATH, encoding="utf-8") as f:
        truth = json.load(f)
    gold_fraction = truth["total_changed_fraction"]

    _cache: dict = {}
    rows = []
    correct = 0
    frac_errors = []
    for item in items:
        key = (item["t1"], item["t2"])
        if key not in _cache:
            _cache[key] = run_change(item["t1"], item["t2"])
        change = _cache[key]
        pred_fraction = change["result"].changed_fraction
        # Use the first pair's analysis as the query-answer context; change.py
        # already carries both scenes via `change`.
        from .common import analyse
        a1 = analyse(item["t1"])
        result = ask(a1, item["question"], change)
        pred_text = result["answer"]

        ok = True
        if "gold_keywords" in item:
            ok = keyword_match(pred_text, item["gold_keywords"])
        correct += int(ok)

        err = None
        if "gold_changed_fraction" in item or gold_fraction is not None:
            gold_f = item.get("gold_changed_fraction", gold_fraction)
            err = abs(pred_fraction - gold_f)
            frac_errors.append(err)

        rows.append({"question": item["question"], "pred": pred_text,
                     "pred_changed_fraction": round(pred_fraction, 5),
                     "keyword_correct": ok,
                     "changed_fraction_abs_error": round(err, 5) if err is not None else None})

    metrics = {
        "benchmark": "cdvqa",
        "split": os.path.basename(split_path),
        "note": "Local stand-in pair (real CDVQA/LEVIR-CD/WHU-CD unavailable "
                "offline) — same schema, swap in a real pair + split later.",
        "n_items": len(items),
        "keyword_accuracy": round(correct / max(len(items), 1), 3),
        "mean_changed_fraction_abs_error": round(sum(frac_errors) / len(frac_errors), 5)
                                           if frac_errors else None,
        "rows": rows,
    }
    write_metrics("cdvqa", metrics)
    return metrics


if __name__ == "__main__":
    m = run()
    print(f"cdvqa: {m['n_items']} items, keyword accuracy {m['keyword_accuracy']}, "
          f"mean |Δfraction error| {m['mean_changed_fraction_abs_error']}")
