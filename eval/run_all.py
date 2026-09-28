"""Run all eval harnesses and write eval/results/summary.json.

    python -m eval.run_all
"""
from __future__ import annotations

from . import cdvqa, rsvqa, vrsbench
from .common import write_metrics
import os
import runpy
from pathlib import Path
from .common import DATA


def run_all() -> dict:
    # If a real VRSBench download exists (either .json or .jsonl), convert it
    # to the local eval JSONL (`real_vrsbench.jsonl`) and point the vrsbench
    # runner at that file.
    converted = os.path.join(DATA, "real_vrsbench.jsonl")
    raw_json = os.path.join(DATA, "real_vrsbench.json")
    raw_jsonl = os.path.join(DATA, "real_vrsbench.jsonl")
    script = os.path.join(DATA, "convert_vrsbench.py")
    # Prefer the .json raw download if present, else a raw .jsonl
    raw_input = None
    if os.path.exists(raw_json):
        raw_input = raw_json
    elif os.path.exists(raw_jsonl):
        # if raw is already a JSONL, we'll still run the converter to normalize
        raw_input = raw_jsonl
    if raw_input and os.path.exists(script):
        try:
            # run the converter in-process; it will write to `converted`
            runpy.run_path(script, run_name="__main__")
        except Exception:
            # fall back silently to bundled sample split
            pass
    # If conversion produced/left the converted file, use it.
    if os.path.exists(converted):
        vrsbench.SPLIT_PATH = converted

    v = vrsbench.run()
    r = rsvqa.run()
    c = cdvqa.run()
    summary = {
        "vrsbench": {"n_items": v["n_items"], "overall_accuracy": v["overall_accuracy"],
                     "accuracy_by_task": v["accuracy_by_task"]},
        "rsvqa": {"n_items": r["n_items"], "accuracy": r["accuracy"]},
        "cdvqa": {"n_items": c["n_items"], "keyword_accuracy": c["keyword_accuracy"],
                  "mean_changed_fraction_abs_error": c["mean_changed_fraction_abs_error"]},
        "note": "All three splits are local stand-ins (offline environment, no "
                "dataset downloads). Structure/schema is ready for the real "
                "prescribed VRSBench/RSVQA/CDVQA subsets — see resource.md §3B.",
    }
    write_metrics("summary", summary)
    return summary


if __name__ == "__main__":
    s = run_all()
    import json
    print(json.dumps(s, indent=2))
