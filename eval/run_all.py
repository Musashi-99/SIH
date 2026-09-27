"""Run all eval harnesses and write eval/results/summary.json.

    python -m eval.run_all
"""
from __future__ import annotations

from . import cdvqa, rsvqa, vrsbench
from .common import write_metrics


def run_all() -> dict:
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
