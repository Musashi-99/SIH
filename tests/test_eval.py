"""Eval harness smoke tests (resource.md §3B "Eval harness")."""
from eval.cdvqa import run as run_cdvqa
from eval.rsvqa import run as run_rsvqa
from eval.vrsbench import run as run_vrsbench


def test_vrsbench_runner_produces_metrics():
    m = run_vrsbench()
    assert m["n_items"] > 0
    assert 0.0 <= m["overall_accuracy"] <= 1.0
    assert set(m["accuracy_by_task"]) >= {"vqa", "caption", "grounding"}


def test_rsvqa_runner_produces_metrics():
    m = run_rsvqa()
    assert m["n_items"] > 0
    assert 0.0 <= m["accuracy"] <= 1.0


def test_cdvqa_runner_produces_metrics_and_matches_ground_truth_order():
    m = run_cdvqa()
    assert m["n_items"] > 0
    # The classical CVA pipeline should land within a coarse tolerance of the
    # synthetic ground-truth changed fraction (0.04472) — this is a sanity
    # bound, not a benchmark-grade accuracy claim.
    assert m["mean_changed_fraction_abs_error"] < 0.15
