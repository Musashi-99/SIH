"""Eval harness for SatQuery AI (resource.md §3B "Eval harness").

Runners: vrsbench.py (caption/VQA/grounding), rsvqa.py (VQA), cdvqa.py
(change-VQA). Each scores the CURRENT classical pipeline against a small
local stand-in split shipped in eval/data/ — real VRSBench/RSVQA/CDVQA
downloads are not available in this offline environment, so the split
files are clearly labelled synthetic and structured to be swapped 1:1 for
the real prescribed subsets later (same JSONL schema).
"""
