#!/usr/bin/env python3
"""Fail-closed validation of the generalist G0 acceptance evidence."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from mjlab_microduck.generalist_transition_graph import LEGAL_EDGES, UNSUPPORTED_EDGES

def validate(evaluation: dict, parity: dict, budget: dict, fallback: dict) -> dict:
    reasons = []
    if not evaluation.get("finite", False): reasons.append("evaluation is non-finite")
    if {(x.get("from"), x.get("to")) for x in evaluation.get("legal_edges", [])} != LEGAL_EDGES:
        reasons.append("legal edge coverage is incomplete")
    unsupported = {(x.get("from"), x.get("to")) for x in evaluation.get("unsupported_edges", []) if not x.get("exercised", True)}
    if unsupported != set(UNSUPPORTED_EDGES): reasons.append("unsupported direct edges are not explicitly unexercised")
    if not parity.get("passed", False): reasons.append("ONNX parity gate failed or is missing")
    if not budget.get("passed", False): reasons.append("50 Hz inference budget failed or is missing")
    if not fallback.get("preserved", False): reasons.append("specialist fallback gate failed or is missing")
    for item in evaluation.get("behaviors", []):
        metrics = item.get("metrics", {})
        if not metrics.get("finite", False): reasons.append(f"non-finite behavior: {item.get('behavior')}")
        if "success" not in item and "passed" not in item:
            reasons.append(f"missing behavior success gate: {item.get('behavior')}")
        elif not item.get("success", item.get("passed", False)):
            reasons.append(f"behavior success gate failed: {item.get('behavior')}")
    for item in evaluation.get("legal_edges", []):
        if item.get("reset_count", 0) != 0: reasons.append(f"reset on legal edge: {item.get('from')}->{item.get('to')}")
        if "success" not in item and "passed" not in item:
            reasons.append(f"missing transition success gate: {item.get('from')}->{item.get('to')}")
        elif not item.get("success", item.get("passed", False)):
            reasons.append(f"transition success gate failed: {item.get('from')}->{item.get('to')}")
    return {"schema": "generalist-g0-acceptance-gate", "version": 1,
            "status": "ACCEPT" if not reasons else "DIAGNOSTIC_FAIL", "accepted": not reasons,
            "reasons": reasons, "legal_edges": len(LEGAL_EDGES), "unsupported_edges": len(UNSUPPORTED_EDGES)}

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--evaluation",type=Path,required=True); p.add_argument("--parity",type=Path,required=True); p.add_argument("--budget",type=Path,required=True); p.add_argument("--fallback",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    result=validate(*(json.loads(x.read_text()) for x in (a.evaluation,a.parity,a.budget,a.fallback)))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["accepted"] else 2)
if __name__ == "__main__": main()
