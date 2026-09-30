"""Detect and reconstruct one event using a trained model and 44 planes."""

import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from .core import projections_from_hits
from .star_model import predict_star
from .physical_fit import reconstruct44
from .physical import peak_vertex
from .completion import complete44


def run_event(model, xz, yz, smoothness=.3, vertex_search=True):
    start = time.perf_counter()
    proposal = predict_star(model, xz, yz)
    report = dict(proposal=proposal, reconstruction=None)
    if proposal["is_star"]:
        inferred = np.asarray(proposal["vertex_mm"])
        peak = peak_vertex(xz, yz)
        seeds = [("regression", inferred)]
        if vertex_search:
            seeds.extend([("peak", peak), ("midpoint", (peak+inferred)/2)])
        candidates = []
        for name, vertex in seeds:
            result = reconstruct44(xz, yz, vertex, proposal["branch_count"], smoothness=smoothness)
            result["vertex_source"] = name
            candidates.append(result)
        valid = [c for c in candidates if c["hypotheses"]]
        result = min(valid, key=lambda c: c["hypotheses"][0]["normalized_objective"]) if valid else candidates[0]
        result["vertex_alternatives"] = [dict(source=c["vertex_source"], vertex_mm=c["vertex_mm"],
                                              score=c["hypotheses"][0]["normalized_objective"] if c["hypotheses"] else None)
                                          for c in candidates]
        report["reconstruction"] = result
        if result["hypotheses"]:
            completion, info = complete44(xz, yz, result["hypotheses"][0]["mask"])
            result["completion"] = completion
            result["completion_info"] = info
    report["elapsed_seconds"] = time.perf_counter()-start
    return report


def serializable(report):
    return {**report, "reconstruction": (
        {**{k: v for k, v in report["reconstruction"].items() if k != "completion"}, "hypotheses": [
            {k: v for k, v in h.items() if k not in ("energy", "mask")}
            for h in report["reconstruction"]["hypotheses"]]}
        if report["reconstruction"] else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--model", type=Path, default=Path("results/star_model.joblib"))
    parser.add_argument("--smoothness", type=float, default=.3)
    parser.add_argument("--out", type=Path, default=Path("results/star_event"))
    args = parser.parse_args()
    data = np.load(args.file, allow_pickle=True).item()
    xz, yz = projections_from_hits(data, args.event_id)
    report = run_event(joblib.load(args.model), xz, yz, args.smoothness)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(serializable(report), indent=2)+"\n")
    recon = report["reconstruction"]
    if recon and recon["hypotheses"]:
        best = recon["hypotheses"][0]
        np.savez_compressed(args.out.with_suffix(".npz"), xz=xz, yz=yz,
                            energy=best["energy"], mask=best["mask"], completion=recon["completion"])
    print(json.dumps(serializable(report), indent=2))


if __name__ == "__main__":
    main()
