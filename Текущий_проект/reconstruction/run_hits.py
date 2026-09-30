"""Reconstruct one event from the historical Geant4 hit-table .npy format.

The file is a NumPy pickled dictionary. Only open a file you trust. No values
from the 3D truth table enter the optimization except the two emulated views.
"""

import argparse
import json
from pathlib import Path
import numpy as np

from .core import (fit_energy, fit_geometry, permutation_candidates, project,
                   projections_from_hits, rasterize, transport_volume,
                   truth_from_hits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--event-id", type=float, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/geant4_event"))
    parser.add_argument("--max-branches", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--evaluate-truth", action="store_true",
                        help="Report paired-plane hit IoU after fitting")
    args = parser.parse_args()
    payload = np.load(args.file, allow_pickle=True).item()
    xz, yz = projections_from_hits(payload, args.event_id)
    best, models = fit_geometry(xz, yz, max_branches=args.max_branches,
                                maxiter=args.maxiter)
    candidates = []
    for geometry in permutation_candidates(best["geometry"]):
        mask = rasterize(geometry)
        energy, info = fit_energy(mask, xz, yz)
        candidates.append((info["xz_mse"] + info["yz_mse"], mask, energy, info))
    _, mask, energy, info = min(candidates, key=lambda row: row[0])
    transport, transport_info = transport_volume(xz, yz, mask)
    px, py = project(mask)
    def overlap(a, b):
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / union) if union else 1.0
    report = {
        "event_id": args.event_id,
        "selected_branches": best["branches"],
        "projection_iou_xz": overlap(px, xz > 0),
        "projection_iou_yz": overlap(py, yz > 0),
        "pairing_candidates": len(candidates),
        "energy_fit": info,
        "transport_fit": transport_info,
        "models": [{k: v for k, v in m.items() if k != "geometry"} for m in models],
        "evaluation_uses_truth": args.evaluate_truth,
    }
    if args.evaluate_truth:
        truth = truth_from_hits(payload, args.event_id)
        report["paired_plane_hit_iou"] = overlap(mask, truth > 0)
        report["transport_paired_plane_hit_iou"] = overlap(transport > 0, truth > 0)
        report["truth_hit_count"] = int((truth > 0).sum())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), xz=xz, yz=yz,
                        mask=mask, energy=energy, transport=transport)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
