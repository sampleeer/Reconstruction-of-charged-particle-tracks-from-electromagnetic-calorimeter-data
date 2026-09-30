"""Fixed-seed exploratory check on 10 sparse events from each Geant4 sample."""

import argparse
import json
from pathlib import Path
import numpy as np

from .core import (fit_energy, fit_geometry, permutation_candidates, project,
                   projections_from_hits, rasterize, transport_volume,
                   truth_from_hits)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results/geant4_pilot.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    rows = []
    for kind, stem in (("proton", "calorimeter_response"),
                       ("antiproton", "calorimeter_response (2)")):
        data = np.load(args.data_dir / (stem + ".npy"), allow_pickle=True).item()
        ids, counts = np.unique(data["event_ID"], return_counts=True)
        pool = ids[(counts >= 30) & (counts <= 100)]
        chosen = np.sort(rng.choice(pool, size=10, replace=False))
        for event_id in chosen:
            xz, yz = projections_from_hits(data, event_id)
            if not np.any(xz) or not np.any(yz):
                continue
            best, _ = fit_geometry(xz, yz, max_branches=2, maxiter=20)
            candidates = []
            for geometry in permutation_candidates(best["geometry"]):
                mask = rasterize(geometry)
                _, info = fit_energy(mask, xz, yz)
                candidates.append((info["xz_mse"] + info["yz_mse"], mask, info))
            _, mask, info = min(candidates, key=lambda item: item[0])
            transport, transport_info = transport_volume(xz, yz, mask)
            truth = truth_from_hits(data, event_id) > 0
            px, py = project(mask)
            rows.append({
                "kind": kind, "event_id": int(event_id),
                "raw_hits": int(counts[np.where(ids == event_id)[0][0]]),
                "branches": best["branches"],
                "xz_iou": iou(px, xz > 0), "yz_iou": iou(py, yz > 0),
                "paired_truth_iou_approx": iou(mask, truth),
                "transport_paired_truth_iou_approx": iou(transport > 0, truth),
                "energy_mse": info["xz_mse"] + info["yz_mse"],
                "transport_energy_mse": (transport_info["xz_mse_raw"] +
                                         transport_info["yz_mse_raw"]),
            })
    summary = {kind: {key: float(np.mean([row[key] for row in rows
                                          if row["kind"] == kind]))
                      for key in ("xz_iou", "yz_iou", "paired_truth_iou_approx",
                                  "transport_paired_truth_iou_approx")}
               for kind in ("proton", "antiproton")}
    report = {
        "selection": "NumPy RNG 2026; 10 IDs per species with 30-100 raw hits; max_branches=2, maxiter=20",
        "caution": "paired_truth_iou_approx merges adjacent physical planes; it is not a true same-plane 3D IoU",
        "summary": summary, "events": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
