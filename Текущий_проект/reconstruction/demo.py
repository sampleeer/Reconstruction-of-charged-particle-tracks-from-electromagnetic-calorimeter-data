"""Run a small reproducible experiment without private calorimeter data."""

import argparse
import json
from pathlib import Path
import numpy as np

from .core import (Geometry, fit_energy, fit_geometry, permutation_candidates,
                   project, rasterize)


def synthetic_event(shape=(22, 96, 96)):
    geometry = Geometry(45, 45, 0.17, -0.12, 9,
                        ((72, 62, 21), (23, 70, 21)))
    mask = rasterize(geometry, shape)
    energy = np.zeros(shape)
    coords = np.argwhere(mask)
    energy[tuple(coords.T)] = 0.5 + 0.03 * coords[:, 0]
    primary = rasterize(Geometry(geometry.start_x, geometry.start_y,
                                 geometry.slope_x, geometry.slope_y,
                                 geometry.vertex_z), shape)
    for branch, extra in zip(geometry.endpoints, (1.0, 3.0)):
        branch_mask = rasterize(Geometry(geometry.start_x, geometry.start_y,
                                         geometry.slope_x, geometry.slope_y,
                                         geometry.vertex_z, (branch,)), shape)
        energy[branch_mask & ~primary] += extra
    energy[9, 47, 44] += 5  # a brighter interaction region
    xz, yz = project(energy)
    return geometry, energy, xz, yz


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_demo.json"))
    parser.add_argument("--max-branches", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=35)
    args = parser.parse_args()
    truth_geometry, truth_energy, xz, yz = synthetic_event()
    best, models = fit_geometry(xz, yz, max_branches=args.max_branches,
                                maxiter=args.maxiter)
    raw_mask = rasterize(best["geometry"])
    alternatives = []
    for candidate in permutation_candidates(best["geometry"]):
        mask = rasterize(candidate)
        fitted, info = fit_energy(mask, xz, yz)
        alternatives.append((info["xz_mse"] + info["yz_mse"], candidate,
                             mask, fitted, info))
    _, _, fitted_mask, fitted_energy, energy_info = min(
        alternatives, key=lambda item: item[0])
    oracle_energy, oracle_info = fit_energy(truth_energy > 0, xz, yz)
    pred_x, pred_y = project(fitted_mask)
    report = {
        "data": "synthetic; no Geant4 or PAMELA events used",
        "truth_branches": len(truth_geometry.endpoints),
        "chosen_branches": best["branches"],
        "binary_projection_iou_xz": iou(xz > 0, pred_x),
        "binary_projection_iou_yz": iou(yz > 0, pred_y),
        "geometry_iou_before_pairing": iou(truth_energy > 0, raw_mask),
        "geometry_iou_after_pairing": iou(truth_energy > 0, fitted_mask),
        "pairing_candidates": len(alternatives),
        "models": [{k: v for k, v in m.items() if k != "geometry"} for m in models],
        "fitted_energy": energy_info,
        "oracle_mask_energy": oracle_info,
        "truth_energy_sum": float(truth_energy.sum()),
        "fitted_energy_sum": float(fitted_energy.sum()),
        "oracle_energy_sum": float(oracle_energy.sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
