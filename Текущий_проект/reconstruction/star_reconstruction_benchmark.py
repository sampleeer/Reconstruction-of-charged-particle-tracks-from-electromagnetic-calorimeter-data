"""Fixed stratified reconstruction audit, separate from all-event detection."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from .core import projections_from_hits
from .star_reconstruct import run_event, serializable
from .physical import project44


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()/union) if union else 1.


def truth44(data, event_id):
    selected = np.asarray(data["event_ID"]) == event_id
    coords = [np.asarray(data[k][selected], int) for k in ("layer", "index_along_x", "index_along_y")]
    energy = data["energy_release"][selected]
    volume = np.zeros((44, 96, 96))
    np.add.at(volume, tuple(coords), energy)
    return volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--smoothness", type=float, default=.3)
    parser.add_argument("--single-vertex", action="store_true")
    args = parser.parse_args()
    if args.split == "test":
        from .protocol import verify_protocol
        protocol = verify_protocol()
        if (args.single_vertex or args.smoothness != protocol["smoothness"] or
                args.per_stratum != protocol["reconstruction_test_per_species_category"]):
            raise ValueError("Test parameters differ from frozen protocol")
    rows = [json.loads(s) for s in Path("data/derived/labels.jsonl").read_text().splitlines()]
    model = joblib.load("results/star_model.joblib")
    rng = np.random.default_rng(450)
    results = []
    for kind, suffix in (("proton", ""), ("antiproton", " (2)")):
        data = np.load(f"data/raw/calorimeter_response{suffix}.npy", allow_pickle=True).item()
        for category in ("multi_prong", "other_interaction", "transportation"):
            pool = [r for r in rows if r["split"] == args.split and r["kind"] == kind
                    and r["category"] == category and r["both_views"]]
            chosen = sorted(rng.choice(len(pool), min(args.per_stratum, len(pool)), replace=False))
            for index in chosen:
                row = pool[index]
                event_id = row["event_id"]
                xz, yz = projections_from_hits(data, event_id)
                report = run_event(model, xz, yz, args.smoothness, not args.single_vertex)
                # Evaluation truth becomes available only after inference.
                truth = truth44(data, event_id)
                metrics = {}
                recon = report["reconstruction"]
                if recon and recon["hypotheses"]:
                    best = recon["hypotheses"][0]
                    mx, my = project44(best["mask"])
                    metrics = dict(voxel_iou44=iou(best["mask"], truth > .01),
                                   projected_x_iou=iou(mx > 0, xz > .01),
                                   projected_y_iou=iou(my > 0, yz > .01),
                                   truth_energy_covered=float(truth[best["mask"]].sum()/max(truth.sum(), 1e-12)))
                    if row["star"]:
                        metrics["vertex_error_mm"] = float(np.linalg.norm(np.asarray(recon["vertex_mm"])-row["vertex_mm"]))
                    metrics["completion_voxel_iou44"] = iou(recon["completion"] > .01, truth > .01)
                    metrics["completion_energy_relative_l1"] = float(np.abs(recon["completion"]-truth).sum()/max(truth.sum(), 1e-12))
                record = dict(kind=kind, event_id=event_id, category=category,
                              truth_vertex_mm=row["vertex_mm"], truth_branch_count=row["branch_count"],
                              metrics=metrics, **serializable(report))
                results.append(record)
                print(kind, event_id, category, metrics, flush=True)
    summary = {}
    for category in ("multi_prong", "other_interaction", "transportation"):
        part = [r for r in results if r["category"] == category]
        reconstructed = [r for r in part if r["metrics"]]
        summary[category] = dict(n=len(part), detected=sum(bool(r["proposal"]["is_star"]) for r in part),
                                reconstructed=len(reconstructed),
                                mean_seconds=float(np.mean([r["elapsed_seconds"] for r in part])),
                                mean_voxel_iou44_on_reconstructed=float(np.mean([r["metrics"]["voxel_iou44"] for r in reconstructed])) if reconstructed else None,
                                mean_voxel_iou44_misses_zero=float(np.mean([r["metrics"].get("voxel_iou44", 0) for r in part])),
                                mean_completion_iou44_misses_zero=float(np.mean([r["metrics"].get("completion_voxel_iou44", 0) for r in part])),
                                mean_completion_energy_relative_l1_on_reconstructed=float(np.mean([r["metrics"]["completion_energy_relative_l1"] for r in reconstructed])) if reconstructed else None,
                                median_refined_vertex_error_mm=float(np.median([r["metrics"]["vertex_error_mm"] for r in reconstructed])) if reconstructed and category == "multi_prong" else None,
                                capped=sum(bool(r["reconstruction"] and r["reconstruction"].get("capped")) for r in part))
    report = dict(split=args.split, selection=f"seed450; {args.per_stratum} random per species/category; categories deliberately balanced",
                  smoothness=args.smoothness, threshold_MeV=.01, summary=summary, events=results)
    tag = "single" if args.single_vertex else "search"
    path = Path(f"results/reconstruction44_{args.split}_{tag}_{args.smoothness:g}.json")
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
