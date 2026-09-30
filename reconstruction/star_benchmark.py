"""Train on train; select on validation; optionally evaluate the frozen test."""

import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def metrics(y, probabilities, threshold):
    prediction = probabilities >= threshold
    return dict(n=len(y), positives=int(y.sum()), threshold=float(threshold),
                precision=float(precision_score(y, prediction, zero_division=0)),
                recall=float(recall_score(y, prediction, zero_division=0)),
                f1=float(f1_score(y, prediction, zero_division=0)),
                average_precision=float(average_precision_score(y, probabilities)) if y.sum() else None,
                roc_auc=float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
                confusion_matrix_tn_fp_fn_tp=confusion_matrix(y, prediction, labels=[0, 1]).ravel().tolist())


def choose_threshold(y, scores):
    thresholds = np.linspace(0.1, 0.9, 81)
    return float(max(thresholds, key=lambda t: f1_score(y, scores >= t)))


def error_summary(error):
    error = np.asarray(error)
    return dict(n=len(error), mean=float(np.mean(error)), median=float(np.median(error)),
                p90=float(np.quantile(error, .9))) if len(error) else dict(n=0)


def evaluate(model, features, peaks, rows, selected):
    ids = np.flatnonzero(selected)
    x = features[selected]
    y = np.array([rows[i]["star"] for i in ids])
    probability = model["detector"].predict_proba(x)[:, 1]
    vertex = model["vertex"].predict(x)
    if model.get("vertex_mode") == "peak_residual":
        vertex += peaks[selected]
    count = np.maximum(np.rint(model["count"].predict(x)), 0).astype(int)
    truth_count = np.array([rows[i]["branch_count"] for i in ids])
    stars = y == 1
    truth_vertex = np.array([r["vertex_mm"] if r["vertex_mm"] is not None else [np.nan]*3
                             for r in (rows[i] for i in ids)])
    distance = np.linalg.norm(vertex - truth_vertex, axis=1)
    predicted = probability >= model["threshold"]
    report = dict(detection=metrics(y, probability, model["threshold"]),
                  vertex_mm_on_true_stars=error_summary(distance[stars]),
                  vertex_z_mm_on_true_stars=error_summary(np.abs(vertex[stars, 2]-truth_vertex[stars, 2])),
                  vertex_mm_on_detected_true_stars=error_summary(distance[stars & predicted]),
                  count_mae_on_true_stars=float(np.mean(np.abs(count[stars]-truth_count[stars]))),
                  count_exact_on_true_stars=float(np.mean(count[stars]==truth_count[stars])),
                  detection_and_vertex_within_10mm_recall=float(np.mean(predicted[stars] & (distance[stars] <= 10))),
                  strata={})
    for name, mask in [(kind, np.array([rows[i]["kind"] == kind for i in ids]))
                       for kind in ("proton", "antiproton")]:
        report["strata"][name] = metrics(y[mask], probability[mask], model["threshold"])
    for lo, hi in ((80, 1000), (1000, 3000), (3000, 10001)):
        mask = np.array([lo <= rows[i]["energy_MeV"] < hi for i in ids])
        report["strata"][f"energy_{lo}_{hi}_MeV"] = metrics(y[mask], probability[mask], model["threshold"])
    for category in ("transportation", "other_interaction"):
        mask = np.array([rows[i]["category"] == category for i in ids])
        report["strata"][category] = dict(n=int(mask.sum()), false_positive_rate=float(predicted[mask].mean()))
    # Simple, interpretable comparators; thresholds/constant learned without test.
    occupancy = features[selected, 4*22:5*22].sum(axis=1) + features[selected, 14*22:15*22].sum(axis=1)
    report["occupancy_baseline"] = metrics(y, occupancy / 44, model["occupancy_threshold"])
    peak_z = peaks[selected, 2]
    report["peak_energy_vertex_z_mm"] = error_summary(np.abs(peak_z[stars]-truth_vertex[stars, 2]))
    report["peak_energy_vertex_3d_mm"] = error_summary(np.linalg.norm(peaks[selected][stars]-truth_vertex[stars], axis=1))
    report["constant_count_mae_on_true_stars"] = float(np.mean(np.abs(model["constant_count"]-truth_count[stars])))
    rng = np.random.default_rng(971)
    boot = [f1_score(y[idx], predicted[idx], zero_division=0)
            for idx in (rng.integers(0, len(y), len(y)) for _ in range(300))]
    report["f1_event_bootstrap_95pct"] = np.quantile(boot, [.025, .975]).tolist()
    predictions = [dict(kind=rows[i]["kind"], event_id=rows[i]["event_id"],
                        truth_star=int(y[j]), probability=float(probability[j]),
                        predicted_star=bool(predicted[j]), vertex_mm=vertex[j].tolist(),
                        true_vertex_mm=rows[i]["vertex_mm"], branch_count=int(count[j]),
                        true_branch_count=int(truth_count[j])) for j, i in enumerate(ids)]
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/derived"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate the saved frozen model, do not refit")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (args.dataset / "labels.jsonl").read_text().splitlines()]
    dataset = np.load(args.dataset / "features.npz")
    features, peaks = dataset["features"], dataset["peak_vertices"]
    eligible = np.array([r["star"] is not None and r["both_views"] for r in rows])
    subsets = {s: eligible & np.array([r["split"] == s for r in rows])
               for s in ("train", "validation", "test")}
    y = np.array([r["star"] if r["star"] is not None else -1 for r in rows])
    started = time.perf_counter()
    if args.evaluate_test:
        from .protocol import verify_protocol
        verify_protocol()
        model = joblib.load(args.out / "star_model.joblib")
        split = "test"
    else:
        train, val = subsets["train"], subsets["validation"]
        candidates = []
        for leaf in (3, 12):
            clf = ExtraTreesClassifier(n_estimators=160, min_samples_leaf=leaf,
                                       max_features=.7, n_jobs=4, random_state=2026)
            clf.fit(features[train], y[train])
            prob = clf.predict_proba(features[val])[:, 1]
            threshold = choose_threshold(y[val], prob)
            score = metrics(y[val], prob, threshold)
            candidates.append((score["f1"], clf, threshold, leaf, score))
        _, detector, threshold, leaf, _ = max(candidates, key=lambda c: c[0])
        stars = train & (y == 1)
        vertex_candidates = []
        vertex_truth = np.array([rows[i]["vertex_mm"] for i in np.flatnonzero(stars)])
        val_stars = val & (y == 1)
        val_truth = np.array([rows[i]["vertex_mm"] for i in np.flatnonzero(val_stars)])
        for mode in ("absolute", "peak_residual"):
            vertex = ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3, max_features=.8,
                                         n_jobs=4, random_state=2027)
            vertex.fit(features[stars], vertex_truth-(peaks[stars] if mode == "peak_residual" else 0))
            pred = vertex.predict(features[val_stars])+(peaks[val_stars] if mode == "peak_residual" else 0)
            error = error_summary(np.linalg.norm(pred-val_truth, axis=1))
            vertex_candidates.append((error["mean"], mode, vertex, error))
        _, vertex_mode, vertex, _ = min(vertex_candidates, key=lambda v: v[0])
        count = ExtraTreesRegressor(n_estimators=180, min_samples_leaf=5, max_features=.8,
                                    n_jobs=4, random_state=2028)
        count.fit(features[train], np.array([rows[i]["branch_count"] for i in np.flatnonzero(train)]))
        occupancy = (features[:, 4*22:5*22].sum(axis=1)+features[:, 14*22:15*22].sum(axis=1))/44
        occupancy_threshold = float(max(np.linspace(0, .2, 201),
                                        key=lambda t: f1_score(y[val], occupancy[val] >= t)))
        model = dict(detector=detector, threshold=threshold, vertex=vertex, count=count,
                     occupancy_threshold=occupancy_threshold,
                     constant_count=float(np.median([rows[i]["branch_count"] for i in np.flatnonzero(stars)])),
                     label_version="coborn-v1", sklearn_version=sklearn.__version__,
                     vertex_mode=vertex_mode,
                     vertex_validation_candidates=[dict(mode=v[1], **v[3]) for v in vertex_candidates],
                     selected_min_samples_leaf=leaf,
                     validation_candidates=[dict(min_samples_leaf=c[3], **c[4]) for c in candidates])
        joblib.dump(model, args.out / "star_model.joblib", compress=3)
        split = "validation"
    report, predictions = evaluate(model, features, peaks, rows, subsets[split])
    report.update(split=split, elapsed_seconds=time.perf_counter()-started,
                  target="weak geometric co-born multi-prong label, not particle species",
                  excluded_unresolved=sum(r["star"] is None for r in rows if r["split"] == split),
                  excluded_missing_views=sum(not r["both_views"] for r in rows if r["split"] == split),
                  selected_min_samples_leaf=model["selected_min_samples_leaf"],
                  vertex_mode=model["vertex_mode"],
                  vertex_validation_candidates=model["vertex_validation_candidates"],
                  validation_candidates=model["validation_candidates"],
                  source_manifest=json.loads((args.dataset / "manifest.json").read_text())["files_sha256"])
    (args.out / f"star_{split}.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / f"star_{split}_predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
