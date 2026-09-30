"""Build audited labels, projection-only features, and fixed event splits."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .labels import label_event, LABEL_VERSION
from .features import all_projections, projection_features
from .physical import peak_vertex


def fingerprint(xz, yz):
    return hashlib.sha256(np.asarray([xz, yz], dtype="<f4").tobytes()).hexdigest()


def split_for(fingerprint_value):
    # Identical measurement pairs stay in one split, including across species.
    value = int(hashlib.sha256(("diploma-v1:" + fingerprint_value).encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 60 else "validation" if value < 80 else "test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/derived"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    explored = {"proton": set(range(500)), "antiproton": set(range(500))}
    pilot = Path("results/geant4_pilot.json")
    if pilot.exists():
        for row in json.loads(pilot.read_text())["events"]:
            explored[row["kind"]].add(row["event_id"])
    rows, features, peak_vertices, explored_hashes, hashes = [], [], [], set(), {}
    for kind, suffix in (("proton", ""), ("antiproton", " (2)")):
        paths = [args.data_dir / (name + suffix + ".npy")
                 for name in ("calorimeter_response", "secondary")]
        for p in paths:
            hashes[p.name] = hashlib.file_digest(p.open("rb"), "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(p.read_bytes()).hexdigest()
        data, secondary = [np.load(p, allow_pickle=True).item() for p in paths]
        xz, yz = all_projections(data)
        for event_id, steps in enumerate(secondary["secondary"]):
            label = label_event(steps, str(data["last_process"][event_id]).strip(), data["Z_end"][event_id])
            digest = fingerprint(xz[event_id], yz[event_id])
            if event_id in explored[kind]:
                explored_hashes.add(digest)
            rows.append(dict(kind=kind, event_id=event_id, fingerprint=digest,
                             energy_MeV=float(data["E_0"][event_id]),
                             process=str(data["last_process"][event_id]).strip(),
                             both_views=bool(xz[event_id].any() and yz[event_id].any()),
                             **label))
            features.append(projection_features(xz[event_id], yz[event_id]))
            peak_vertices.append(peak_vertex(xz[event_id], yz[event_id]))
        print(f"Annotated {kind}: {len(secondary['secondary'])}", flush=True)
    for row in rows:
        row["split"] = "train" if row["fingerprint"] in explored_hashes else split_for(row["fingerprint"])
    (args.out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.savez_compressed(args.out / "features.npz", features=np.asarray(features),
                        peak_vertices=np.asarray(peak_vertices))
    summary = dict(label_version=LABEL_VERSION, files_sha256=hashes, total=len(rows),
                   categories=dict(Counter(r["category"] for r in rows)),
                   splits={split: dict(Counter(r["category"] for r in rows if r["split"] == split))
                           for split in ("train", "validation", "test")},
                   forced_train_explored_events={k: sorted(v) for k, v in explored.items()},
                   distinct_measurements=len({r["fingerprint"] for r in rows}),
                   no_both_views=sum(not r["both_views"] for r in rows))
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "forced_train_explored_events"}, indent=2))


if __name__ == "__main__":
    main()
