"""Freeze artifacts before opening test results; verify them when evaluating."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol():
    protocol = json.loads(Path("results/frozen_protocol.json").read_text())
    for path, expected in protocol["artifacts_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen experiment changed: {path}. Create a new experiment/version.")
    return protocol


def main():
    path = Path("results/frozen_protocol.json")
    if path.exists():
        raise SystemExit("Protocol already frozen; do not overwrite an evaluated experiment.")
    sources = sorted(str(p) for p in Path("reconstruction").glob("*.py"))
    artifacts = ["results/star_model.joblib", "data/derived/features.npz",
                 "data/derived/labels.jsonl", "data/derived/manifest.json", *sources]
    protocol = dict(version="stars-v1", label_version="coborn-v1",
                    split="sha256 of measured projections, 60/20/20; inspected events forced to train",
                    detection_selection="ExtraTrees leaf 3 vs 12; threshold by validation F1",
                    vertex_selection="absolute vs peak residual regression; validation mean 3D error",
                    reconstruction_selection="three vertex proposals; minimum regularized energy objective; at most 4 matched rays",
                    smoothness=.3, completion_continuity=.25, threshold_MeV=.01,
                    reconstruction_test_per_species_category=100, reconstruction_seed=450,
                    reconstruction_validation="16 stars + 32 controls; single/search and smoothness .03/.3 explored; .3 retained (negligible difference)",
                    limitations=["co-birth is not a ParentID", "long truth branches are not guaranteed observable",
                                 "inferred simulation geometry", "test is from the same two simulation runs",
                                 "no real PAMELA validation", "completion uses an imputed marginal"],
                    artifacts_sha256={p: sha256(p) for p in artifacts})
    path.write_text(json.dumps(protocol, indent=2)+"\n")
    Path("results/dataset_manifest.json").write_text(Path("data/derived/manifest.json").read_text())
    print(path)


if __name__ == "__main__":
    main()
