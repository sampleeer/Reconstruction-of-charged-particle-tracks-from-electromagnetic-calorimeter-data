"""Simulation-only labels. Never import this module from an inference path.

No ParentID was saved. A track born within 0.01 mm of the primary endpoint
is a co-born candidate, not a proven direct daughter. Long means >= 8.09 mm
maximum displacement from birth within the detector envelope. This is a
geometric truth definition, not a claim that all such branches are resolved.
"""

import numpy as np

LABEL_VERSION = "coborn-v1"
BIRTH_TOLERANCE_MM = 0.01
MIN_BRANCH_SPAN_MM = 8.09
# Repeated material boundaries in the supplied primary trajectories.
# These are inferred simulation dimensions, pending original Geant4 geometry.
DETECTOR_Z_MAX_MM = 176.08
DETECTOR_HALF_WIDTH_MM = 123.0
KNOWN_CHARGED = {11, 13, 15, 211, 321, 2212, 24, 3112, 3222, 3312, 3334}
KNOWN_NEUTRAL = {12, 14, 16, 22, 111, 130, 310, 311, 2112, 3122, 3212, 3322}


def charge_known(pdg):
    code = abs(int(pdg))
    if code >= 1_000_000_000:  # PDG nuclear code: 10LZZZAAAI
        return (code // 10000) % 1000 > 0
    if code in KNOWN_CHARGED:
        return True
    if code in KNOWN_NEUTRAL:
        return False
    return None


def label_event(steps, process, z_end):
    steps = np.asarray(steps, dtype=float)
    empty = dict(label_version=LABEL_VERSION, category="unresolved", star=None,
                 branch_count=None, vertex_mm=None, candidates=[],
                 unknown_pdg=[], reason="missing_or_invalid_trajectory")
    if steps.ndim != 2 or steps.shape[1] != 6 or not np.all(np.isfinite(steps)):
        return empty
    primary = steps[steps[:, 0] == 1]
    if not len(primary):
        return empty
    vertex = primary[-1, 2:5]
    if abs(vertex[2] - float(z_end)) > BIRTH_TOLERANCE_MM:
        return {**empty, "reason": "primary_endpoint_disagrees_with_Z_end"}
    interaction = "Inelastic" in process or "CaptureAtRest" in process
    inside = (0 <= vertex[2] <= DETECTOR_Z_MAX_MM and
              np.all(np.abs(vertex[:2]) < DETECTOR_HALF_WIDTH_MM))
    # Stable grouping preserves the chronological order within each track.
    order = np.argsort(steps[:, 0], kind="stable")
    sorted_steps = steps[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_steps[:, 0]))]
    groups = np.split(sorted_steps, starts[1:])
    candidates, unknown = [], []
    for track in groups:
        if track[0, 0] == 1:
            continue
        birth = track[0, 2:5]
        if np.linalg.norm(birth - vertex) > BIRTH_TOLERANCE_MM:
            continue
        pdg = int(track[0, 1])
        charged = charge_known(pdg)
        if charged is None:
            unknown.append(pdg)
        if charged is not True:
            continue
        points = track[:, 2:5]
        in_detector = ((points[:, 2] >= 0) & (points[:, 2] <= DETECTOR_Z_MAX_MM) &
                       np.all(np.abs(points[:, :2]) < DETECTOR_HALF_WIDTH_MM, axis=1))
        points = points[in_detector]
        span = float(np.max(np.linalg.norm(points - birth, axis=1))) if len(points) else 0.
        candidates.append(dict(track_id=int(track[0, 0]), pdg=pdg, span_mm=span,
                               long=span >= MIN_BRANCH_SPAN_MM))
    count = sum(t["long"] for t in candidates)
    common = dict(label_version=LABEL_VERSION, branch_count=count,
                  vertex_mm=vertex.tolist() if interaction and inside else None,
                  candidates=candidates, unknown_pdg=sorted(set(unknown)),
                  reason=None)
    if unknown or (interaction and not inside):
        return {**common, "category": "unresolved", "star": None,
                "reason": "unknown_charge_or_endpoint_outside_detector"}
    if interaction:
        return {**common, "category": "multi_prong" if count >= 2 else "other_interaction",
                "star": int(count >= 2)}
    if process == "Transportation" and count == 0:
        return {**common, "category": "transportation", "star": 0,
                "branch_count": 0}
    return {**common, "category": "unresolved", "star": None,
            "reason": "unsupported_terminal_process"}
