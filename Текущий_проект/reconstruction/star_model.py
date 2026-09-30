"""Supervised star proposal from two measured views, trained on weak MC labels."""

import numpy as np
from .features import projection_features
from .physical import peak_vertex


def predict_star(model, xz, yz):
    features = projection_features(xz, yz)[None]
    if not np.any(xz) or not np.any(yz):
        return dict(status="insufficient_views", probability=None, is_star=None,
                    vertex_mm=None, branch_count=None)
    probability = float(model["detector"].predict_proba(features)[0, 1])
    vertex = model["vertex"].predict(features)[0]
    if model.get("vertex_mode") == "peak_residual":
        vertex += peak_vertex(xz, yz)
    return dict(status="ok", probability=probability,
                is_star=probability >= model["threshold"],
                vertex_mm=vertex.tolist(),
                branch_count=max(0, int(np.rint(model["count"].predict(features)[0]))))
