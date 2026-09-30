"""Exploratory 44-plane completion with an explicit interpolated missing view.

At each physical plane only one marginal is observed. The other is an
interpolation prior, normalized to the observed energy. This assumption is
not an additional measurement and exact reprojection cannot validate 3D.
"""

import numpy as np
from .physical import PLANE_Z, project44
from .core import transport_volume


def complete44(xz, yz, track_mask, continuity=.25):
    xs = np.column_stack([np.interp(PLANE_Z, PLANE_Z[1::2], xz[:, c]) for c in range(96)])
    ys = np.column_stack([np.interp(PLANE_Z, PLANE_Z[0::2], yz[:, c]) for c in range(96)])
    xs[1::2], ys[0::2] = xz, yz
    fallback_layers = []
    for layer in range(44):
        observed, missing = (xs, ys) if layer % 2 else (ys, xs)
        total = observed[layer].sum()
        if total == 0:
            missing[layer] = 0
            continue
        if missing[layer].sum() == 0:
            source = yz if layer % 2 else xz
            source_z = PLANE_Z[0::2] if layer % 2 else PLANE_Z[1::2]
            available = np.flatnonzero(source.sum(axis=1))
            if len(available):
                nearest = available[np.argmin(abs(source_z[available]-PLANE_Z[layer]))]
                missing[layer] = source[nearest]
                fallback_layers.append(layer)
        if missing[layer].sum() > 0:
            missing[layer] *= total/missing[layer].sum()
    volume, info = transport_volume(xs, ys, track_mask, continuity=continuity)
    px, py = project44(volume)
    return volume, dict(method="interpolated unmeasured marginal + track/continuity transport prior",
                        continuity=continuity, nearest_nonempty_fallback_layers=fallback_layers,
                        skipped_layers=info["skipped_one_view_layers"],
                        observed_xz_mse=float(np.mean((px-xz)**2)),
                        observed_yz_mse=float(np.mean((py-yz)**2)),
                        nonzero_voxels=int((volume > 0).sum()))
