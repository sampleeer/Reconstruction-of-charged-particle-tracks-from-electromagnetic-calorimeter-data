"""Inference features from measured strip energies only; no truth fields."""

import numpy as np
from .core import _check_images


def projection_features(xz, yz):
    xz, yz = _check_images(xz, yz)
    features = []
    for image in (xz, yz):
        # Include several fixed energy cuts: the simulator has arbitrarily
        # tiny deposits and the experimental electronic threshold is unknown.
        energy = image.sum(axis=1)
        coordinate = np.arange(image.shape[1]) / (image.shape[1] - 1)
        mean = image @ coordinate / np.maximum(energy, 1e-12)
        variance = image @ (coordinate ** 2) / np.maximum(energy, 1e-12) - mean ** 2
        features.extend([np.log1p(energy), np.log1p(image.max(axis=1)), mean,
                         np.sqrt(np.maximum(variance, 0))])
        for cut in (0., 0.01, 0.1):
            present = image > cut
            features.extend([present.sum(axis=1) / image.shape[1],
                             (present & ~np.pad(present[:, :-1], ((0, 0), (1, 0)))).sum(axis=1) / 10])
    return np.concatenate(features).astype(np.float32)


def all_projections(data):
    """Batch adapter. Unmeasured coordinates never participate in selection."""
    n = len(data["E_0"])
    event = np.asarray(data["event_ID"], dtype=int)
    layer = np.asarray(data["layer"], dtype=int)
    energy = np.asarray(data["energy_release"], dtype=float)
    views = []
    for parity, key in ((1, "index_along_x"), (0, "index_along_y")):
        coord = np.asarray(data[key], dtype=int)
        valid = ((event >= 0) & (event < n) & (layer >= 0) & (layer < 44) &
                 (layer % 2 == parity) & (coord >= 0) & (coord < 96))
        bins = (event[valid] * 22 + layer[valid] // 2) * 96 + coord[valid]
        image = np.bincount(bins, weights=energy[valid], minlength=n*22*96)
        views.append(image.reshape(n, 22, 96).astype(np.float32))
    return views
