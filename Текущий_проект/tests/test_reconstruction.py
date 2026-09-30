import numpy as np

from reconstruction import (Geometry, fit_energy, fit_geometry,
                            permutation_candidates, project,
                            projections_from_hits, rasterize,
                            transport_volume)
from reconstruction.demo import iou, synthetic_event


def test_44_plane_adapter_uses_only_the_measured_coordinate():
    hits = {"event_ID": np.array([7, 7, 7, 7]),
            "layer": np.array([0, 1, 2, 3]),
            "index_along_x": np.array([-1, 2, -1, 4]),
            "index_along_y": np.array([5, -1, 7, -1]),
            "energy_release": np.array([2., 3., 4., 5.])}
    xz, yz = projections_from_hits(hits, 7, shape=(2, 10, 10))
    assert xz[0, 2] == 3 and xz[1, 4] == 5
    assert yz[0, 5] == 2 and yz[1, 7] == 4
    assert xz.sum() == 8 and yz.sum() == 6


def test_backward_branch_and_boundaries():
    g = Geometry(5, 5, 0, 0, 5, ((0, 9, 0), (9, 0, 9)))
    mask = rasterize(g, (10, 10, 10))
    assert mask[0, 0, 9] and mask[9, 9, 0] and mask[5, 5, 5]


def test_synthetic_pipeline_recovers_projections_and_energy_pairing():
    _, truth, xz, yz = synthetic_event()
    best, models = fit_geometry(xz, yz, max_branches=2, maxiter=20)
    assert best["branches"] == 2
    assert len(models) == 3
    candidates = []
    for g in permutation_candidates(best["geometry"]):
        mask = rasterize(g)
        volume, info = fit_energy(mask, xz, yz)
        candidates.append((info["xz_mse"] + info["yz_mse"], mask, volume))
    score, mask, volume = min(candidates, key=lambda item: item[0])
    assert len(candidates) == 2
    assert iou(mask, truth > 0) == 1
    assert score < 1e-4
    assert np.all(volume >= 0)
    pred_x, pred_y = project(mask)
    assert np.array_equal(pred_x, xz > 0)
    assert np.array_equal(pred_y, yz > 0)


def test_inconsistent_independent_views_are_fit_without_false_exactness():
    _, truth, xz, yz = synthetic_event()
    yz = yz * 1.25
    volume, info = fit_energy(truth > 0, xz, yz)
    assert np.all(volume >= 0)
    assert info["xz_total"] != info["yz_total"]
    assert info["xz_mse"] > 0 and info["yz_mse"] > 0


def test_transport_couples_strips_and_accounts_for_unequal_view_totals():
    _, truth, xz, yz = synthetic_event()
    volume, info = transport_volume(xz, yz, truth > 0)
    assert iou(volume > 0, truth > 0) == 1
    assert info["xz_mse_raw"] < 1e-12
    unequal, diag = transport_volume(xz, yz * 1.2, truth > 0)
    assert diag["xz_mse_raw"] > 0 and diag["yz_mse_raw"] > 0
    assert np.all(unequal >= 0)
