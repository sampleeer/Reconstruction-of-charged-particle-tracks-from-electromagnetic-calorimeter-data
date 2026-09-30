import numpy as np
from reconstruction.labels import label_event
from reconstruction.features import all_projections, projection_features
from reconstruction.build_dataset import fingerprint, split_for


def test_labels_require_cobirth_and_long_tracks_not_species():
    steps = np.array([[1, -2212, 0, 0, 0, 100], [1, -2212, 0, 0, 40, 0],
                      [2, 211, 0, 0, 40, 20], [2, 211, 10, 0, 50, 0],
                      [3, 2212, 0, 0, 40, 20], [3, 2212, -10, 0, 30, 0],
                      [4, 11, 0, 0, 10, 20], [4, 11, 40, 0, 40, 0],
                      [5, 2212, 0, 0, 40, 1], [5, 2212, 0, 0, 41, 0]])
    label = label_event(steps, "anti_protonInelastic", 40)
    assert label["star"] == 1 and label["branch_count"] == 2
    assert [c["track_id"] for c in label["candidates"]] == [2, 3, 5]
    assert label_event(steps, "anti_protonInelastic", 42)["star"] is None
    # An antiproton traversing the detector is not automatically a star.
    assert label_event(steps[:2], "Transportation", 40)["star"] == 0


def test_batch_projection_adapter_is_insensitive_to_hidden_truth():
    data = dict(E_0=np.ones(2), event_ID=np.array([0, 0, 1, 1]),
                layer=np.array([0, 1, 2, 3]), index_along_x=np.array([999, 4, -1, 6]),
                index_along_y=np.array([3, 999, 5, -1]), energy_release=np.arange(1, 5.))
    x, y = all_projections(data)
    assert x[0, 0, 4] == 2 and y[0, 0, 3] == 1
    before = projection_features(x[0], y[0])
    data["index_along_x"][[0, 2]] = 0
    data["index_along_y"][[1, 3]] = 95
    data["E_0"] *= 999
    xx, yy = all_projections(data)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(y, yy)
    np.testing.assert_array_equal(before, projection_features(xx[0], yy[0]))
    assert len(before) == 440


def test_identical_measurements_cannot_cross_splits():
    x = np.zeros((22, 96)); y = x.copy()
    x[3, 4] = .123
    assert split_for(fingerprint(x, y)) == split_for(fingerprint(x.copy(), y.copy()))


def test_physical_forward_operator_preserves_alternating_planes_and_gaps():
    from reconstruction.physical import project44, strip_index, segment_voxels
    volume = np.zeros((44, 96, 96))
    volume[0, 4, 9] = 2
    volume[1, 7, 3] = 5
    x, y = project44(volume)
    assert x[0, 7] == 5 and y[0, 9] == 2
    assert x.sum() == 5 and y.sum() == 2
    assert strip_index(40.) == -1  # physical inter-sensor gap
    # Backward and horizontal tracks intersect finite sensitive slabs.
    assert len(segment_voxels([1, 1, 20], [-10, -10, 0])) > 0
    assert len(segment_voxels([-10, -10, .19], [10, 10, .19])) > 1


def test_physical_energy_fit_never_enforces_equality_of_adjacent_planes():
    from reconstruction.physical import project44, segment_voxels
    from reconstruction.physical_fit import energy_on_segments
    segments = [([1, 1, 0], [1, 1, 176.08])]
    truth = np.zeros((44, 96, 96))
    for z, x, y in segment_voxels(*segments[0]):
        truth[z, x, y] = 1. if z % 2 else 3.
    x, y = project44(truth)
    volume, mask, info = energy_on_segments(segments, x, y, smoothness=0, ridge=0)
    assert info["normalized_energy_residual"] < 1e-10
    assert info["forward_operator_error"] < 1e-10
    assert np.all(volume >= 0) and np.all(volume[~mask] == 0)
    assert x.sum() != y.sum()


def test_completion_conserves_each_observed_plane_not_adjacent_pair_totals():
    from reconstruction.completion import complete44
    from reconstruction.physical import project44
    x, y = np.zeros((22, 96)), np.zeros((22, 96))
    x[:, 48] = 1
    y[:, 48] = 3
    mask = np.zeros((44, 96, 96), bool)
    mask[:, 48, 48] = True
    volume, info = complete44(x, y, mask)
    xx, yy = project44(volume)
    np.testing.assert_allclose(xx, x, atol=1e-8)
    np.testing.assert_allclose(yy, y, atol=1e-8)
    assert not info["skipped_layers"]


def test_ambiguity_limit_checked_before_factorial_allocation():
    import pytest
    from reconstruction import Geometry, permutation_candidates
    g = Geometry(5, 5, 0, 0, 3, tuple((i, i, 9) for i in range(20)))
    with pytest.raises(ValueError, match="exceed"):
        list(permutation_candidates(g))
