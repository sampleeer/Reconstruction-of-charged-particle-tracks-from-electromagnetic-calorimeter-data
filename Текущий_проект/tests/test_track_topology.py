import numpy as np
from reconstruction.physical import PLANE_Z, strip_index
from reconstruction.track_topology import extract_tracks, infer_topology
from reconstruction.topology_truth import trajectory_views


def test_finite_lines_survive_noise_without_extending_to_detector_boundary():
    image = np.zeros((22, 96))
    zs = PLANE_Z[1::2]
    for slope, intercept, lo, hi in ((.25, -35, 1, 10), (-.3, 45, 5, 18)):
        for row in range(lo, hi+1):
            col = int(strip_index(slope*zs[row]+intercept))
            if col >= 0: image[row, col] = .2
    image[0, 80] = 50  # a bright isolated deposit is not a track
    result = extract_tracks(image, 1)
    tracks = result['tracks']
    assert len(tracks) == 2
    for slope, intercept in ((.25, -35), (-.3, 45)):
        assert any(abs(t['slope']-slope) < .03 and abs(t['intercept']-intercept) < 2 for t in tracks)
    assert all(t['z_max'] < zs[-1] for t in tracks)


def test_empty_or_single_plane_does_not_invent_tracks_or_a_vertex():
    x = np.zeros((22, 96)); x[5, 30:35] = 1
    result = infer_topology(x, np.zeros_like(x))
    assert result['x']['tracks'] == [] and result['vertex_candidates'] == []


def test_truth_projection_handles_backward_tracks_and_inter_sensor_gaps():
    steps = np.array([[2, 211, 1, 1, 30, 10], [2, 211, 1, 1, 0, 0]], float)
    x, y = trajectory_views(steps)
    assert len(x) >= 3 and len(y) >= 3
    steps[:, 2] = 40  # between sensitive sensor regions
    assert trajectory_views(steps) == [[], []]


def test_geometric_vertex_and_multiple_3d_pairings_from_two_views_only():
    from reconstruction.physical import segment_voxels, project44
    from reconstruction.topology_reconstruct import reconstruct_topology
    vertex = np.array([1., 1., 60.])
    volume = np.zeros((44, 96, 96))
    for start, end in [([1,1,0], vertex), (vertex, [90,65,170]), (vertex, [-70,-90,165])]:
        for voxel in segment_voxels(start, end): volume[voxel] = .2
    x, y = project44(volume)
    result = reconstruct_topology(x, y)
    assert result['status'] == 'ok' and len(result['hypotheses']) >= 2
    assert np.linalg.norm(np.array(result['hypotheses'][0]['vertex_mm'])-vertex) < 3
    assert all(h['success'] and h['forward_operator_error'] < 1e-10 for h in result['hypotheses'])
    from reconstruction.topology_reconstruct import reconstruct_hybrid
    # A valid geometry must not need a learned model or reference labels.
    hybrid = reconstruct_hybrid(x, y, model=None)
    assert hybrid['source'] == 'segments' and hybrid['status'] == 'ok'
    empty = reconstruct_topology(np.zeros((22,96)), np.zeros((22,96)))
    assert empty['status'] == 'no_vertex' and not empty['hypotheses']
