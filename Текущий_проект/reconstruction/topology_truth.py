"""Evaluation-only visibility proxy from MC trajectories, never inference."""
import numpy as np
from .physical import PLANE_Z, strip_index, STRIP_CENTERS


def trajectory_views(steps):
    views = [set(), set()]  # x, y; (view row, strip)
    for a, b in zip(steps[:-1, 2:5], steps[1:, 2:5]):
        dz = b[2]-a[2]
        if abs(dz) < 1e-9:
            continue
        layers = np.flatnonzero((PLANE_Z >= min(a[2], b[2])) & (PLANE_Z <= max(a[2], b[2])))
        if not len(layers):
            continue
        xy = a[:2]+(PLANE_Z[layers]-a[2])[:, None]*(b[:2]-a[:2])/dz
        strips = strip_index(xy)
        for layer, (x, y) in zip(layers, strips):
            if x < 0 or y < 0:
                continue
            view = 0 if layer % 2 else 1
            views[view].add((int(layer//2), int(x if view == 0 else y)))
    return [sorted(v) for v in views]


def visible_truth(steps, label, xz, yz):
    ids = [1]+[c['track_id'] for c in label['candidates'] if c['long']]
    projections = {i: trajectory_views(steps[steps[:, 0] == i]) for i in ids}
    tracks = []
    for track_id in ids:
        views = []
        for axis, image in enumerate((xz, yz)):
            samples = projections[track_id][axis]
            others = {tuple(p) for i in ids if i != track_id for p in projections[i][axis]}
            supported = [(r, s) for r, s in samples if np.any(image[r, max(0, s-1):min(96, s+2)] > .01)]
            separated = [(r, s) for r, s in supported if not any((r, t) in others for t in range(s-1, s+2))]
            points = np.array([[PLANE_Z[2*r+(1-axis)], STRIP_CENTERS[s]] for r, s in supported])
            residual = None
            if len(np.unique(points[:, 0])) >= 3 if len(points) else False:
                fit = np.polyfit(points[:, 0], points[:, 1], 1)
                residual = float(np.quantile(np.abs(np.polyval(fit, points[:, 0])-points[:, 1]), .9))
            views.append(dict(samples=samples, supported_planes=len({r for r, _ in supported}),
                              separated_planes=len({r for r, _ in separated}), straightness_p90_mm=residual))
        visible = all(v['supported_planes'] >= 3 and v['separated_planes'] >= 2 for v in views)
        straight = all(v['straightness_p90_mm'] is not None and v['straightness_p90_mm'] <= 3.66 for v in views)
        tracks.append(dict(track_id=track_id, primary=track_id == 1, views=views,
                           visible=visible, approximately_straight=straight))
    branches = [t for t in tracks if not t['primary']]
    explained = []
    for axis, image in enumerate((xz, yz)):
        mask = np.zeros_like(image, bool)
        for views in projections.values():
            for row, strip in views[axis]:
                mask[row, max(0, strip-1):min(96, strip+2)] = True
        active = image > .01
        explained.append(float((active & mask).sum()/max(active.sum(), 1)))
    return dict(tracks=tracks, visible_branches=sum(t['visible'] for t in branches),
                explained_strip_fraction=explained,
                simple_star=(2 <= len(branches) <= 3 and min(explained) >= .7 and
                             all(t['visible'] and t['approximately_straight'] for t in tracks)))
