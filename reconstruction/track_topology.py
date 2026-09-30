"""Projection-only finite track extraction, independent of a vertex seed.

Development version: strips are clustered per plane; deterministic RANSAC
finds finite line segments. Several segments can describe a bent track.
No truth trajectory or event label is accepted by this module.
"""
from itertools import combinations
import numpy as np
from .physical import PLANE_Z, STRIP_CENTERS


def cluster_points(image, parity, threshold=.01):
    points = []
    for row, energy in enumerate(np.asarray(image)):
        active = np.flatnonzero(energy > threshold)
        for group in np.split(active, np.flatnonzero(np.diff(active) > 1)+1):
            if not len(group):
                continue
            # A centroid per contiguous cluster prevents a broad deposit from
            # receiving as many votes as several independent track planes.
            weights = np.minimum(energy[group], .5)
            value = np.average(STRIP_CENTERS[group], weights=weights)
            points.append([PLANE_Z[2*row+parity], value, float(energy[group].sum()), row])
    return np.asarray(points, float).reshape(-1, 4)


def extract_tracks(image, parity, *, max_tracks=7, tolerance_mm=2.8, min_planes=3):
    points = cluster_points(image, parity)
    remaining = np.ones(len(points), bool)
    tracks = []
    rng = np.random.default_rng(904)
    for _ in range(max_tracks):
        ids = np.flatnonzero(remaining)
        if len(ids) < 2:
            break
        # A junction cluster can support two tracks. After finding a track,
        # permit one shared ENDPOINT, while requiring >=2 new planes.
        pool = np.arange(len(points)) if tracks else ids
        pairs = np.array(list(combinations(pool, 2)), int).reshape(-1, 2)
        if not len(pairs):
            break
        pairs = pairs[np.abs(points[pairs[:, 0], 0]-points[pairs[:, 1], 0]) >= 16.]
        pairs = pairs[remaining[pairs[:, 0]] | remaining[pairs[:, 1]]]
        if len(pairs) > 1000:
            pairs = pairs[rng.choice(len(pairs), 1000, replace=False)]
        if not len(pairs):
            break
        z, value = points[:, 0], points[:, 1]
        slopes = (value[pairs[:, 1]]-value[pairs[:, 0]])/(z[pairs[:, 1]]-z[pairs[:, 0]])
        intercepts = value[pairs[:, 0]]-slopes*z[pairs[:, 0]]
        residual = np.abs(slopes[:, None]*z+intercepts[:, None]-value)
        # Keep the original disjoint fit as a candidate too: a line may
        # intersect already assigned tracks at interior points. Those points
        # must not cause rejection of its otherwise valid fresh support.
        supports = np.concatenate(((residual <= tolerance_mm) & remaining,
                                   residual <= tolerance_mm))
        best = None
        for k, support in enumerate(supports):
            k %= len(residual)
            members = np.flatnonzero(support)
            # One point per layer, preferring the closest cluster.
            unique = [m[np.argmin(residual[k, m])]
                      for row in np.unique(points[members, 3])
                      if len(m := members[points[members, 3] == row])]
            unique = np.asarray(unique, int)
            if len(unique) < min_planes:
                continue
            unique = unique[np.argsort(points[unique, 3])]
            for segment in np.split(unique, np.flatnonzero(np.diff(points[unique, 3]) > 3)+1):
                if len(segment) < min_planes:
                    continue
                reused = np.flatnonzero(~remaining[segment])
                fresh = int(remaining[segment].sum())
                if fresh < 2 or len(reused) > 1 or (len(reused) and reused[0] not in (0, len(segment)-1)):
                    continue
                holes = (points[segment[-1], 3]-points[segment[0], 3]+1)-len(segment)
                score = fresh+.25*len(reused)-.3*holes-.05*np.mean(residual[k, segment])
                if best is None or score > best[0]:
                    best = score, segment
        if best is None:
            break
        members = best[1]
        slope, intercept = np.polyfit(z[members], value[members], 1)
        # Reject a poor refit instead of silently extending it to boundaries.
        keep = np.abs(slope*z[members]+intercept-value[members]) <= tolerance_mm
        members = members[keep]
        if len(members) < min_planes:
            break
        tracks.append(dict(slope=float(slope), intercept=float(intercept),
                           z_min=float(z[members].min()), z_max=float(z[members].max()),
                           n_planes=len(members), point_ids=members.tolist(),
                           rms_mm=float(np.sqrt(np.mean((slope*z[members]+intercept-value[members])**2)))))
        remaining[members] = False
    return dict(points=points.tolist(), tracks=tracks, unmatched_points=int(remaining.sum()))


def vertex_proposals(xview, yview):
    def intersections(view):
        tracks = view['tracks']
        if not tracks:
            return []
        # Incoming candidate must begin near the detector entrance. Choose
        # the longest among tracks beginning in the first two view planes.
        incoming = [i for i, t in enumerate(tracks) if t['z_min'] <= 15.]
        if not incoming:
            return []
        primary = max(incoming, key=lambda i: tracks[i]['n_planes'])
        a = tracks[primary]
        found = []
        for j, b in enumerate(tracks):
            if j == primary or abs(a['slope']-b['slope']) < .025:
                continue
            z = (b['intercept']-a['intercept'])/(a['slope']-b['slope'])
            if not 0 <= z <= 176.08:
                continue
            # A secondary can be almost collinear with the incoming track;
            # its fitted line may continue past the interaction. Do not
            # require the incoming candidate to end at the proposed vertex.
            if not a['z_min']-8.09 <= z <= a['z_max']+16.18:
                continue
            if min(abs(z-b['z_min']), abs(z-b['z_max'])) > 24:
                continue
            found.append((float(z), primary, j))
        return found
    candidates = []
    for zx, ix, bx in intersections(xview):
        for zy, iy, by in intersections(yview):
            if abs(zx-zy) > 16.18:
                continue
            z = (zx+zy)/2
            tx, ty = xview['tracks'][ix], yview['tracks'][iy]
            vertex = [tx['intercept']+tx['slope']*z, ty['intercept']+ty['slope']*z, z]
            if np.max(np.abs(vertex[:2])) > 123:
                continue
            candidates.append(dict(vertex_mm=vertex, z_disagreement_mm=abs(zx-zy),
                                   source_track_ids=[ix, bx, iy, by]))
    candidates.sort(key=lambda r: r['z_disagreement_mm'])
    unique = []
    for c in candidates:
        if not any(np.linalg.norm(np.array(c['vertex_mm'])-u['vertex_mm']) < 5 for u in unique):
            unique.append(c)
    return unique[:4]


def infer_topology(xz, yz):
    xview, yview = extract_tracks(xz, 1), extract_tracks(yz, 0)
    return dict(x=xview, y=yview, vertex_candidates=vertex_proposals(xview, yview))
