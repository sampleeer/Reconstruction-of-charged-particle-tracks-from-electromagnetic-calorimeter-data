"""Partial 3D reconstruction from independently fitted projected segments.

Returns no_vertex when the geometric intersection criteria fail. This is
an explicit abstention, not a fallback to a learned/true interaction point.
"""
from itertools import permutations
import numpy as np
from .track_topology import infer_topology
from .physical_fit import energy_on_segments


def outgoing(view, primary_id, vertex, axis):
    z = vertex[2]
    rays = []
    for i, t in enumerate(view['tracks']):
        distance = abs(t['intercept']+t['slope']*z-vertex[axis])
        if distance > 4.88:
            continue
        if i == primary_id:
            # Collinear incoming/outgoing tracks may be one fitted segment.
            if t['z_max']-z < 16.18:
                continue
            endz = t['z_max']
        else:
            if min(abs(z-t['z_min']), abs(z-t['z_max'])) > 24:
                continue
            endz = max((t['z_min'], t['z_max']), key=lambda e: abs(e-z))
            if abs(endz-z) < 16.18:
                continue
        rays.append(dict(slope=t['slope'], end_z=endz, distance=distance,
                         n_planes=t['n_planes'], source_track_id=i))
    return sorted(rays, key=lambda r: (-r['n_planes'], r['distance']))[:4]


def reconstruct_topology(xz, yz, topology=None):
    topology = infer_topology(xz, yz) if topology is None else topology
    hypotheses = []
    for c in topology['vertex_candidates']:
        vertex = np.array(c['vertex_mm'])
        ix, _, iy, _ = c['source_track_ids']
        x = outgoing(topology['x'], ix, vertex, 0)
        y = outgoing(topology['y'], iy, vertex, 1)
        xx, yy = [], []
        for direction in (-1, 1):
            gx = [r for r in x if np.sign(r['end_z']-vertex[2]) == direction]
            gy = [r for r in y if np.sign(r['end_z']-vertex[2]) == direction]
            n = min(len(gx), len(gy)); xx.extend(gx[:n]); yy.extend(gy[:n])
        if len(xx) < 2:
            continue
        primary = [topology['x']['tracks'][ix], topology['y']['tracks'][iy]]
        start = vertex-np.array([primary[0]['slope'], primary[1]['slope'], 1])*vertex[2]
        for order in permutations(range(len(yy))):
            if any(np.sign(a['end_z']-vertex[2]) != np.sign(yy[j]['end_z']-vertex[2]) for a, j in zip(xx, order)):
                continue
            segments = [(start, vertex)]
            for a, j in zip(xx, order):
                b = yy[j]; endz = max((a['end_z'], b['end_z']), key=lambda e: abs(e-vertex[2]))
                end = vertex+np.array([a['slope'], b['slope'], 1])*(endz-vertex[2])
                segments.append((vertex, end))
            volume, mask, info = energy_on_segments(segments, xz, yz)
            hypotheses.append(dict(vertex_mm=vertex.tolist(), segments=[(a.tolist(), b.tolist()) for a,b in segments],
                                   x_source_track_ids=[r['source_track_id'] for r in xx],
                                   y_source_track_ids=[yy[j]['source_track_id'] for j in order],
                                   energy=volume, mask=mask, **info))
    hypotheses.sort(key=lambda h: h['normalized_objective'])
    return dict(status='ok' if hypotheses else 'insufficient_connected_tracks' if topology['vertex_candidates'] else 'no_vertex',
                hypotheses=hypotheses, vertex_proposal_count=len(topology['vertex_candidates']))


def serializable(result):
    return {**result, 'hypotheses': [{k:v for k,v in h.items() if k not in ('mask','energy')} for h in result['hypotheses']]}


def reconstruct_hybrid(xz, yz, model, topology=None):
    """Use the new geometric proposal when available, otherwise retain v1.

The route uses inference status only, never reference reconstruction error.
It is a development comparison, not a validated uncertainty policy.
"""
    candidate = reconstruct_topology(xz, yz, topology)
    if candidate['status'] == 'ok':
        return dict(source='segments', **candidate)
    from .star_reconstruct import run_event
    baseline = run_event(model, xz, yz)
    old = baseline['reconstruction']
    return dict(source='v1_fallback', status='ok' if old and old['hypotheses'] else 'no_reconstruction',
                topology_status=candidate['status'], vertex_proposal_count=candidate['vertex_proposal_count'],
                hypotheses=old['hypotheses'] if old else [])
