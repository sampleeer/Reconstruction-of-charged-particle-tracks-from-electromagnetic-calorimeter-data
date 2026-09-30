"""Finite-ray hypotheses on 44 alternating physical planes.

Geometry proposals use only the measured views and an inferred vertex.
Energy association requires an explicit smoothness prior along each ray;
independent arbitrary voxel energies alone cannot resolve the X/Y pairing.
"""

from itertools import permutations
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from .physical import PLANE_Z, STRIP_CENTERS, strip_index, segment_voxels, measured_image, project44


def view_ray(vertex, zv, slope, endz, zs):
    values = vertex + slope*(zs-zv)
    strips = strip_index(values)
    valid = ((zs >= min(zv, endz)-.19) & (zs <= max(zv, endz)+.19) & (strips >= 0))
    mask = np.zeros((22, 96), bool)
    rows = np.flatnonzero(valid)
    mask[rows, strips[rows]] = True
    return mask


def fit_view(image, vertex, zv, count, parity, threshold=.01):
    observed = image > threshold
    zs = PLANE_Z[parity::2]
    points = np.argwhere(observed)
    # Incoming track: scan measured points before the candidate interaction.
    before = [(row, col) for row, col in points if zs[row] < zv-1.]
    primary = (0., np.zeros_like(observed))
    best = -np.inf
    for row, col in before:
        slope = (STRIP_CENTERS[col]-vertex)/(zs[row]-zv)
        mask = view_ray(vertex, zv, slope, 0., zs)
        score = (mask & observed).sum() - .5*(mask & ~observed).sum()
        if score > best:
            best, primary = score, (float(slope), mask)
    remaining = observed & ~primary[1]
    proposals = []
    for row, col in points:
        endz = zs[row]
        if abs(endz-zv) < 2.:
            continue
        slope = (STRIP_CENTERS[col]-vertex)/(endz-zv)
        mask = view_ray(vertex, zv, slope, endz, zs)
        proposals.append((float(slope), float(endz), mask))
    rays = []
    for _ in range(count):
        if not proposals:
            break
        scores = [float((mask & remaining).sum()-.5*(mask & ~observed).sum())
                  for _, _, mask in proposals]
        chosen = int(np.argmax(scores))
        slope, endz, mask = proposals.pop(chosen)
        if scores[chosen] < 1.5:
            break
        rays.append(dict(slope=slope, end_z=endz, score=scores[chosen]))
        remaining &= ~mask
    return primary[0], rays


def energy_on_segments(segments, xz, yz, smoothness=.3, ridge=.001):
    """Nonnegative per-ray plane deposits, coupled by a first-difference prior."""
    column_voxels, links = [], []
    for start, end in segments:
        voxels = np.asarray(segment_voxels(start, end), dtype=int)
        previous = None
        for layer in np.unique(voxels[:, 0]) if len(voxels) else []:
            v = voxels[voxels[:, 0] == layer]
            column = len(column_voxels)
            column_voxels.append(v)
            if previous is not None:
                old, old_z = previous
                links.append((old, column, np.sqrt(8.09/abs(PLANE_Z[layer]-old_z))))
            previous = column, PLANE_Z[layer]
    if not column_voxels:
        raise ValueError("No rays intersect sensitive planes")
    rows, cols, values = [], [], []
    for c, voxels in enumerate(column_voxels):
        for layer, x, y in voxels:
            rows.append(layer*96+(x if layer % 2 else y))
            cols.append(c)
            values.append(1./len(voxels))
    A = coo_matrix((values, (rows, cols)), shape=(44*96, len(column_voxels))).tocsr()
    dr, dc, dv = [], [], []
    for row, (old, new, scale) in enumerate(links):
        dr.extend([row, row]); dc.extend([old, new]); dv.extend([-scale, scale])
    D = coo_matrix((dv, (dr, dc)), shape=(len(links), len(column_voxels))).tocsr()
    b = measured_image(xz, yz).ravel()
    H = A.T @ A + smoothness*(D.T @ D)
    atb = A.T @ b
    def fun(e):
        return float(e @ (H @ e)-2*e @ atb+ridge*(e @ e)+b @ b)
    def jac(e):
        return 2*(H @ e-atb+ridge*e)
    opt = minimize(fun, np.maximum(atb/np.maximum(H.diagonal()+ridge, 1e-8), 0),
                   jac=jac, bounds=[(0., None)]*len(column_voxels), method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    volume = np.zeros((44, 96, 96))
    mask = np.zeros_like(volume, dtype=bool)
    for e, voxels in zip(opt.x, column_voxels):
        np.add.at(volume, tuple(voxels.T), max(e, 0)/len(voxels))
        mask[tuple(voxels.T)] = True
    denom = max(float(b @ b), 1e-12)
    residual = A @ opt.x-b
    px, py = project44(volume)
    return volume, mask, dict(success=bool(opt.success), solver_message=str(opt.message),
                              normalized_energy_residual=float(residual @ residual/denom),
                              normalized_objective=fun(opt.x)/denom,
                              smoothness=smoothness, ridge=ridge,
                              deposited_energy_MeV=float(volume.sum()),
                              measured_energy_MeV=float(b.sum()),
                              forward_operator_error=float(np.max(np.abs(measured_image(px, py).ravel()-A @ opt.x))))


def reconstruct44(xz, yz, vertex_mm, branch_count, *, max_branches=4, smoothness=.3):
    vertex = np.asarray(vertex_mm, float).copy()
    vertex[:2] = np.clip(vertex[:2], -119.54, 119.54)
    vertex[2] = np.clip(vertex[2], 0., 176.08)
    count = min(max(int(branch_count), 0), max_branches)
    sx, xr = fit_view(xz, vertex[0], vertex[2], count, 1)
    sy, yr = fit_view(yz, vertex[1], vertex[2], count, 0)
    # Match forward and backward rays separately. Missing counterparts are
    # explicit; a ray seen in one projection does not define a unique 3D ray.
    xgroups, ygroups = [], []
    unmatched = 0
    for direction in (-1, 1):
        xx = [r for r in xr if np.sign(r["end_z"]-vertex[2]) == direction]
        yy = [r for r in yr if np.sign(r["end_z"]-vertex[2]) == direction]
        n = min(len(xx), len(yy))
        unmatched += abs(len(xx)-len(yy))
        xgroups.extend(xx[:n]); ygroups.extend(yy[:n])
    start = vertex-np.array([sx, sy, 1.])*vertex[2]
    hypotheses = []
    for assignment in permutations(range(len(ygroups))):
        if any(np.sign(x["end_z"]-vertex[2]) != np.sign(ygroups[j]["end_z"]-vertex[2])
               for x, j in zip(xgroups, assignment)):
            continue
        segments = [(start, vertex)]
        for x, j in zip(xgroups, assignment):
            y = ygroups[j]
            endz = max((x["end_z"], y["end_z"]), key=lambda z: abs(z-vertex[2]))
            end = vertex+np.array([x["slope"], y["slope"], 1.])*(endz-vertex[2])
            segments.append((vertex, end))
        try:
            volume, mask, info = energy_on_segments(segments, xz, yz, smoothness)
        except ValueError:
            continue
        hypotheses.append(dict(assignment=list(assignment), segments=[(a.tolist(), b.tolist()) for a, b in segments],
                               energy=volume, mask=mask, **info))
    hypotheses.sort(key=lambda h: h["normalized_objective"])
    if not hypotheses:
        return dict(status="no_intersection", vertex_mm=vertex.tolist(), hypotheses=[])
    best = hypotheses[0]["normalized_objective"]
    near = sum(h["normalized_objective"] <= best + .01*max(best, .001) for h in hypotheses)
    return dict(status="ok", vertex_mm=vertex.tolist(), requested_branches=int(branch_count),
                branch_cap=max_branches, capped=branch_count > max_branches,
                matched_branches=len(xgroups), unmatched_projected_rays=unmatched,
                pairing_count=len(hypotheses), near_optimal_pairings=near,
                relative_pairing_gap=(float((hypotheses[1]["normalized_objective"]-best)/max(best, 1e-12))
                                      if len(hypotheses) > 1 else None),
                hypotheses=hypotheses)
