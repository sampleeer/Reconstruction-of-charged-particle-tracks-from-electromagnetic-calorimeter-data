"""A small, testable baseline for PAMELA-style two-view reconstruction.

Array convention is always (z, x, y), with z in 0..21 for paired planes.
The two measured images have shapes (z, x) and (z, y). The geometric fit
uses their binary occupancy only; energy fitting uses the measured values.
Known three-dimensional simulated hits are reserved for evaluation.
"""

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial, prod
import numpy as np
from scipy.optimize import differential_evolution, linprog, minimize
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Geometry:
    start_x: float
    start_y: float
    slope_x: float
    slope_y: float
    vertex_z: float
    # Each branch ends at (x, y, z); z may be before or after the vertex.
    endpoints: tuple[tuple[float, float, float], ...] = ()


def _check_images(xz, yz):
    xz, yz = np.asarray(xz, dtype=float), np.asarray(yz, dtype=float)
    if xz.ndim != 2 or yz.ndim != 2 or xz.shape[0] != yz.shape[0]:
        raise ValueError("Expected images (z,x) and (z,y) with equal z size")
    if not np.all(np.isfinite(xz)) or not np.all(np.isfinite(yz)):
        raise ValueError("Images must contain finite values")
    if np.any(xz < 0) or np.any(yz < 0):
        raise ValueError("Images must be nonnegative")
    return xz, yz


def _line(mask, p0, p1):
    nz, nx, ny = mask.shape
    z0, z1 = int(round(p0[0])), int(round(p1[0]))
    if z0 == z1:
        return
    z = np.arange(z0, z1 + (1 if z1 > z0 else -1), 1 if z1 > z0 else -1)
    f = (z - z0) / (z1 - z0)
    x = np.rint(p0[1] + f * (p1[1] - p0[1])).astype(int)
    y = np.rint(p0[2] + f * (p1[2] - p0[2])).astype(int)
    inside = (z >= 0) & (z < nz) & (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    mask[z[inside], x[inside], y[inside]] = True


def rasterize(geometry: Geometry, shape=(22, 96, 96)):
    """Rasterize the primary ray and secondary branches into a boolean volume."""
    mask = np.zeros(shape, dtype=bool)
    zv = float(geometry.vertex_z)
    vertex = (zv, geometry.start_x + zv * geometry.slope_x,
              geometry.start_y + zv * geometry.slope_y)
    _line(mask, (0.0, geometry.start_x, geometry.start_y), vertex)
    for x, y, z in geometry.endpoints:
        _line(mask, vertex, (z, x, y))
    return mask


def project(volume):
    """Return (XZ, YZ) sums; boolean input produces binary occupancy."""
    a = np.asarray(volume)
    if a.ndim != 3:
        raise ValueError("Expected a (z,x,y) volume")
    if a.dtype == bool:
        return a.any(axis=2), a.any(axis=1)
    return a.sum(axis=2), a.sum(axis=1)


def _chamfer(a, b):
    pa, pb = np.argwhere(a), np.argwhere(b)
    if not len(pa) or not len(pb):
        return 1e3
    return float((cKDTree(pa).query(pb)[0].mean() +
                  cKDTree(pb).query(pa)[0].mean()) / 2)


def _decode(params, n):
    p = np.asarray(params, dtype=float)
    endpoints = tuple(tuple(v) for v in p[5:].reshape(n, 3))
    return Geometry(*p[:5], endpoints)


def _view_rays(image, start, vertex, zv, count):
    """Greedy line voting in one measured projection; used only for seeding."""
    nz = image.shape[0]
    present = image > 0
    remaining = present.copy()
    if zv >= 1:
        z = np.arange(zv + 1)
        v = np.rint(start + (vertex - start) * z / zv).astype(int)
        inside = (v >= 0) & (v < image.shape[1])
        remaining[z[inside], v[inside]] = False
    rays = []
    for _ in range(count):
        best = None
        for endz, endv in np.argwhere(remaining):
            if abs(int(endz) - zv) < 2:
                continue
            zz = np.arange(min(zv, endz), max(zv, endz) + 1)
            vv = np.rint(vertex + (endv - vertex) * (zz - zv) / (endz - zv)).astype(int)
            inside = (vv >= 0) & (vv < image.shape[1])
            zz, vv = zz[inside], vv[inside]
            hits = int(remaining[zz, vv].sum())
            misses = int((~present[zz, vv]).sum())
            score = hits - 0.5 * misses
            if best is None or score > best[0]:
                best = (score, int(endv), int(endz), zz, vv)
        if best is None or best[0] <= 0:
            break
        _, endv, endz, zz, vv = best
        rays.append((endv, endz))
        remaining[zz, vv] = False
    return rays


def _view_mask(start, slope, zv, ends, endzs, shape):
    nz, nv = shape
    mask = np.zeros(shape, bool)
    def segment(z0, v0, z1, v1):
        z0, z1 = int(round(z0)), int(round(z1))
        if z0 == z1: return
        zz = np.arange(min(z0, z1), max(z0, z1) + 1)
        vv = np.rint(v0 + (v1 - v0) * (zz - z0) / (z1 - z0)).astype(int)
        valid = (zz >= 0) & (zz < nz) & (vv >= 0) & (vv < nv)
        mask[zz[valid], vv[valid]] = True
    vertex = start + slope * zv
    segment(0, start, zv, vertex)
    for value, z in zip(ends, endzs):
        segment(zv, vertex, z, value)
    return mask


def _refine_view(image, start, slope, zv, ends, endzs, seed):
    params = np.r_[start, slope, ends]
    nv = image.shape[1]
    bounds = [(max(0, start - 1.5), min(nv - 1, start + 1.5)),
              (max(-2, slope - 0.35), min(2, slope + 0.35))]
    bounds += [(max(0, e - 5), min(nv - 1, e + 5)) for e in ends]
    def objective(p):
        return _chamfer(_view_mask(p[0], p[1], zv, p[2:], endzs, image.shape),
                        image > 0)
    opt = differential_evolution(objective, bounds, x0=params, seed=seed,
                                 maxiter=25, popsize=5, polish=False)
    return min((params, opt.x), key=objective)


def _initial(xz, yz, n):
    nx, ny = xz.shape[1], yz.shape[1]
    first_x = np.flatnonzero(xz[0])
    first_y = np.flatnonzero(yz[0])
    x0 = float(np.median(first_x)) if len(first_x) else nx / 2
    y0 = float(np.median(first_y)) if len(first_y) else ny / 2
    zmax = xz.shape[0] - 1
    zsum = xz.sum(axis=1) + yz.sum(axis=1)
    # A broad shower can have more total energy after the interaction;
    # the brightest paired strips give a better vertex proposal.
    peak = xz.max(axis=1) + yz.max(axis=1)
    zv = int(np.argmax(peak[1:]) + 1) if n else int(np.flatnonzero(zsum)[-1])
    vx = int(np.argmax(xz[zv]))
    vy = int(np.argmax(yz[zv]))
    sx = float(np.clip((vx - x0) / max(zv, 1), -2, 2))
    sy = float(np.clip((vy - y0) / max(zv, 1), -2, 2))
    p = [x0, y0, sx, sy, float(zv)]
    xr = _view_rays(xz, x0, vx, zv, n)
    yr = _view_rays(yz, y0, vy, zv, n)
    for i in range(n):
        endx, endz = xr[i] if i < len(xr) else (vx, zmax)
        endy, _ = yr[i] if i < len(yr) else (vy, endz)
        p.extend([float(endx), float(endy), float(endz)])
    return np.asarray(p)


def fit_geometry(xz, yz, *, max_branches=2, seed=0, maxiter=35,
                 popsize=5, complexity_penalty=0.15):
    """Fit 0..max_branches hypotheses using only binary projections.

    Scores are symmetric projected Chamfer distances plus a stated branch
    penalty. They are model-selection heuristics, not calibrated probabilities.
    """
    xz, yz = _check_images(xz, yz)
    if not np.any(xz) or not np.any(yz):
        raise ValueError("Both images need at least one active strip")
    nz, nx, ny = xz.shape[0], xz.shape[1], yz.shape[1]
    if nz < 2:
        raise ValueError("At least two paired planes are required")
    bx, by = xz > 0, yz > 0
    results = []
    for n in range(max_branches + 1):
        bounds = [(0, nx - 1), (0, ny - 1), (-2, 2), (-2, 2), (1, nz - 1)]
        bounds += [(0, nx - 1), (0, ny - 1), (0, nz - 1)] * n
        def objective(p):
            g = _decode(p, n)
            mx, my = project(rasterize(g, (nz, nx, ny)))
            if not np.any(mx): return 1e3
            return (_chamfer(mx, bx) + _chamfer(my, by)) / 2
        x0 = _initial(xz, yz, n)
        opt = differential_evolution(objective, bounds, x0=x0, seed=seed + n,
                                     maxiter=maxiter, popsize=popsize,
                                     polish=False, updating="immediate")
        # Preserve both the supplied initial guess and the optimizer result.
        candidates = [(objective(x0), x0), (float(opt.fun), opt.x)]
        raw, p = min(candidates, key=lambda item: item[0])
        # Binary rasterization has plateaus. Refining the two observed views
        # separately is cheap and preserves their actual measurement geometry.
        g = _decode(p, n)
        endzs = [v[2] for v in g.endpoints]
        px = _refine_view(xz, g.start_x, g.slope_x, g.vertex_z,
                          [v[0] for v in g.endpoints], endzs, seed + 100 + n)
        py = _refine_view(yz, g.start_y, g.slope_y, g.vertex_z,
                          [v[1] for v in g.endpoints], endzs, seed + 200 + n)
        refined = np.r_[px[0], py[0], px[1], py[1], g.vertex_z,
                        np.array([(x, y, z) for x, y, z in zip(px[2:], py[2:], endzs)]).ravel()]
        candidates.append((objective(refined), refined))
        raw, p = min(candidates, key=lambda item: item[0])
        results.append({"branches": n, "geometry": _decode(p, n),
                        "projection_chamfer": raw,
                        "selection_score": raw + complexity_penalty * n,
                        "evaluations": int(opt.nfev)})
    return min(results, key=lambda r: r["selection_score"]), results


def permutation_candidates(geometry, *, max_candidates=720):
    """Enumerate exact X/Y branch-pairing ambiguities for equal end planes.

    Only branches with the same rounded terminal z can exchange their Y
    endpoints without changing either binary projection. This is a subset of
    all possible ambiguities and is intentionally explicit about its limit.
    """
    endpoints = list(geometry.endpoints)
    groups = {}
    for i, (_, _, z) in enumerate(endpoints):
        groups.setdefault(int(round(z)), []).append(i)
    count = prod(factorial(len(indices)) for indices in groups.values())
    if count > max_candidates:
        raise ValueError(f"{count} permutations exceed max_candidates={max_candidates}")
    choices = [list(permutations(indices)) for indices in groups.values()]
    seen = set()
    for assignment in product(*choices):
        new = endpoints.copy()
        for indices, selected in zip(groups.values(), assignment):
            for target, source in zip(indices, selected):
                x, _, z = endpoints[target]
                new[target] = (x, endpoints[source][1], z)
        key = tuple(new)
        if key not in seen:
            seen.add(key)
            yield Geometry(geometry.start_x, geometry.start_y,
                           geometry.slope_x, geometry.slope_y,
                           geometry.vertex_z, tuple(new))


def fit_energy(mask, xz, yz, *, ridge=0.01):
    """Nonnegative least squares on occupied voxels, with optional L2 ridge.

    Independent XZ/YZ totals are allowed: this finds a compromise and reports
    each residual. It does not pretend inconsistent measurements are exact.
    """
    xz, yz = _check_images(xz, yz)
    mask = np.asarray(mask, dtype=bool)
    nz, nx, ny = mask.shape
    if xz.shape != (nz, nx) or yz.shape != (nz, ny):
        raise ValueError("Mask and projection shapes differ")
    coords = np.argwhere(mask)
    if not len(coords): raise ValueError("Geometry has no occupied voxels")
    k = len(coords)
    j = np.arange(k)
    row_x = coords[:, 0] * nx + coords[:, 1]
    row_y = nz * nx + coords[:, 0] * ny + coords[:, 2]
    A = coo_matrix((np.ones(2 * k),
                    (np.r_[row_x, row_y], np.r_[j, j])),
                   shape=(nz * (nx + ny), k)).tocsr()
    b = np.r_[xz.ravel(), yz.ravel()]
    # Ridge makes otherwise indistinguishable voxel allocations reproducible.
    if ridge < 0: raise ValueError("ridge must be nonnegative")
    ata = A.T @ A
    atb = A.T @ b
    def fun(e):
        return float(e @ (ata @ e) - 2 * e @ atb + b @ b + ridge * (e @ e))
    def jac(e):
        return 2 * (ata @ e - atb + ridge * e)
    x0 = np.maximum(atb / np.maximum(ata.diagonal() + ridge, 1), 0)
    opt = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=[(0, None)] * k,
                   options={"maxiter": 1000, "ftol": 1e-12})
    e = np.maximum(opt.x, 0)
    volume = np.zeros(mask.shape, dtype=float)
    volume[tuple(coords.T)] = e
    px, py = project(volume)
    return volume, {"success": bool(opt.success), "voxels": k,
                    "xz_mse": float(np.mean((px - xz) ** 2)),
                    "yz_mse": float(np.mean((py - yz) ** 2)),
                    "xz_total": float(xz.sum()), "yz_total": float(yz.sum())}


def transport_volume(xz, yz, track_mask, *, continuity=0.25):
    """Sparse per-layer X/Y coupling guided by an inferred track.

    This optional shower/residual model allows energy away from exact line
    voxels. Each layer's unequal measured totals are scaled to their average
    before transport; original totals and raw-view residuals are returned.
    It uses no known 3D hits. All active X/Y strip pairs are candidates.
    """
    xz, yz = _check_images(xz, yz)
    track_mask = np.asarray(track_mask, dtype=bool)
    nz, nx, ny = track_mask.shape
    if xz.shape != (nz, nx) or yz.shape != (nz, ny):
        raise ValueError("Mask and projection shapes differ")
    if continuity < 0: raise ValueError("continuity must be nonnegative")
    volume = np.zeros(track_mask.shape, dtype=float)
    skipped = []
    previous = None
    all_track = np.argwhere(track_mask)
    for z in range(nz):
        xs, ys = np.flatnonzero(xz[z]), np.flatnonzero(yz[z])
        if not len(xs) or not len(ys):
            if len(xs) or len(ys): skipped.append(z)
            continue
        a, b = xz[z, xs], yz[z, ys]
        total = (a.sum() + b.sum()) / 2
        a, b = a * total / a.sum(), b * total / b.sum()
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pairs = np.c_[gx.ravel(), gy.ravel()]
        at_layer = np.argwhere(track_mask[z])
        if not len(at_layer) and len(all_track):
            nearest_z = all_track[np.argmin(abs(all_track[:, 0] - z)), 0]
            at_layer = np.argwhere(track_mask[nearest_z])
        cost = (cKDTree(at_layer).query(pairs)[0] if len(at_layer)
                else np.zeros(len(pairs)))
        if previous is not None and continuity:
            cost = cost + continuity * cKDTree(previous).query(pairs)[0]
        # Independent row sums and all but one column sum remove redundancy.
        m, n = len(xs), len(ys)
        cols = np.arange(m * n)
        row_ids = np.repeat(np.arange(m), n)
        col_ids = np.tile(np.arange(n), m)
        A = coo_matrix((np.r_[np.ones(m * n), np.ones(m * (n - 1))],
                        (np.r_[row_ids, m + col_ids[col_ids < n - 1]],
                         np.r_[cols, cols[col_ids < n - 1]])),
                       shape=(m + n - 1, m * n)).tocsr()
        result = linprog(cost, A_eq=A, b_eq=np.r_[a, b[:-1]],
                         bounds=(0, None), method="highs")
        if not result.success:
            raise RuntimeError(f"Transport failed in layer {z}: {result.message}")
        mass = result.x
        positive = mass > max(float(total) * 1e-10, 1e-12)
        volume[z, pairs[positive, 0], pairs[positive, 1]] = mass[positive]
        previous = pairs[positive]
    px, py = project(volume)
    return volume, {"nonzero_voxels": int(np.count_nonzero(volume)),
                    "skipped_one_view_layers": skipped,
                    "xz_mse_raw": float(np.mean((px - xz) ** 2)),
                    "yz_mse_raw": float(np.mean((py - yz) ** 2))}


def _hits(data, event_id):
    keys = ("event_ID", "layer", "index_along_x", "index_along_y", "energy_release")
    if any(k not in data for k in keys):
        raise KeyError(f"Hit table needs {keys}")
    ids = np.asarray(data["event_ID"]).ravel()
    sel = ids == event_id
    if not np.any(sel): raise ValueError(f"Event {event_id} not present")
    out = [np.asarray(data[k]).ravel()[sel] for k in keys[1:]]
    if any(len(v) != len(out[0]) for v in out):
        raise ValueError("Hit table columns have different lengths")
    return out


def projections_from_hits(data, event_id, *, shape=(22, 96, 96)):
    """Simulated hit table -> strip measurements.

    Per supervisor's 44-plane convention: even planes measure Y and odd
    planes measure X. The unused coordinate of each hit is deliberately ignored.
    """
    layer, x, y, energy = _hits(data, event_id)
    nz, nx, ny = shape
    layer, x, y = (np.asarray(a, dtype=int) for a in (layer, x, y))
    energy = np.asarray(energy, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("Energies must be finite and nonnegative")
    z = layer // 2
    valid_layer = (layer >= 0) & (z < nz)
    xz, yz = np.zeros((nz, nx)), np.zeros((nz, ny))
    # The coordinate perpendicular to each strip plane is not an input.
    odd = valid_layer & (layer % 2 == 1) & (x >= 0) & (x < nx)
    even = valid_layer & (layer % 2 == 0) & (y >= 0) & (y < ny)
    np.add.at(xz, (z[odd], x[odd]), energy[odd])
    np.add.at(yz, (z[even], y[even]), energy[even])
    return xz, yz


def truth_from_hits(data, event_id, *, shape=(22, 96, 96)):
    """Approximate paired-plane Geant4 truth; use for evaluation ONLY."""
    layer, x, y, energy = _hits(data, event_id)
    layer, x, y = (np.asarray(a, dtype=int) for a in (layer, x, y))
    energy = np.asarray(energy, dtype=float)
    z = layer // 2
    nz, nx, ny = shape
    valid = (layer >= 0) & (z < nz) & (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    volume = np.zeros(shape, dtype=float)
    np.add.at(volume, (z[valid], x[valid], y[valid]), energy[valid])
    return volume
