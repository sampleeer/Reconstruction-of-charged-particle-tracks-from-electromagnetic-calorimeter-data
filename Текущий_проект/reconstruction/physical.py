"""44-plane simulation geometry inferred from development trajectories.

The 8.09 mm repeat, sensitive slabs [0,.38] and [5.81,6.19], 2.44 mm
strips and three sensor centers (-80.5,0,80.5) agree with recurring step
boundaries and single-hit primary crossings in events 0..499 (train only).
These constants still require confirmation against the Geant4 source.
"""

import numpy as np

PLANE_Z = np.column_stack((np.arange(22)*8.09+.19,
                           np.arange(22)*8.09+6.)).ravel()
STRIP_CENTERS = np.concatenate([(np.arange(32)-15.5)*2.44 + center
                                for center in (-80.5, 0., 80.5)])


def strip_index(mm):
    mm = np.asarray(mm)
    nearest = np.abs(mm[..., None]-STRIP_CENTERS).argmin(axis=-1)
    return np.where(np.abs(mm-STRIP_CENTERS[nearest]) <= 1.22+1e-8, nearest, -1)


def peak_vertex(xz, yz):
    z = int(np.argmax(xz.max(axis=1)+yz.max(axis=1)))
    coords = []
    for view in (xz, yz):
        nonempty = np.flatnonzero(view.sum(axis=1))
        row = nonempty[np.argmin(np.abs(nonempty-z))] if len(nonempty) else z
        coords.append(float(STRIP_CENTERS[np.argmax(view[row])]))
    return np.array([*coords, float(PLANE_Z[2*z:2*z+2].mean())])


def measured_image(xz, yz):
    image = np.zeros((44, 96))
    image[1::2], image[0::2] = xz, yz
    return image


def segment_voxels(start, end):
    """Intersect a finite 3D segment with each sensitive slab, including gaps."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end-start
    voxels = set()
    for layer, z in enumerate(PLANE_Z):
        if abs(delta[2]) < 1e-10:
            if abs(start[2]-z) > .19:
                continue
            lo, hi = 0., 1.
        else:
            interval = sorted(((z-.19-start[2])/delta[2], (z+.19-start[2])/delta[2]))
            lo, hi = max(0., interval[0]), min(1., interval[1])
            if lo > hi:
                continue
        count = max(2, int(np.linalg.norm(delta[:2])*(hi-lo)/.3)+2)
        xy = start[:2] + np.linspace(lo, hi, count)[:, None]*delta[:2]
        ij = strip_index(xy)
        voxels.update((layer, int(x), int(y)) for x, y in ij if x >= 0 and y >= 0)
    return sorted(voxels)


def project44(volume):
    """Only the measured coordinate on each physical plane is observable."""
    a = np.asarray(volume)
    return (a[1::2].sum(axis=2), a[0::2].sum(axis=1))
