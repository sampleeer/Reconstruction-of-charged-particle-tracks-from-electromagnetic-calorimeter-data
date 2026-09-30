"""Plot two measured views and Geant4 truth versus reconstructed volume."""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import truth_from_hits


def _points(ax, volume, title):
    coords = np.argwhere(volume > 0)
    if len(coords):
        values = volume[tuple(coords.T)]
        ax.scatter(coords[:, 1], coords[:, 2], coords[:, 0],
                   c=values, s=8 + 22 * values / max(values.max(), 1e-12),
                   cmap="viridis", alpha=0.8)
    ax.set(xlim=(0, 95), ylim=(0, 95), zlim=(0, 21),
           xlabel="X", ylabel="Y", zlabel="paired layer", title=title)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path, help=".npz from reconstruction.run_hits")
    p.add_argument("data", type=Path, help="trusted original Geant4 .npy")
    p.add_argument("--event-id", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = np.load(args.result)
    payload = np.load(args.data, allow_pickle=True).item()
    truth = truth_from_hits(payload, args.event_id)
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax1.imshow(result["xz"], origin="lower", aspect="auto")
    ax1.set(xlabel="X strip", ylabel="paired layer", title="Measured XZ")
    ax2.imshow(result["yz"], origin="lower", aspect="auto")
    ax2.set(xlabel="Y strip", ylabel="paired layer", title="Measured YZ")
    _points(ax3, truth, "Geant4 hits (paired planes, approximate truth)")
    _points(ax4, result["transport"], "Reconstruction from the two views")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
