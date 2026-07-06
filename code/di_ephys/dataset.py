"""Self-supervised DeepInterpolation dataset for a slice of ecephys traces.

A slice of one AP recording is read into memory as a (T, C) float32 array and
z-scored per channel. Each training sample is:

    context : (pre + post, C)   frames [t-omission-pre .. t-omission-1]
                                     + [t+omission+1 .. t+omission+post]
    target  : (1, C)            the center frame at time t

The center frame (plus an `omission` gap on each side) is excluded from the
context so its noise is independent of the network input.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def zscore_per_channel(traces: np.ndarray, eps: float = 1e-3):
    """Return (normed, mean, std) with per-channel (axis 0) statistics."""
    traces = np.asarray(traces, dtype=np.float32)
    mean = traces.mean(axis=0, keepdims=True)
    std = traces.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (traces - mean) / std, mean, std


def context_offsets(pre: int, post: int, omission: int) -> np.ndarray:
    """Sample offsets (relative to center t) used as context frames."""
    left = np.arange(-(omission + pre), -omission)          # pre frames
    right = np.arange(omission + 1, omission + post + 1)     # post frames
    return np.concatenate([left, right]).astype(np.int64)


def split_centers(t_lo: int, t_hi: int, pre: int, post: int, omission: int,
                  val_frac: float = 0.2, val_gap: int = 0):
    """Valid center indices in [t_lo, t_hi), split train/val by time.

    A window centered at t needs [t-omission-pre, t+omission+post] in range.
    Val is the final `val_frac` of the range; a `val_gap` guard band between
    train and val prevents their windows from overlapping (no leakage).
    """
    margin = omission + max(pre, post)
    first = t_lo + margin
    last = t_hi - margin                       # exclusive
    n = max(0, last - first)
    n_val = int(n * val_frac)
    split = last - n_val
    train = np.arange(first, max(first, split - val_gap), dtype=np.int64)
    val = np.arange(split, last, dtype=np.int64)
    return train, val


def build_channel_grid(locations, x_tol: float = 1.0, y_tol: float = 1.0,
                       compact: bool = False):
    """Map channels onto a 2-D probe grid from their physical (x, y) locations.

    Neuropixels contacts sit on a narrow lattice (depth x width); this returns
    the mapping needed to convolve over that real geometry instead of the
    channel-index axis. Columns come from the distinct x positions, rows from
    the distinct y (depth) positions, binned with a tolerance. Returns a dict:

        H, W       grid shape (rows = depth, cols = width)
        flat_pos   (C,) int, flat grid index (row * W + col) per channel
        row, col   (C,) int per-channel grid coordinates
        mask       (H, W) float, 1 where a real contact sits
        fill       fraction of occupied cells (1.0 for a full 2-column probe)
        x_um,y_um  the sorted unique column / row coordinates
    """
    locs = np.asarray(locations, dtype=np.float64)
    if locs.ndim != 2 or locs.shape[1] < 2:
        raise ValueError(f"locations must be (C, >=2); got {locs.shape}")
    x, y = locs[:, 0], locs[:, 1]

    def _axis(v, tol):
        uniq = []
        for val in np.sort(v):
            if not uniq or abs(val - uniq[-1]) > tol:
                uniq.append(float(val))
        uniq = np.asarray(uniq)
        idx = np.abs(v[:, None] - uniq[None, :]).argmin(axis=1)
        return uniq, idx.astype(np.int64)

    xs, col = _axis(x, x_tol)
    ys, row = _axis(y, y_tol)
    H = int(len(ys))
    if compact:
        # dense columns: within each depth row, rank the contacts by x so a
        # half-empty checkerboard (e.g. NP1 -> W=4, 50% filled) collapses to a
        # full 2-column grid: ~2x fewer cells and no compute wasted on empties.
        col = np.zeros(x.shape[0], dtype=np.int64)
        for r in range(H):
            sel = np.where(row == r)[0]
            col[sel] = np.argsort(np.argsort(x[sel]))
        W = int(col.max()) + 1
        xs = np.arange(W, dtype=np.float64)
    else:
        xs, col = _axis(x, x_tol)
        W = int(len(xs))
    flat = (row * W + col).astype(np.int64)
    if np.unique(flat).size != flat.size:
        raise ValueError("colliding grid cells -- adjust x_tol/y_tol or check the probe geometry")
    mask = np.zeros((H, W), dtype=np.float32)
    mask[row, col] = 1.0
    return {"H": H, "W": W, "flat_pos": flat, "row": row.astype(np.int64),
            "col": col.astype(np.int64), "mask": mask,
            "fill": float(mask.mean()), "x_um": xs, "y_um": ys}


class EphysWindowDataset(Dataset):
    def __init__(self, traces: np.ndarray, centers: np.ndarray,
                 pre: int, post: int, omission: int):
        # traces: (T, C) float32, already normalized
        self.traces = torch.from_numpy(np.ascontiguousarray(traces, dtype=np.float32))
        self.centers = np.asarray(centers, dtype=np.int64)
        self.offsets = torch.from_numpy(context_offsets(pre, post, omission))
        self.n_context = int(self.offsets.numel())

    def __len__(self):
        return int(self.centers.shape[0])

    def __getitem__(self, i):
        t = int(self.centers[i])
        idx = self.offsets + t                     # (n_context,)
        context = self.traces[idx]                 # (n_context, C)
        target = self.traces[t:t + 1]              # (1, C)
        return context, target
