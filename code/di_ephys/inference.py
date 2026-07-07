"""Apply a trained DeepInterpolation model to a full SpikeInterface recording.

The training pipeline (``di_ephys/train.py``) only reconstructs a short held-out
validation segment. For deployment we need to denoise an entire recording, which
this module does as a lazy SpikeInterface preprocessor:

    denoised = deepinterpolate(recording, checkpoint_path)
    denoised.save(folder=..., format="binary")  # materializes the denoised traces

The model predicts each center frame from a symmetric window of neighbour frames
(plus, for the 3-frame SUPPORT hole, the immediately-adjacent t-1/t+1 frames fed
through the blind-spot branch). ``get_traces`` therefore fetches an extra
``margin`` frames of context on each side of every requested chunk, z-scores with
per-channel statistics estimated once over the whole recording (matching the
training normalization), runs batched GPU inference for the requested centers,
and un-z-scores back into the recording's native units.

The model is geometry-specific (the "fold" architecture folds the probe width
into the feature axis), so the grid is rebuilt from the *inference* recording's
channel locations; ``load_state_dict`` will fail loudly if the probe geometry
does not match the one the checkpoint was trained on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)

from di_ephys.model import build_model
from di_ephys.dataset import context_offsets, build_channel_grid


# ----------------------------------------------------------------------------
# checkpoint + offsets
# ----------------------------------------------------------------------------
def load_checkpoint(checkpoint_path, map_location="cpu"):
    """Load a ``best_model.pt`` produced by di_ephys/train.py.

    Returns (state_dict, config, in_frames).
    """
    ck = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    if "state_dict" not in ck or "config" not in ck:
        raise ValueError(
            f"{checkpoint_path} is not a di_ephys checkpoint "
            f"(keys: {list(ck.keys())[:8]})"
        )
    return ck["state_dict"], dict(ck["config"]), int(ck["in_frames"])


def build_offsets(cfg: dict) -> np.ndarray:
    """Reproduce exactly the context offsets used at training time.

    ``off = context_offsets(pre, post, omission)`` for the neighbour frames,
    followed by the blind-spot center frame(s): ``[0]`` for a 1-frame hole or
    ``[-1, 0, 1]`` for the 3-frame hole. The neighbour frames come first and the
    center frame(s) last, matching ``FoldDeepInterp1D.forward`` (which splits the
    last ``bs_frames`` frames into the hole branch).
    """
    pre, post, om = int(cfg["pre"]), int(cfg["post"]), int(cfg["omission"])
    neigh = context_offsets(pre, post, om)
    blind = bool(cfg.get("blind_spot", False))
    if not blind:
        return neigh.astype(np.int64)
    bs_frames = int(cfg.get("bs_frames", 1))
    bs_off = np.array([-1, 0, 1]) if bs_frames == 3 else np.array([0])
    return np.concatenate([neigh, bs_off]).astype(np.int64)


def _global_zscore_stats(recording, sample_seconds=60.0, n_chunks=20, seed=0):
    """Per-channel mean/std estimated from chunks spread across the recording.

    Uses the recording's *unscaled* (raw) values, matching the training
    normalization (``zscore_per_channel`` on the raw int16 traces). Returns
    ``(mean, std)`` each shaped ``(1, C)`` float32.
    """
    fs = recording.get_sampling_frequency()
    n = recording.get_num_samples(segment_index=0)
    chunk = max(1, int(round(sample_seconds * fs / n_chunks)))
    chunk = min(chunk, n)
    if chunk >= n:
        starts = [0]
        chunk = n
    else:
        rng = np.random.default_rng(seed)
        hi = max(1, n - chunk)
        starts = np.unique(np.linspace(0, hi, n_chunks).astype(np.int64))
    pieces = []
    for s in starts:
        tr = recording.get_traces(
            start_frame=int(s), end_frame=int(s) + chunk,
            segment_index=0, return_scaled=False,
        )
        pieces.append(np.asarray(tr, dtype=np.float32))
    x = np.concatenate(pieces, axis=0)                 # (S, C)
    mean = x.mean(axis=0, keepdims=True)
    std = np.maximum(x.std(axis=0, keepdims=True), 1e-3)
    return mean.astype(np.float32), std.astype(np.float32)


# ----------------------------------------------------------------------------
# SpikeInterface preprocessor
# ----------------------------------------------------------------------------
class DeepInterpolationRecording(BasePreprocessor):
    """Lazy DeepInterpolation denoiser. Output is float32 in the recording's
    native (raw) scale; channel gains/locations/properties are inherited so the
    downstream pipeline sees an ordinary recording."""

    def __init__(self, recording, checkpoint_path, device="cuda", batch_size=256,
                 norm_sample_seconds=60.0, norm_n_chunks=20, seed=0):
        state_dict, cfg, in_frames = load_checkpoint(checkpoint_path, map_location="cpu")

        off = build_offsets(cfg)
        if off.size != in_frames:
            raise ValueError(
                f"offset count {off.size} != checkpoint in_frames {in_frames}; "
                f"config/offset mismatch"
            )

        # rebuild the probe grid from THIS recording's geometry
        locs = np.asarray(recording.get_channel_locations())
        grid = build_channel_grid(locs, compact=bool(cfg.get("geom2d_compact", False)))

        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        model = build_model(cfg, in_frames=in_frames, grid=grid)
        # grid.flat_pos is a geometry-dependent buffer (sized by n_channels): keep the
        # one built from THIS recording rather than forcing the checkpoint's training
        # geometry. All learned weights depend only on the folded width W, so a genuine
        # geometry mismatch (different W) still raises a size error here.
        sd = {k: v for k, v in state_dict.items() if not k.endswith("grid.flat_pos")}
        result = model.load_state_dict(sd, strict=False)
        leftover = [k for k in result.missing_keys if not k.endswith("grid.flat_pos")]
        if leftover or result.unexpected_keys:
            raise RuntimeError(
                f"checkpoint/model mismatch: missing={leftover} "
                f"unexpected={result.unexpected_keys}"
            )
        model.eval().to(dev)

        mean, std = _global_zscore_stats(
            recording, sample_seconds=norm_sample_seconds,
            n_chunks=norm_n_chunks, seed=seed,
        )

        BasePreprocessor.__init__(self, recording, dtype="float32")
        margin = int(max(abs(int(off.min())), abs(int(off.max()))))
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(
                DeepInterpolationSegment(
                    parent_segment, model, dev, off, margin, mean, std, int(batch_size)
                )
            )

        # serializable provenance only (the live model is rebuilt from the path)
        self._kwargs = dict(
            recording=recording,
            checkpoint_path=str(checkpoint_path),
            device=str(device),
            batch_size=int(batch_size),
            norm_sample_seconds=float(norm_sample_seconds),
            norm_n_chunks=int(norm_n_chunks),
            seed=int(seed),
        )


class DeepInterpolationSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, model, device, offsets, margin, mean, std, batch_size):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.model = model
        self.device = device
        self.margin = int(margin)
        self.batch_size = int(batch_size)
        self._offsets = torch.as_tensor(offsets, dtype=torch.long, device=device)
        self._mean = torch.as_tensor(mean, dtype=torch.float32, device=device)   # (1, C)
        self._std = torch.as_tensor(std, dtype=torch.float32, device=device)     # (1, C)

    def get_traces(self, start_frame, end_frame, channel_indices):
        n = self.get_num_samples()
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = n
        start_frame, end_frame = int(start_frame), int(end_frame)
        m = self.margin

        lo, hi = start_frame - m, end_frame + m
        lo_c, hi_c = max(0, lo), min(n, hi)
        parent = self.parent_recording_segment.get_traces(
            start_frame=lo_c, end_frame=hi_c, channel_indices=slice(None)
        )
        parent = np.asarray(parent, dtype=np.float32)
        pad_l, pad_r = lo_c - lo, hi - hi_c
        if pad_l > 0 or pad_r > 0:                      # edge-pad at recording boundaries
            parent = np.pad(parent, ((pad_l, pad_r), (0, 0)), mode="edge")

        t_out = end_frame - start_frame
        with torch.no_grad():
            normed = (torch.from_numpy(parent).to(self.device) - self._mean) / self._std  # (L, C)
            centers = m + torch.arange(t_out, device=self.device, dtype=torch.long)       # into normed
            out = torch.empty((t_out, normed.shape[1]), dtype=torch.float32, device=self.device)
            for i in range(0, t_out, self.batch_size):
                cl = centers[i:i + self.batch_size]                     # (b,)
                idx = cl[:, None] + self._offsets[None, :]              # (b, in_frames)
                ctx = normed[idx]                                       # (b, in_frames, C)
                pred = self.model(ctx)[:, 0]                            # (b, C)
                out[i:i + cl.shape[0]] = pred
            out = out * self._std + self._mean                         # un-z-score
            traces = out.cpu().numpy()

        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return np.ascontiguousarray(traces, dtype=np.float32)


def deepinterpolate(recording, checkpoint_path, device="cuda", batch_size=256,
                    norm_sample_seconds=60.0, norm_n_chunks=20, seed=0):
    """Return a lazy DeepInterpolation-denoised version of ``recording``."""
    return DeepInterpolationRecording(
        recording, checkpoint_path, device=device, batch_size=batch_size,
        norm_sample_seconds=norm_sample_seconds, norm_n_chunks=norm_n_chunks, seed=seed,
    )
