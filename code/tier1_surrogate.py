"""Tier-1 surrogate scorer for DeepInterpolation architecture/training search.

Goal: rank DI checkpoints by *how extractable they make the hybrid ground-truth
units* — WITHOUT running a spike sorter. Because the hybrid GT gives us exact
spike times, we only need to denoise the short windows around GT spikes (plus a
sample of background windows for the noise/false-alarm baseline), not the whole
recording. That turns a ~10 h benchmark into seconds–minutes per checkpoint.

Two per-unit scores are reported for the raw recording and the denoised one:

  * template SNR  = peak-to-peak of the unit's average waveform (its own template,
                    per domain) on its peak channel, divided by the background
                    noise sigma in that domain.
  * matched-filter d' / AUC = separability of spike vs background detection scores
                    using a FIXED filter (the *raw* template). Because the filter
                    is the same for both domains, a rise means denoising made the
                    true spike shape more detectable against noise; a fall flags
                    shape distortion. Captures both the recall and precision axes.

Usage
-----
Real run (needs torch + di_ephys + a DI checkpoint):
    python tier1_surrogate.py --recording <si_recording> --gt <si_sorting> \
        --checkpoint checkpoints/best_model.pt --device cuda --out scores.csv

Validate the harness with no external data / no torch:
    python tier1_surrogate.py --selftest
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# make di_ephys importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# scoring core (spike-sorter-free; operates on any SpikeInterface recordings)
# ---------------------------------------------------------------------------
def _extract(recording, times, nbefore, nafter):
    """Stack (n_times, T, n_channels) waveforms by denoising/reading only the
    requested windows. For a lazy DI recording each ``get_traces`` call denoises
    just that window (all channels, as the fold model needs full-probe context)."""
    T = nbefore + nafter
    C = recording.get_num_channels()
    out = np.empty((len(times), T, C), dtype=np.float32)
    for i, t in enumerate(times):
        tr = recording.get_traces(
            segment_index=0, start_frame=int(t) - nbefore, end_frame=int(t) + nafter
        )
        out[i] = np.asarray(tr, dtype=np.float32)
    return out


def _dprime(hit, null):
    hit = np.asarray(hit, float); null = np.asarray(null, float)
    denom = np.sqrt(0.5 * (hit.var() + null.var()))
    return float((hit.mean() - null.mean()) / max(denom, 1e-9))


def _auc(hit, null):
    try:
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(hit, null, alternative="greater").statistic
        return float(u / (len(hit) * len(null)))
    except Exception:
        return float("nan")


def compute_surrogate(raw_rec, denoised_rec, gt_sorting, n_spikes=200, n_bg=500,
                      ms_before=1.5, ms_after=2.5, peak_frac=0.5, max_channels=24,
                      max_units=None, seed=0, verbose=True):
    """Return a per-GT-unit DataFrame of raw vs denoised SNR and matched-filter d'."""
    fs = raw_rec.get_sampling_frequency()
    nbefore = int(round(ms_before * fs / 1000.0))
    nafter = int(round(ms_after * fs / 1000.0))
    ntot = raw_rec.get_num_samples(segment_index=0)
    guard = nbefore + nafter + 64
    rng = np.random.default_rng(seed)

    unit_ids = list(gt_sorting.unit_ids)
    if max_units is not None:
        unit_ids = unit_ids[:max_units]

    # all GT spike times, so background windows avoid real spikes (clean noise floor)
    all_spk = (np.sort(np.concatenate(
        [np.asarray(gt_sorting.get_unit_spike_train(u, segment_index=0)) for u in unit_ids]))
        if unit_ids else np.array([], dtype=np.int64))

    def _sample_background(n):
        keep, tries = [], 0
        while len(keep) < n and tries < 50:
            c = rng.integers(guard, ntot - guard, size=2 * n)
            if all_spk.size:
                idx = np.searchsorted(all_spk, c)
                dl = c - all_spk[np.clip(idx - 1, 0, all_spk.size - 1)]
                dr = all_spk[np.clip(idx, 0, all_spk.size - 1)] - c
                d = np.minimum(np.where(idx > 0, dl, 1 << 30),
                               np.where(idx < all_spk.size, dr, 1 << 30))
                c = c[d > (nbefore + nafter)]
            keep.extend(c.tolist())
            tries += 1
        return np.asarray(keep[:n], dtype=np.int64)

    bg = _sample_background(n_bg)
    if verbose:
        print(f"[surrogate] fs={fs:.0f}  window=({nbefore},{nafter})  "
              f"denoising {len(bg)} background windows once...", flush=True)
    raw_bg_full = _extract(raw_rec, bg, nbefore, nafter)          # (n_bg,T,C)
    deep_bg_full = _extract(denoised_rec, bg, nbefore, nafter)    # (n_bg,T,C)

    rows = []
    for k, uid in enumerate(unit_ids):
        st = np.asarray(gt_sorting.get_unit_spike_train(uid, segment_index=0))
        st = st[(st > guard) & (st < ntot - guard)]
        if st.size == 0:
            continue
        sel = st if st.size <= n_spikes else rng.choice(st, n_spikes, replace=False)

        raw_wf = _extract(raw_rec, sel, nbefore, nafter)          # (n,T,C)
        raw_tmpl = raw_wf.mean(0)                                 # (T,C)
        pp = raw_tmpl.max(0) - raw_tmpl.min(0)                    # (C,)
        peak = int(np.argmax(pp))
        ch = np.where(pp >= peak_frac * pp[peak])[0]
        if ch.size > max_channels:                               # keep strongest channels
            ch = ch[np.argsort(pp[ch])[::-1][:max_channels]]
        if peak not in ch:
            ch = np.append(ch, peak)
        pc = int(np.where(ch == peak)[0][0])                     # peak index within ch

        deep_wf = _extract(denoised_rec, sel, nbefore, nafter)[:, :, ch]   # (n,T,|ch|)
        raw_wf_ch = raw_wf[:, :, ch]
        raw_bg_ch = raw_bg_full[:, :, ch]
        deep_bg_ch = deep_bg_full[:, :, ch]

        raw_tmpl_ch = raw_wf_ch.mean(0)                          # (T,|ch|)
        deep_tmpl_ch = deep_wf.mean(0)

        # template SNR on the peak channel (self-template, per-domain noise)
        snr_raw = float((raw_tmpl_ch[:, pc].max() - raw_tmpl_ch[:, pc].min())
                        / max(raw_bg_ch[:, :, pc].std(), 1e-9))
        snr_deep = float((deep_tmpl_ch[:, pc].max() - deep_tmpl_ch[:, pc].min())
                         / max(deep_bg_ch[:, :, pc].std(), 1e-9))

        # matched-filter detectability with a FIXED filter (the raw template)
        f = raw_tmpl_ch.reshape(-1)
        f = f / max(np.linalg.norm(f), 1e-9)
        proj = lambda w: w.reshape(w.shape[0], -1) @ f
        h_raw, n_raw = proj(raw_wf_ch), proj(raw_bg_ch)
        h_deep, n_deep = proj(deep_wf), proj(deep_bg_ch)

        rows.append(dict(
            unit_id=uid, n_spikes=int(sel.size), peak_ch=peak, n_ch=int(ch.size),
            snr_raw=snr_raw, snr_deep=snr_deep, dsnr=snr_deep - snr_raw,
            dprime_raw=_dprime(h_raw, n_raw), dprime_deep=_dprime(h_deep, n_deep),
            auc_raw=_auc(h_raw, n_raw), auc_deep=_auc(h_deep, n_deep),
        ))
        if verbose:
            r = rows[-1]
            print(f"[surrogate] unit {uid} ({k+1}/{len(unit_ids)}): "
                  f"SNR {r['snr_raw']:.2f}->{r['snr_deep']:.2f}  "
                  f"d' {r['dprime_raw']:.2f}->{r['dprime_deep']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["ddprime"] = df["dprime_deep"] - df["dprime_raw"]
        df["dauc"] = df["auc_deep"] - df["auc_raw"]
    return df


def summarize(df):
    if df.empty:
        return "no units scored"
    m = df[["snr_raw", "snr_deep", "dprime_raw", "dprime_deep", "auc_raw", "auc_deep"]].mean()
    return (
        f"units={len(df)}\n"
        f"  SNR   raw {m.snr_raw:.3f} -> deep {m.snr_deep:.3f}  (Δ {m.snr_deep - m.snr_raw:+.3f})\n"
        f"  d'    raw {m.dprime_raw:.3f} -> deep {m.dprime_deep:.3f}  (Δ {m.dprime_deep - m.dprime_raw:+.3f})\n"
        f"  AUC   raw {m.auc_raw:.3f} -> deep {m.auc_deep:.3f}  (Δ {m.auc_deep - m.auc_raw:+.3f})\n"
        f"  units with d' improved: {(df.dprime_deep > df.dprime_raw).sum()}/{len(df)}"
    )


# ---------------------------------------------------------------------------
# data loading + checkpoint sweep
# ---------------------------------------------------------------------------
def load_aind_hybrid(folder):
    """Load [(name, recording, gt_sorting), ...] from an AIND hybrid folder
    (``gt_*.pkl/json`` GT sortings + matching ``job_*.json/pkl`` carrying a
    ``recording_dict``), as produced by aind-ephys-hybrid-generation."""
    import json
    import pickle
    import spikeinterface as si
    folder = Path(folder)
    pairs = []
    for gt_file in sorted(p for p in folder.iterdir() if p.name.startswith("gt_")):
        name = gt_file.name[len("gt_"):]
        for ext in (".pkl", ".json"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        try:
            gt = si.load(gt_file)
        except Exception:
            gt = si.load(gt_file, base_folder=folder)
        rec = None
        for jext in (".json", ".pkl"):
            jf = folder / f"job_{name}{jext}"
            if jf.is_file():
                d = (json.load(open(jf)) if jext == ".json"
                     else pickle.load(open(jf, "rb")))
                rec = si.load(d["recording_dict"], base_folder=folder)
                break
        if rec is None:
            raise FileNotFoundError(f"no job_{name}.(json|pkl) beside {gt_file.name}")
        pairs.append((name, rec, gt))
    if not pairs:
        raise FileNotFoundError(f"no gt_* files found in {folder}")
    return pairs


def sweep_checkpoints(raw, gt, checkpoint_paths, device="cuda", batch_size=256,
                      n_spikes=200, n_bg=500, ms_before=1.5, ms_after=2.5,
                      max_units=None, seed=0, per_unit_dir=None):
    """Score every checkpoint on (raw, gt); return a ranking (by mean denoised
    matched-filter d'). raw-domain columns are checkpoint-independent and repeat
    across rows as a sanity check."""
    from di_ephys.inference import deepinterpolate
    rows = []
    for cp in checkpoint_paths:
        cp = Path(cp)
        print(f"\n===== {cp.name} =====", flush=True)
        denoised = deepinterpolate(raw, str(cp), device=device, batch_size=batch_size)
        df = compute_surrogate(raw, denoised, gt, n_spikes=n_spikes, n_bg=n_bg,
                               ms_before=ms_before, ms_after=ms_after,
                               max_units=max_units, seed=seed, verbose=False)
        if per_unit_dir is not None:
            Path(per_unit_dir).mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(per_unit_dir) / f"{cp.stem}.csv", index=False)
        m = df[["snr_raw", "snr_deep", "dprime_raw", "dprime_deep", "auc_deep"]].mean()
        rows.append(dict(
            checkpoint=cp.name, n_units=len(df),
            snr_raw=m.snr_raw, snr_deep=m.snr_deep, dsnr=m.snr_deep - m.snr_raw,
            dprime_raw=m.dprime_raw, dprime_deep=m.dprime_deep,
            ddprime=m.dprime_deep - m.dprime_raw, auc_deep=m.auc_deep,
            units_dprime_up=int((df.dprime_deep > df.dprime_raw).sum()),
        ))
        print(f"  mean d' {m.dprime_raw:.2f}->{m.dprime_deep:.2f} "
              f"(\u0394{m.dprime_deep - m.dprime_raw:+.2f})  "
              f"SNR {m.snr_raw:.2f}->{m.snr_deep:.2f}", flush=True)
    return (pd.DataFrame(rows)
            .sort_values("dprime_deep", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _selftest():
    """Validate the scorer end to end with a small synthetic GT recording and a
    trivial band-pass 'denoiser' (no torch, no checkpoint, no external data)."""
    import spikeinterface as si
    import spikeinterface.preprocessing as spre
    print("[selftest] generating synthetic ground-truth recording...")
    rec, sorting = si.generate_ground_truth_recording(
        durations=[60.0], sampling_frequency=30000.0, num_channels=32,
        num_units=6, seed=0,
    )
    denoised = spre.bandpass_filter(rec, freq_min=300, freq_max=6000)   # stand-in denoiser
    df = compute_surrogate(rec, denoised, sorting, n_spikes=100, n_bg=200,
                           max_units=6, seed=0)
    print("\n" + df.round(3).to_string())
    print("\n" + summarize(df))
    assert not df.empty and np.isfinite(df[["snr_raw", "snr_deep"]].values).all()
    print("\n[selftest] OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run synthetic self-test and exit")
    # data source: an AIND hybrid folder, OR an explicit recording+GT pair
    ap.add_argument("--hybrid", help="AIND hybrid folder (gt_*.pkl/json + job_*.json/pkl)")
    ap.add_argument("--hybrid-index", type=int, default=0,
                    help="which (recording,GT) pair from --hybrid to score")
    ap.add_argument("--recording", help="SpikeInterface-loadable hybrid recording")
    ap.add_argument("--gt", help="SpikeInterface GT sorting (hybrid injected units)")
    # checkpoint: a single one, OR a directory to sweep + rank
    ap.add_argument("--checkpoint", help="a single di_ephys best_model.pt")
    ap.add_argument("--checkpoints-dir", help="sweep + rank every *.pt in this directory")
    ap.add_argument("--per-unit-dir", help="[sweep] also write each checkpoint's per-unit CSV here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-spikes", type=int, default=200)
    ap.add_argument("--n-bg", type=int, default=500)
    ap.add_argument("--ms-before", type=float, default=1.5)
    ap.add_argument("--ms-after", type=float, default=2.5)
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tier1_scores.csv")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    import spikeinterface as si
    from di_ephys.inference import deepinterpolate

    # --- resolve the (raw, GT) pair ---
    if args.hybrid:
        pairs = load_aind_hybrid(args.hybrid)
        print(f"[surrogate] {len(pairs)} (recording,GT) pair(s) in {args.hybrid}")
        name, raw, gt = pairs[args.hybrid_index]
        print(f"[surrogate] using pair[{args.hybrid_index}]: {name}")
    elif args.recording and args.gt:
        raw, gt = si.load(args.recording), si.load(args.gt)
    else:
        ap.error("provide --hybrid <folder>  OR  --recording <r> --gt <g>  (or --selftest)")

    print(f"[surrogate] {raw.get_num_channels()} ch, "
          f"{raw.get_num_samples(0) / raw.get_sampling_frequency():.0f}s, "
          f"{len(gt.unit_ids)} GT units")

    # --- score a single checkpoint, or sweep + rank a directory ---
    if args.checkpoints_dir:
        cps = sorted(Path(args.checkpoints_dir).glob("*.pt"))
        if not cps:
            ap.error(f"no *.pt found in {args.checkpoints_dir}")
        print(f"[surrogate] sweeping {len(cps)} checkpoint(s)")
        rank = sweep_checkpoints(
            raw, gt, cps, device=args.device, batch_size=args.batch_size,
            n_spikes=args.n_spikes, n_bg=args.n_bg, ms_before=args.ms_before,
            ms_after=args.ms_after, max_units=args.max_units, seed=args.seed,
            per_unit_dir=args.per_unit_dir,
        )
        rank.to_csv(args.out, index=False)
        print("\n=== RANKING (by mean denoised matched-filter d') ===")
        print(rank.round(3).to_string())
        print(f"\n[surrogate] wrote {args.out}")
    elif args.checkpoint:
        denoised = deepinterpolate(raw, args.checkpoint, device=args.device,
                                   batch_size=args.batch_size)
        df = compute_surrogate(raw, denoised, gt, n_spikes=args.n_spikes, n_bg=args.n_bg,
                               ms_before=args.ms_before, ms_after=args.ms_after,
                               max_units=args.max_units, seed=args.seed)
        df.to_csv(args.out, index=False)
        print("\n" + df.round(3).to_string())
        print("\n" + summarize(df))
        print(f"\n[surrogate] wrote {args.out}")
    else:
        ap.error("provide --checkpoint <pt>  OR  --checkpoints-dir <dir>")


if __name__ == "__main__":
    main()
