"""DeepInterpolation inference capsule for the AIND ephys pipeline.

Mirrors the IO contract of ``aind-ephys-compress`` / ``aind-ephys-preprocessing``:
reads the ``*job*.json`` configuration file(s) produced by ``aind-ephys-job-dispatch``
(or the hybrid-generation step), loads each recording with SpikeInterface, applies
a trained DeepInterpolation model, writes the denoised recording to ``../results``
as a flat binary folder (mirroring ``aind-ephys-preprocessing``, so the downstream
sorter memory-maps it directly instead of decompressing a Zarr), and rewrites the
job configuration so the next pipeline step (preprocessing) loads the denoised
recording.

Placement in the hybrid-benchmark pipeline mirrors ``compress``:

    hybrid-generation -> [ deepinterpolation ] -> preprocessing -> spike-sorting

Because the shipped model was trained on *raw* AP-band traces (per-channel
z-scored, no CMR/filter), it operates in the raw domain and runs before
preprocessing.
"""

import os

# limit scipy/blas threads; SpikeInterface handles parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface.core.core_tools import SIJsonEncoder

from di_ephys.inference import deepinterpolate

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-deepinterpolation-inference"
VERSION = "0.1.0"

data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "best_model.pt"


parser = argparse.ArgumentParser(description="DeepInterpolation denoising of AIND Neuropixels data")
parser.add_argument("--checkpoint", default=None,
                    help="Path to a di_ephys best_model.pt checkpoint. "
                         "Defaults to the bundled checkpoint, or the first *.pt found under ../data.")
parser.add_argument("--device", default="cuda", help="torch device (cuda|cpu)")
parser.add_argument("--batch-size", default=256, type=int, help="frames per forward pass")
parser.add_argument("--chunk-duration", default="1s", help="save chunk duration (SpikeInterface)")
parser.add_argument("--norm-sample-seconds", default=60.0, type=float,
                    help="seconds sampled across the recording to estimate per-channel z-score stats")
parser.add_argument("--max-test-recordings", default=1, type=int,
                    help="[standalone] max AP recordings to denoise when no job config is present")
parser.add_argument("--test-duration-s", default=10.0, type=float,
                    help="[standalone] clip each recording to this many seconds (0 = full recording)")


def _resolve_checkpoint(cli_value):
    if cli_value:
        return Path(cli_value)
    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT
    pts = sorted(data_folder.rglob("*.pt"))
    if len(pts) == 1:
        return pts[0]
    raise FileNotFoundError(
        "No checkpoint given, none bundled at "
        f"{DEFAULT_CHECKPOINT}, and {len(pts)} '*.pt' files under ../data "
        "(expected exactly one). Pass --checkpoint."
    )


def _find_ap_recordings(folder):
    """AP-band zarr recordings under a folder (NP1 '*-AP.zarr' or NP2 'ProbeX.zarr')."""
    import re
    zarrs = sorted(str(p) for p in Path(folder).rglob("*.zarr"))
    def _is_ap(p):
        b = os.path.basename(p.rstrip("/"))
        return ("-AP.zarr" in b) or bool(re.search(r"Probe[A-Z]\.zarr$", b))
    ap = [z for z in zarrs if _is_ap(z)]
    return ap or zarrs


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    device = args.device

    # SpikeInterface save is single-process (the model lives on one GPU); the GPU
    # batches frames internally.
    si.set_global_job_kwargs(n_jobs=1, progress_bar=False)

    logging.info("Running DeepInterpolation inference with:")
    logging.info(f"\tCHECKPOINT: {checkpoint_path}")
    logging.info(f"\tDEVICE: {device}")
    logging.info(f"\tBATCH_SIZE: {args.batch_size}")
    logging.info(f"\tCHUNK_DURATION: {args.chunk_duration}")

    job_config_files = [
        p for p in data_folder.rglob("*")
        if (p.suffix in (".json", ".pickle", ".pkl")) and "job" in p.name
        and ".zarr" not in str(p)  # exclude zarr-internal metadata files
    ]
    logging.info(f"Found {len(job_config_files)} job configuration(s)")

    t_all = time.perf_counter()
    for job_config_file in job_config_files:
        if job_config_file.suffix == ".json":
            with open(job_config_file, "r") as f:
                job_config = json.load(f)
        else:
            with open(job_config_file, "rb") as f:
                job_config = pickle.load(f)

        session_name = job_config["session_name"]
        recording_name = job_config["recording_name"]
        recording_dict = job_config["recording_dict"]
        skip_times = job_config.get("skip_times", False)

        try:
            recording = si.load(recording_dict, base_folder=data_folder)
        except Exception as e:
            raise RuntimeError(
                f"Could not load recording {recording_name} from dict ({e}). "
                f"Make sure the mapping is correct!"
            )
        if skip_times:
            recording.reset_times()
        if recording.get_dtype().kind == "u":
            recording = spre.unsigned_to_signed(recording)

        logging.info(f"Denoising: {session_name} - {recording_name}")
        logging.info(f"\t{recording}")

        t0 = time.perf_counter()
        denoised = deepinterpolate(
            recording,
            checkpoint_path,
            device=device,
            batch_size=args.batch_size,
            norm_sample_seconds=args.norm_sample_seconds,
        )
        denoised_saved = denoised.save(
            folder=results_folder / recording_name,
            format="binary",
            chunk_duration=args.chunk_duration,
        )
        elapsed = round(time.perf_counter() - t0, 2)
        logging.info(f"\tDenoised in {elapsed}s -> {recording_name} (binary)")

        # rewrite the job config to point at the denoised recording
        job_config["recording_dict"] = denoised_saved.to_dict(
            recursive=True, relative_to=results_folder
        )
        with open(results_folder / f"{job_config_file.stem}.json", "w") as f:
            json.dump(job_config, f, indent=4, cls=SIJsonEncoder)

    if not job_config_files:
        # Standalone / self-test mode: no pipeline job config present, so denoise raw
        # AP zarr recording(s) found under ../data directly. Validates the capsule end
        # to end on a real recording (attach an ecephys asset and run with no config).
        ap = _find_ap_recordings(data_folder)
        logging.info(f"Standalone mode: found {len(ap)} AP recording(s) under ../data")
        for zpath in ap[: args.max_test_recordings]:
            name = os.path.basename(zpath.rstrip("/"))
            if name.endswith(".zarr"):
                name = name[:-5]
            recording = si.read_zarr(zpath)
            if recording.get_dtype().kind == "u":
                recording = spre.unsigned_to_signed(recording)
            if args.test_duration_s and args.test_duration_s > 0:
                fs = recording.get_sampling_frequency()
                end = min(recording.get_num_samples(), int(args.test_duration_s * fs))
                recording = recording.frame_slice(start_frame=0, end_frame=end)
            logging.info(f"[standalone] denoising {name}: {recording}")
            t0 = time.perf_counter()
            denoised = deepinterpolate(
                recording, checkpoint_path, device=device,
                batch_size=args.batch_size, norm_sample_seconds=args.norm_sample_seconds,
            )
            denoised.save(folder=results_folder / f"{name}_denoised",
                          format="binary", chunk_duration=args.chunk_duration)
            logging.info(f"[standalone] wrote {name}_denoised (binary) in "
                         f"{round(time.perf_counter() - t0, 2)}s")

    logging.info(f"DEEPINTERPOLATION time: {round(time.perf_counter() - t_all, 2)}s")
