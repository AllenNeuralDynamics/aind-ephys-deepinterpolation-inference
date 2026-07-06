# GPU container for the DeepInterpolation inference capsule.
#
# The trained model runs on the GPU, so this image is CUDA-enabled (unlike the
# CPU `aind-ephys-pipeline-base` image used by compress/preprocessing). It matches
# the pattern of the Kilosort4 spike-sorting step, which also ships its own GPU
# image.
#
# Build & push, then register the resulting tag as the DEEPINTERP container in the
# hybrid-benchmark pipeline (parse_capsule_versions / nextflow config).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Match the SpikeInterface version the rest of the pipeline uses (capsule_versions.env),
# so the denoised Zarr recording this capsule writes is read cleanly by the
# downstream preprocessing / spike-sorting steps.
ARG SPIKEINTERFACE_VERSION=0.103.2

# SpikeInterface core (does NOT pull torch, preserving the CUDA torch above),
# plus zarr + WavPack so the capsule can read compressed AIND recordings.
RUN pip install --no-cache-dir \
        "spikeinterface==${SPIKEINTERFACE_VERSION}" \
        wavpack-numcodecs \
        "zarr>=2.16,<3"
