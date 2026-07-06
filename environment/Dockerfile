# GPU container for the DeepInterpolation inference capsule.
#
# Code Ocean requires base images from its own registry (ARG REGISTRY_HOST). We
# reuse the same CUDA base as the Kilosort4 spike-sorting capsule (it already ships
# a working CUDA + PyTorch stack), then add SpikeInterface (matching the pipeline's
# version) + zarr / WavPack so the capsule can read compressed AIND recordings and
# write the denoised Zarr.
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.1.7_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive

# SpikeInterface core (does NOT pull torch, preserving the base CUDA torch),
# plus zarr + WavPack. Pinned to the pipeline's SpikeInterface version so the
# denoised Zarr is read cleanly by the downstream preprocessing / sorting steps.
RUN pip install -U --no-cache-dir \
        "spikeinterface==0.103.2" \
        wavpack-numcodecs \
        "zarr>=2.16,<3"
