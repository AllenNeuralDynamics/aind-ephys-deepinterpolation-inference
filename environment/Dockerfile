# hash:sha256:e2822ede7736101a6bc7c49e9e40147f9b983a918748bdda9ab876f4dc9991b2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.1.7_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    s3fs==2026.6.0 \
    spikeinterface==0.104.8
