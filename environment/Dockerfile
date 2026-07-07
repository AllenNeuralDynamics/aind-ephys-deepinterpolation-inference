# hash:sha256:a51fededc75c1b1d92b8281378b933933865499824a1c5dfcb9344fcac2a616a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.1.7_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    spikeinterface==0.104.8
