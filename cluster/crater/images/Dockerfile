# CUDA devel environment only. Mount the project checkout and run build_nccl.sh.
# NCCL itself always comes from the checked-out, verified project submodule.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       g++-11 gcc-11 libc6-dev make git python3 binutils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda CXX=g++-11 JOBS=2
WORKDIR /workspace/mycroft
CMD ["bash"]
