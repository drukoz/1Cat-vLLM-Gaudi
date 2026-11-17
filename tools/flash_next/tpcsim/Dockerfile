# SPDX-License-Identifier: Apache-2.0
#
# x86-64 Ubuntu 22.04 image with the Habana TPC toolchain (tpc-clang) and the
# Gaudi2 instruction simulator, for compiling and executing TPC kernels on
# Apple Silicon via Rosetta. See docs/design/flash-next/tpcsim.md.
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
        build-essential cmake libboost-dev \
        python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://vault.habana.ai/artifactory/api/gpg/key/public | gpg --dearmor -o /usr/share/keyrings/habana-artifactory.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/habana-artifactory.gpg] https://vault.habana.ai/artifactory/debian jammy main" > /etc/apt/sources.list.d/habanalabs.list \
    && apt-get update \
    && (apt-get install -y --no-install-recommends habanalabs-tools=1.24.1-482 \
        || (echo "apt install failed, falling back to the direct deb" \
            && curl -fsSL -o /tmp/habanalabs-tools.deb \
               https://vault.habana.ai/artifactory/debian/jammy/pool/main/h/habanalabs-tools/habanalabs-tools_1.24.1-482_amd64.deb \
            && apt-get install -y --no-install-recommends /tmp/habanalabs-tools.deb \
            && rm -f /tmp/habanalabs-tools.deb)) \
    && rm -rf /var/lib/apt/lists/*

# habanalabs-tools ships its .so files without RPATH/RUNPATH, so the
# simulator's transitive dependencies (e.g. libTpcElfReader.so) only resolve
# when /usr/lib/habanatools is in the system loader cache.
RUN echo "/usr/lib/habanatools" > /etc/ld.so.conf.d/habanatools.conf && ldconfig

RUN python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch pytest numpy \
    && python3 -c "import torch; print('torch', torch.__version__)"

WORKDIR /work
