# MetaX MACA dev image aligned with infinilm-prefill-dev (vLLM + MACA stack).
# Digest sha256:6e519687a9e4cd71036d20bbe0660435fe13636692dded392f2d06936676a44e
FROM cr.metax-tech.com/public-ai-release/maca/vllm-metax@sha256:6e519687a9e4cd71036d20bbe0660435fe13636692dded392f2d06936676a44e

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    make \
    g++ \
    libdw-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
