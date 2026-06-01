FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS yopo-trainer

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libpcl-dev libopencv-dev libyaml-cpp-dev libomp-dev \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY YOPO/ YOPO/

RUN uv sync

COPY docker/train/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

VOLUME ["/app/dataset/data"]
VOLUME ["/app/YOPO/saved"]

ENTRYPOINT ["/app/entrypoint.sh"]
