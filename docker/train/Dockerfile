ARG CUDA_ARCH=86
ARG CONFIG_FILE=/app/config/data-gen.yaml

# ===== Stage 1: Build dataset_generator =====
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS data-gen-builder
ARG CUDA_ARCH
ARG CONFIG_FILE

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake libpcl-dev libopencv-dev libeigen3-dev libyaml-cpp-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker/data-gen/CMakeLists.txt /src/
COPY docker/data-gen/src/ /src/src/
COPY docker/data-gen/include/ /src/include/

RUN cmake -B /build -S /src -DCMAKE_BUILD_TYPE=Release \
    -DCONFIG_FILE=${CONFIG_FILE} -DCUDA_ARCH=${CUDA_ARCH} \
    && cmake --build /build --target dataset_generator -j$(nproc)

# ===== Stage 2: Training environment =====
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS yopo-train
ARG CONFIG_FILE

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

COPY --from=data-gen-builder /build/dataset_generator /app/dataset_generator
COPY docker/data-gen/config/ /app/config/
COPY docker/data-gen/pointcloud/ /app/src/pointcloud/
COPY docker/train/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

VOLUME ["/app/dataset/data"]
VOLUME ["/app/YOPO/saved"]

ENTRYPOINT ["/app/entrypoint.sh"]
