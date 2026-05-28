ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG CUDA_ARCH=86
ARG APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG CONFIG_FILE=/app/config/config.yaml

# ===== Stage 1: Build =====
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder
ARG APT_MIRROR
ARG CUDA_ARCH
ARG CONFIG_FILE

COPY setup-apt-mirror.sh /tmp/setup-apt-mirror.sh
RUN chmod +x /tmp/setup-apt-mirror.sh && /tmp/setup-apt-mirror.sh

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake libpcl-dev libopencv-dev libeigen3-dev libyaml-cpp-dev libomp-dev \
  && rm -rf /var/lib/apt/lists/*

COPY CMakeLists.txt /src/
COPY src/           /src/src/
COPY include/       /src/include/

RUN cmake -B /build -S /src -DCMAKE_BUILD_TYPE=Release \
  -DCONFIG_FILE=${CONFIG_FILE} -DCUDA_ARCH=${CUDA_ARCH} \
  && cmake --build /build --target dataset_generator -j$(nproc)

# ===== Stage 2: Runtime =====
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
ARG APT_MIRROR

COPY setup-apt-mirror.sh /tmp/setup-apt-mirror.sh
RUN chmod +x /tmp/setup-apt-mirror.sh && /tmp/setup-apt-mirror.sh

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libpcl-dev libopencv-dev libyaml-cpp-dev libomp-dev \
  && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
COPY --from=builder /build/dataset_generator /app/
COPY config/        /app/config/
RUN chmod 666 /app/config/config.yaml
COPY pointcloud/    /app/src/pointcloud/

WORKDIR /app
VOLUME ["/output"]
ENTRYPOINT ["/app/entrypoint.sh"]
