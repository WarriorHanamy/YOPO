ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG CUDA_ARCH=86
ARG APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG CONFIG_FILE=/app/config/config.yaml

# ===== Stage 1: Build =====
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

RUN rm -f /etc/apt/sources.list.d/ubuntu.sources && . /etc/os-release && cat > /etc/apt/sources.list <<EOF
deb $APT_MIRROR $VERSION_CODENAME main restricted universe multiverse
deb $APT_MIRROR $VERSION_CODENAME-updates main restricted universe multiverse
deb $APT_MIRROR $VERSION_CODENAME-security main restricted universe multiverse
deb $APT_MIRROR $VERSION_CODENAME-backports main restricted universe multiverse
EOF

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libpcl-dev libopencv-dev libeigen3-dev libyaml-cpp-dev libomp-dev && rm -rf /var/lib/apt/lists/*

COPY Simulator/src/ /src/
COPY docker/CMakeLists_standalone.txt /src/CMakeLists.txt

RUN cmake -B /build -S /src -DCMAKE_BUILD_TYPE=Release -DCONFIG_FILE=${CONFIG_FILE} -DCUDA_ARCH=${CUDA_ARCH} && cmake --build /build --target dataset_generator -j$(nproc)

# ===== Stage 2: Runtime =====
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN rm -f /etc/apt/sources.list.d/ubuntu.sources && . /etc/os-release && cat > /etc/apt/sources.list <<EOF
deb $APT_MIRROR $VERSION_CODENAME main restricted universe multiverse
deb $APT_MIRROR $VERSION_CODENAME-updates main restricted universe multiverse
deb $APT_MIRROR $VERSION_CODENAME-security main restricted universe multiverse
deb $APT_MIRROR $VERSION_CODENAME-backports main restricted universe multiverse
EOF

RUN apt-get update && apt-get install -y --no-install-recommends libpcl-dev libopencv-dev libyaml-cpp-dev libomp-dev && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dataset_generator /app/
COPY --from=builder /src/config/        /app/config/
COPY --from=builder /src/pointcloud/    /app/src/pointcloud/

WORKDIR /app
VOLUME ["/dataset"]
ENTRYPOINT ["./dataset_generator"]
