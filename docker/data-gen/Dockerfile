# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG CUDA_ARCH=86
ARG APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG CONFIG_FILE=/app/config/config.yaml
ARG DATASET_GENERATOR_TARGET=dataset_generator

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG CUDA_ARCH
ARG APT_MIRROR
ARG CONFIG_FILE
ARG DATASET_GENERATOR_TARGET
ENV DATASET_GENERATOR_TARGET=${DATASET_GENERATOR_TARGET}

RUN rm -f /etc/apt/sources.list.d/ubuntu.sources \
  && . /etc/os-release \
  && { \
    echo "deb ${APT_MIRROR} ${VERSION_CODENAME} main restricted universe multiverse"; \
    echo "deb ${APT_MIRROR} ${VERSION_CODENAME}-updates main restricted universe multiverse"; \
    echo "deb ${APT_MIRROR} ${VERSION_CODENAME}-security main restricted universe multiverse"; \
    echo "deb ${APT_MIRROR} ${VERSION_CODENAME}-backports main restricted universe multiverse"; \
  } > /etc/apt/sources.list

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libpcl-dev libopencv-dev libeigen3-dev libyaml-cpp-dev libomp-dev

# Build layout:
# /app/CMakeLists.txt
# /app/src/*
# /app/include/*
# /app/build/*
COPY CMakeLists.txt /app/
COPY src/           /app/src/
COPY include/       /app/include/

RUN cmake -B /app/build -S /app -DCMAKE_BUILD_TYPE=Release \
  -DCONFIG_FILE=${CONFIG_FILE} -DCUDA_ARCH=${CUDA_ARCH} \
  -DDATASET_GENERATOR_TARGET=${DATASET_GENERATOR_TARGET} \
  && cmake --build /app/build --target ${DATASET_GENERATOR_TARGET} -j$(nproc) \
  && cp /app/build/${DATASET_GENERATOR_TARGET} /app/ \
  && rm -rf /app/build

# Runtime layout:
# /app/${DATASET_GENERATOR_TARGET}
# /app/entrypoint.sh
# /app/config/config.yaml
# /app/src/pointcloud/*
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
COPY config/        /app/config/
RUN chmod 666 /app/config/config.yaml
COPY pointcloud/    /app/src/pointcloud/

WORKDIR /app
VOLUME ["/output"]
ENTRYPOINT ["/app/entrypoint.sh"]
