# === Output Artifact Schema ==================================================
DATA_DIR     ?= $(CURDIR)/dataset

# === Build Configuration ======================================================
CUDA_VER     ?= 12.4.1
UBUNTU_VER   ?= 22.04
CUDA_ARCH    ?= 86
APT_MIRROR   ?= http://mirrors.tuna.tsinghua.edu.cn/ubuntu
IMAGE        ?= yopo-data-gen
TAG          ?= latest

# === Source Paths =============================================================
SIM_SRC      ?= $(CURDIR)/Simulator/src
CONFIG_SRC   ?= $(SIM_SRC)/config

# =============================================================================

.PHONY: image data clean shell help

image:  ## Build the data generation Docker image
	docker build --build-arg CUDA_VERSION=$(CUDA_VER) --build-arg UBUNTU_VERSION=$(UBUNTU_VER) --build-arg CUDA_ARCH=$(CUDA_ARCH) --build-arg APT_MIRROR=$(APT_MIRROR) -f docker/data_generation.Dockerfile -t $(IMAGE):$(TAG) .

data: image  ## Generate dataset (requires NVIDIA GPU)
	mkdir -p $(DATA_DIR)
	docker run --gpus all -v $(DATA_DIR):/dataset $(IMAGE):$(TAG)

clean:  ## Remove all generated datasets
	rm -rf $(DATA_DIR)

shell:  ## Enter container for debugging
	docker run --gpus all -it -v $(DATA_DIR):/dataset $(IMAGE):$(TAG) bash

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS=":.*## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'
