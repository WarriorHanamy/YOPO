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
DATA_GEN_DIR ?= $(CURDIR)/docker/data-gen

# =============================================================================

.PHONY: image data clean shell help

image:  ## Build the data generation Docker image
	docker build --build-arg CUDA_VERSION=$(CUDA_VER) --build-arg UBUNTU_VERSION=$(UBUNTU_VER) --build-arg CUDA_ARCH=$(CUDA_ARCH) --build-arg APT_MIRROR=$(APT_MIRROR) -t $(IMAGE):$(TAG) $(DATA_GEN_DIR)

data: image  ## Generate dataset (requires NVIDIA GPU)
	mkdir -p $(DATA_DIR)
	docker run --gpus all -v $(DATA_DIR):/output $(IMAGE):$(TAG)

clean:  ## Remove all generated datasets
	docker run --rm -v $(DATA_DIR):/output alpine rm -rf /output/data 2>/dev/null || true
	rmdir $(DATA_DIR) 2>/dev/null || true

shell:  ## Enter container for debugging
	docker run --gpus all -it -v $(DATA_DIR):/output $(IMAGE):$(TAG) bash

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS=":.*## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'
