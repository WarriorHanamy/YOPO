#!/bin/sh
set -e
mkdir -p /output/data
sed -i "s|save_path:.*|save_path: \"/output/data/\"|" /app/config/config.yaml
exec "/app/${DATASET_GENERATOR_TARGET}"
