#!/bin/bash
set -e

DATA_DIR="/app/dataset/data"
DATA_GEN_CONFIG="/app/config/data-gen.yaml"

if ! ls "$DATA_DIR"/pointcloud-*.ply >/dev/null 2>&1; then
    echo "=== Dataset not found, generating ==="
    mkdir -p "$DATA_DIR"
    sed -i "s|^save_path:.*|save_path: \"$DATA_DIR/\"|" "$DATA_GEN_CONFIG"
    cd /app && /app/dataset_generator
    echo "=== Dataset generation complete ==="
else
    echo "=== Dataset cached, skipping generation ==="
fi

exec uv run yopo "$@"
