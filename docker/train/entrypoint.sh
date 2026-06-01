#!/bin/bash
set -e

DATA_DIR="/app/dataset/data"

if ! ls "$DATA_DIR"/pointcloud-*.ply >/dev/null 2>&1; then
    echo "ERROR: Dataset not found at $DATA_DIR"
    echo "Run 'yopo data-gen' first or mount a pre-generated dataset volume."
    exit 1
fi

echo "=== Dataset cached, proceeding to training ==="
exec uv run yopo "$@"
