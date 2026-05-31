# Agent Operating Rules — YOPO

## Workspace Layout

- `YOPO/`             — Python package (training + inference, PyTorch)
- `Controller/`       — ROS catkin workspace (C++ SO(3) controller)
- `Simulator/`        — ROS catkin workspace (C++/CUDA sensor simulator)
- `docker/data-gen/`  — Docker-based offline data generation (CUDA, no ROS)
- `dataset/`          — Training data mount point (gitignored, never commit)
- `docs/`             — Media assets (GIFs, PNGs) + architecture diagrams (PlantUML *.puml)
- `tests/`            — pytest scaffolded, tests not yet written

## Naming Conventions

- **Python classes**: `UpperCamelCase` (e.g., `YopoNetwork`, `YOPODataset`)
- **Python functions/variables**: `snake_case` (e.g., `configure_random_seed`)
- **Python private helpers**: `_leading_underscore`
- **C++ classes/structs**: `UpperCamelCase` (e.g., `SO3Control`, `Quadrotor`, `HGDO`)
- **YAML config keys**: `snake_case`

## Forbidden Assumptions

- **Python version**: ==3.12 (per `pyproject.toml` and `uv.lock`). Do NOT assume 3.8.
- **CUDA**: NOT guaranteed — always check `torch.cuda.is_available()` and fall back to CPU.
- **ROS**: Noetic required for Controller and Simulator workspaces.
- The `docker/data-gen/` container does NOT require ROS.

## Safety Rules (NEVER do these)

- Do NOT modify generated ROS messages in `Controller/src/utils/quadrotor_msgs/` or `Controller/src/utils/mavros_msgs/`.
- Do NOT commit files inside `dataset/` (gitignored mount point).
- Do NOT run ruff on `Simulator/` or `Controller/` — excluded from pre-commit.
- Do NOT modify `.github/LICENSE` (used by pre-commit for header insertion).

## Build & Run Constraints

- **Python**: Use `uv sync` for deps, `uv run yopo <cmd>` for execution.
- **ROS builds**: `catkin_make` in `Controller/` and `Simulator/` separately.
- **Data generation**: `yopo data-gen` (builds Docker image + runs with `--gpus all`).
- **Pre-commit**: Run `pre-commit install` after `uv sync`.

## Code Style (enforced by pre-commit)

- Ruff configured in `pyproject.toml` (line-length=99, indent-width=4, quote-style=single).
- License header required on all `.py` files.

## Agent Behavior Constraints

- Before running ROS nodes: source `devel/setup.bash` in the relevant workspace.
- Before catkin_make: source `/opt/ros/noetic/setup.bash`.
- TensorRT: requires manual `pip install nvidia-tensorrt` from NGC + `torch2trt` from GitHub.
- Conda is legacy; use `uv` for all Python environment management.

## Artifact Contracts

All I/O artifacts (dataset, checkpoints, Docker images, deploy configs) are documented in [docs/artifacts.md](docs/artifacts.md). Corresponding Pydantic schemas live in `YOPO/schema.py`. When adding or modifying pipeline artifacts, update both files.
