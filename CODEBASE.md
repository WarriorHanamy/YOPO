# YOPO — Codebase Documentation

> Auto-generated repository documentation
>
> YOPO (You Only Plan Once) — learning-based one-stage quadrotor motion planner
> TJU-Aerial-Robotics, Tianjin University · MIT License

---

## Overview

YOPO replaces the traditional three-stage autonomous drone planning pipeline (perception → path search → trajectory optimization) with a single convolutional neural network. The network takes a depth image and odometry state, then directly outputs a collision-free, smooth flight trajectory — analogous to how YOLO unified object detection.

**Key innovation:** Instead of imitation learning or reinforcement learning, YOPO backpropagates gradients of analytical trajectory costs (smoothness, safety, guidance) directly into network weights. At inference, the network simultaneously predicts 15 candidate trajectories and their quality scores, selecting the best one without any iterative optimization — hence "You Only Plan Once."

**Repository type:** ROS-based polyglot research codebase (Python/PyTorch + C++/ROS + CUDA), not a production system.

---

## Repository Structure

### Architecture Pattern

**ROS polyglot composition** — three independently built subsystems communicate via ROS topics at runtime. Not a monorepo (no shared build), not microservices (tightly coupled via ROS), not a pure monolith (separate build systems).

### Top-Level Directory Map

| Directory              | Purpose                                                        | Language / Build                       |
| ---------------------- | -------------------------------------------------------------- | -------------------------------------- |
| `YOPO/`                | Neural planner: network, training, loss, inference, TensorRT   | Python/PyTorch (`pip`)                 |
| `Controller/`          | Quadrotor dynamics simulator + SO(3) attitude/position control | C++11 + ROS (`catkin_make`)            |
| `Simulator/`           | CUDA-accelerated depth/LiDAR sensor simulator + random maps    | C++17 + CUDA + ROS (`catkin_make`)     |
| `docker/data-gen/`     | Standalone dataset generator (no ROS)                          | C++17 + CUDA + Docker + CMake          |
| `dataset/`             | Generated training data — gitignored                           | Data artifact                          |
| `docs/`                | README images/GIFs (14 media files)                            | —                                      |
| `hardware/`            | Drone hardware design files (SolidWorks) — referenced, absent  | —                                      |

### Key Entry Points

| File                                                 | Role                                   |
| ---------------------------------------------------- | -------------------------------------- |
| `YOPO/train_yopo.py`                                 | Training entry (`--trial`, `--epoch`)  |
| `YOPO/test_yopo_ros.py`                              | ROS inference node (`--use_tensorrt`)  |
| `YOPO/yopo_trt_transfer.py`                          | PyTorch → TensorRT conversion          |
| `Controller/src/so3_control/src/network_control_node.cpp` | SO(3) controller ROS node              |
| `Controller/src/so3_quadrotor_simulator/src/dynamics/Quadrotor.cpp` | Rigid-body dynamics model  |
| `Simulator/src/src/sensor_simulator.cu`              | CUDA depth/LiDAR raycasting kernel     |
| `docker/data-gen/src/dataset_generator.cpp`          | Offline batch data collector           |

### Python Package Structure (`YOPO/`)

```
YOPO/
├── train_yopo.py              # Training entry
├── test_yopo_ros.py           # ROS inference node
├── yopo_trt_transfer.py       # TensorRT conversion
├── requirements.txt
├── config/
│   ├── config.py              # Global Config singleton (loads YAML)
│   └── traj_opt.yaml          # All hyperparameters
├── control_msg/
│   └── _PositionCommand.py    # ROS PositionCommand Python stub
├── loss/
│   ├── loss_function.py       # YOPOLoss: composite cost + denormalization
│   ├── smoothness_loss.py     # Jerk/acceleration smoothness cost
│   ├── safety_loss.py         # ESDF-based collision cost
│   └── guidance_loss.py       # Goal-direction guidance cost
├── policy/
│   ├── yopo_network.py        # YopoNetwork: backbone + head composition
│   ├── yopo_trainer.py        # YopoTrainer: gradient-based self-supervised loop
│   ├── yopo_dataset.py        # YOPODataset: data loading + augmentation
│   ├── state_transform.py     # Body ↔ Primitive frame transforms
│   ├── primitive.py           # LatticePrimitive: anchor grid (15 directions)
│   ├── poly_solver.py         # Poly5Solver: 5th-order polynomial trajectory
│   └── models/
│       ├── backbone.py        # ResNet-18/14 depth feature extractor
│       ├── head.py            # YopoHead: 3-layer 1×1 conv
│       └── resnet.py          # Full ResNet implementation
└── saved/
    └── YOPO_1/epoch50.pth     # Pretrained checkpoint
```

### File Naming Conventions

| Convention                  | Examples                                                    |
| --------------------------- | ----------------------------------------------------------- |
| Python: `snake_case.py`     | `train_yopo.py`, `state_transform.py`, `smoothness_loss.py` |
| C++: `PascalCase.cpp/.h`    | `Quadrotor.cpp`, `SO3Control.cpp`, `SensorSimulator.cu`     |
| CUDA: `snake_case.cu/.cuh`  | `sensor_simulator.cu`, `sensor_simulator.cuh`               |
| ROS msgs: `_PascalCase.py`  | `_PositionCommand.py`, `_SO3Command.py`                     |
| Config: `snake_case.yaml`   | `traj_opt.yaml`, `config.yaml`, `gains_pelican.yaml`        |
| Launch: `snake_case.launch` | `simulator_attitude_control.launch`                         |

---

## Getting Started

### Prerequisites

- Ubuntu 20.04 (or Jetson Orin/Xavier NX)
- CUDA toolkit + NVIDIA GPU
- ROS Noetic
- Conda

### Installation

```bash
git clone --depth 1 git@github.com:TJU-Aerial-Robotics/YOPO.git
cd YOPO

# Python environment
conda create --name yopo python=3.8
conda activate yopo
pip install -r YOPO/requirements.txt

# Build Controller (C++/ROS)
cd Controller && catkin_make

# Build Simulator (C++/CUDA/ROS)
cd Simulator && catkin_make

# (Optional) TensorRT for edge deployment
pip install nvidia-tensorrt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt && cd torch2trt && python setup.py install

# (Optional) Dataset generation via Docker
make image  # build data-gen Docker image
make data   # generate dataset (requires --gpus all)
```

### Runtime Quick Start

1. Launch simulator + controller:
   ```bash
   roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
   ```
2. Launch YOPO planner:
   ```bash
   python test_yopo_ros.py --trial=1 --epoch=50
   ```
3. Click "2D Nav Goal" in RViz to set the navigation target.

### Training

```bash
python train_yopo.py                    # from scratch
python train_yopo.py --pretrained=1 \   # resume
    --trial=1 --epoch=50
```

### Data Generation (Docker)

```bash
make image CUDA_ARCH=86 CUDA_VER=12.4.1
make data DATA_DIR=./dataset
```

---

## Architecture

### System Topology (3-node ROS Graph)

```
┌──────────────────────┐    /depth_image     ┌──────────────────────┐
│  sensor_simulator   │ ──────────────────→  │   yopo_net (Python)  │
│  (C++/CUDA)         │                      │   (PyTorch/TensorRT) │
│                      │    /sim/odom        │                      │
│  Pub: /depth_image  │ ←────────────────── │  Pub: /pos_cmd       │──┐
│       /lidar_points  │                      │       /trajs_visual  │  │
│       /mock_map      │                      │                      │  │
└──────────────────────┘                      └──────────────────────┘  │
                                                                        │
                                          ┌─────────────────────────────┘
                                          │  /so3_control/pos_cmd
                                          ▼
                               ┌──────────────────────┐
                               │  network_controller  │
                               │  (C++/ROS nodelet)   │
                               │                      │
                               │  Sub: /pos_cmd       │
                               │  Pub: so3_cmd        │──┐
                               └──────────────────────┘  │
                                                          │ so3_cmd
                                                          ▼
                               ┌──────────────────────┐
                               │  quadrotor_sim_so3   │
                               │  (C++ dynamics)      │
                               │                      │
                               │  Pub: /sim/odom       │
                               │       /sim/imu        │
                               └──────────────────────┘
```

### Data Flow (Inference Pipeline)

```
Depth Image [160×96]            Odometry [pos, vel, quat]
       │                              │
       ▼                              ▼
  ResNet-18 Backbone           normalize + rotate to
  → 64-channel features          15 primitive frames
       │                              │
       └──────── CONCAT ──────────────┘
                    │
                    ▼
             YopoHead (3× Conv1×1)
                    │
         ┌─────────┴─────────┐
         ▼                   ▼
  15 endstates (tanh)   15 scores (softplus)
         │                   │
         │    argmin(score)  │
         │         │         │
         └─────────┘         │
                    │        │
                    ▼        │
             Best primitive  │
                    │        │
                    ▼        │
        Poly5Solver(start, end, T)
                    │
                    ▼
        PositionCommand @ 50 Hz
```

### Frame Convention

All spatial reasoning resolves to **NWU** (North-West-Up) world frame.

| Frame        | Origin      | Axes                          | Purpose                         |
| ------------ | ----------- | ----------------------------- | ------------------------------- |
| World (W)    | Inertial    | NWU                           | Absolute position, ESDF map     |
| Body (B)     | Drone CoM   | X=forward, Y=left, Z=up       | Odometry input, control output  |
| Primitive (P) | Body origin | Z points toward anchor        | Network predicts offsets here   |
| Camera (C)   | Camera sensor | Z=forward, X=right, Y=down  | Ray casting, depth image        |

---

## Data Layer

YOPO has **no traditional database** — no SQL/NoSQL, no ORM/ODM, no migration system. Data is file-based.

### Dataset Format

```
dataset/
├── 0/                  # Map_id=0 image directory
│   ├── img_0.png       # 16-bit PNG depth (160×90)
│   ├── img_1.png
│   └── ...
├── 1/                  # Map_id=1
├── pose-0.csv          # Pose labels: px,py,pz,qw,qx,qy,qz per row
├── pose-1.csv
├── pointcloud-0.ply    # Point cloud map for ESDF construction
└── pointcloud-1.ply
```

- **Depth images:** 16-bit PNG, `[0, 65535]` encodes `[0, 20m]` range.
- **Pose CSV:** `px, py, pz, qw, qx, qy, qz` per row, index-aligned with images.
- **Point clouds:** Binary PLY, converted to ESDF at training start.

### ESDF Map System

The closest component to a "database" — a 3D voxel grid of signed distances to nearest obstacle:
- Built from `.ply` point clouds via `scipy.ndimage.distance_transform_edt`
- Voxel resolution: 0.2 m
- Stored as GPU tensor `[N, 1, D, H, W]`
- Queried at 30 sample points per trajectory via `F.grid_sample` (bilinear interpolation)
- Local batch cropping avoids memory blowup when training across multiple maps

### Model Checkpoints

- Format: PyTorch `.pth` (state_dict)
- Path: `YOPO/saved/YOPO_{trial}/epoch{epoch}.pth`
- Saved every epoch, with `atexit` crash-safe handler
- TensorBoard logs in same directory
- TensorRT path: converted via `torch2trt` to `yopo_trt.pth` (FP16, ~1 ms on Orin NX)

### Data Augmentation (Online)

Performed in `YOPODataset.__getitem__()`:
- Velocity: log-normal Vx, normal Vy/Vz
- Acceleration: normal, clipped to `acc_max`
- Goal: uniformly sampled yaw/pitch within camera FOV; 10% probability nearby goal
- 90/10 train/validation split per map folder

---

## Core Logic

### Motion Primitive Lattice

The planner does not regress trajectories directly. It defines **15 fixed anchor directions** (5 horizontal × 3 vertical × 1 radial) that cover the drivable space, analogous to YOLO anchor boxes:

| Parameter            | Value                |
| -------------------- | -------------------- |
| Horizontal anchors   | 5 (across 90° FOV)   |
| Vertical anchors     | 3 (across 60° FOV)   |
| Radial levels        | 1 (at 5.0 m range)   |
| Total primitives     | **15**               |
| Delta offset range   | ±15° yaw, ±15° pitch |

### YopoNetwork Architecture

```
Input: depth [B, 1, 96, 160]
       obs   [B, 9, 3, 5]  (vel+acc+goal in primitive frames)

ResNet-18 backbone (modified: single-channel conv1, output 64ch@3×5)
       +
Observation (9 channels, identity, no state backbone)
       ↓
Concat [B, 73, 3, 5]
       ↓
YopoHead: Conv1×1(73→256) → Conv1×1(256→256) → Conv1×1(256→10)
       ↓
┌─────────────────────┐
│ endstate [B, 9, 3, 5]  tanh → normalized delta (pos+vel+acc)
│ score    [B, 1, 3, 5]  softplus → scalar cost per primitive
└─────────────────────┘
```

- `ResNet-14` variant removes layer4 for faster embedded inference.

### State Transform Pipeline

1. **Normalize:** velocity by `vel_max` (6 m/s), acceleration by `acc_max` (6 m/s²)
2. **Body → Primitive frame:** rotate observation through `Rbp` (3×3 per primitive)
3. **Network forward**
4. **Primitive → Body frame (decode):**
   - Position: `delta_yaw` → spherical → Cartesian → rotate by `Rbp`
   - Velocity/Acceleration: denormalize + rotate by `Rbp`

### Trajectory Parameterization (Poly5Solver)

Each axis is a **5th-order polynomial** with 6 boundary conditions:
```
p(t) = A₀ + A₁t + A₂t² + A₃t³ + A₄t⁴ + A₅t⁵
```
Boundary: `pos₀,vel₀,acc₀` (odometry) → `pos₁,vel₁,acc₁` (network prediction).
Coefficient matrix `Coef_inv` (6×6) is precomputed for closed-form solution.

### Training: Gradient-Based Self-Supervision

**No expert demonstrations, no environment interaction, no RL.**

1. Forward pass → 15 endstate predictions + 15 scores
2. Expand batch: each sample → 15 trajectories (one per primitive)
3. Transform to world frame
4. Compute **analytical costs** per trajectory:
   - **Smoothness:** ∫jerk² dt (weight 10.0, scaled by v⁵)
   - **Acceleration:** ∫acc² dt (weight 1.0, scaled by v³)
   - **Safety:** ∫exp(-(d-1.2)/0.6) dt via ESDF query
   - **Guidance:** L1 projection onto goal direction (weight 0.15)
5. **Score loss:** `smooth_l1_loss(score_pred, cost_detach)` — network learns to predict its own cost
6. **Trajectory loss:** mean of weighted costs across all primitives
7. Backward → AdamW optimizer

### Yaw Calculation

Blended: `yaw = atan2(vel_dir + weight × goal_dir)`, weight proportional to yaw error. Limited by `max_yaw_rate = 0.5 rad/s`.

---

## Interface Layer (APIs & Routes)

**No web API, no GraphQL, no WebSocket.** Communication is entirely via ROS topics/services.

### ROS Topics

| Topic                            | Type                              | Direction             | Rate    | Purpose                        |
| -------------------------------- | --------------------------------- | --------------------- | ------- | ------------------------------ |
| `/sim/odom`                      | `nav_msgs/Odometry`               | sim → planner, ctrl   | 100 Hz  | Drone state                    |
| `/sim/imu`                       | `sensor_msgs/Imu`                 | sim → controller      | 100 Hz  | IMU data                       |
| `/depth_image`                   | `sensor_msgs/Image` (32FC1)       | sim → planner         | 33 Hz   | Depth frame                    |
| `/lidar_points`                  | `sensor_msgs/PointCloud2`         | sim                   | 10 Hz   | LiDAR output                   |
| `/mock_map`                      | `sensor_msgs/PointCloud2`         | sim → RViz            | 1 Hz    | Map visualization              |
| `/so3_control/pos_cmd`           | `quadrotor_msgs/PositionCommand`  | planner → controller  | 50 Hz   | Position/velocity/acc setpoint |
| `so3_cmd`                        | `quadrotor_msgs/SO3Command`       | controller → sim      | 50 Hz   | Force + attitude target        |
| `/move_base_simple/goal`         | `geometry_msgs/PoseStamped`       | RViz → planner        | on-click | User navigation goal           |
| `/force_disturbance`             | `geometry_msgs/Vector3`           | external → sim        | on-demand | Wind disturbance test          |
| `/moment_disturbance`            | `geometry_msgs/Vector3`           | external → sim        | on-demand | Moment disturbance             |
| `/yopo_net/trajs_visual`         | `sensor_msgs/PointCloud2`         | planner → RViz        | 33 Hz   | All 15 trajectories (colored)  |
| `/yopo_net/best_traj_visual`     | `sensor_msgs/PointCloud2`         | planner → RViz        | 33 Hz   | Selected trajectory            |
| `/yopo_net/lattice_trajs_visual` | `sensor_msgs/PointCloud2`         | planner → RViz        | 33 Hz   | Anchor lattice visualization   |

### ROS Services

| Service                                   | Provider           | Purpose                         |
| ----------------------------------------- | ------------------ | ------------------------------- |
| `/network_controller_node/takeoff_land`   | network_controller | Arm/takeoff or land/disarm      |

### CLI Entry Points

| Script                    | Arguments                                          |
| ------------------------- | -------------------------------------------------- |
| `YOPO/train_yopo.py`      | `--pretrained`, `--trial`, `--epoch`               |
| `YOPO/test_yopo_ros.py`   | `--trial`, `--epoch`, `--use_tensorrt`, `--verbose`, `--visualize` |
| `YOPO/yopo_trt_transfer.py` | `--trial`, `--epoch`, `--dir`                    |

### Authentication

**None.** Local ROS network only, no auth/encryption.

---

## Testing

### Status: Research-Grade (Minimal)

YOPO has **near-zero automated test coverage**. This is typical for academic robotics code — validation is performed empirically through training curves (TensorBoard), simulation visualization (RViz), and physical flight tests.

### Existing Tests

| File                                                          | Framework    | Content                                      |
| ------------------------------------------------------------- | ------------ | -------------------------------------------- |
| `Controller/src/utils/uav_utils/src/uav_utils_test.cpp`       | Google Test  | Geometry utils: rotation, skew, angle round-trips (7 test cases) |
| `Controller/.../ode/libs/numeric/odeint/test/*.cpp`           | Boost.Test   | Vendored ODE solver tests (third-party)      |
| `Controller/.../so3_quadrotor_simulator/src/test_dynamics.cpp`| Manual smoke | PD controller behavior benchmark (no assertions) |
| `Simulator/src/src/test_simulator.cpp`                        | Manual smoke | CPU simulator smoke test                     |
| `Simulator/src/src/test_simulator_cuda.cpp`                   | Manual smoke | CUDA simulator smoke test                    |

### What's Missing

- No `pytest`/`unittest` tests for the Python planner (core innovation)
- No unit tests for loss functions, state transforms, or polynomial solver
- No integration tests for ROS nodes
- No mocking framework (no `unittest.mock`, no `pytest-mock`, no Google Mock)
- No CI pipeline (no `.github/workflows/`, no `.gitlab-ci.yml`)
- No code coverage tooling (no `gcov`, no `coverage.py`, no `.coveragerc`)
- `catkin_add_nosetests(test)` is commented out in `uav_utils/CMakeLists.txt`

---

## Deployment

### CI/CD: None

No automated pipelines. Build, test, and deployment are entirely manual.

### Docker

**Single image** for offline dataset generation only (not for the planner or ROS stack):

| Artifact                     | Location                     | Purpose                            |
| ---------------------------- | ---------------------------- | ---------------------------------- |
| Multi-stage Dockerfile       | `docker/data-gen/Dockerfile` | Builder + runtime, CUDA 12.4.1     |
| Entrypoint                   | `docker/data-gen/entrypoint.sh` | Rewrites save_path → runs generator |
| Mirror setup                 | `docker/data-gen/setup-apt-mirror.sh` | APT mirror for China |

No `docker-compose.yml`.

### Makefile (Operational Interface)

| Target   | Description                      |
| -------- | -------------------------------- |
| `image`  | Build data-gen Docker image      |
| `data`   | Generate dataset (needs GPU)     |
| `clean`  | Remove generated datasets        |
| `shell`  | Debug container shell            |
| `help`   | Print all targets                |

### Edge Deployment

- **TensorRT (FP16):** ~1 ms inference on NVIDIA Orin NX
- **RKNN (INT8):** ~20 ms on RK3566 NPU (1 TOPS)
- **PyTorch native:** <5 ms (ResNet-18), 1-2 ms (ResNet-14)

### Monitoring

- TensorBoard for training loss curves
- ROS log files with `log_plot.py` post-processing
- No structured logging, no metrics infrastructure
- No Kubernetes, no Terraform, no cloud configs

---

## Dependencies

### Python (YOPO/)

**File:** `YOPO/requirements.txt`

| Package            | Version      | Purpose                                                  |
| ------------------ | ------------ | -------------------------------------------------------- |
| `torch`            | 2.4.1+cu118  | Neural network + autograd                                |
| `torchvision`      | 0.19.1+cu118 | ResNet definitions                                       |
| `opencv-python`    | 4.11.0       | Depth image resize, NaN inpainting                       |
| `scipy`            | 1.10.1       | Rotation transforms, `distance_transform_edt`            |
| `ruamel-yaml`      | 0.17.21      | YAML config loading                                      |
| `numpy`            | 1.22.3       | Numerical operations                                     |
| `tensorboard`      | 2.14.0       | Training visualization                                   |
| `open3d`           | 0.19.0       | Point cloud I/O for ESDF maps                            |
| `rich`             | 14.0.0       | CLI progress bars                                        |

Extra index: `https://download.pytorch.org/whl/cu118`

**Optional (not in requirements.txt):**
- `torch2trt` (from source) — TensorRT conversion
- `nvidia-tensorrt` — TensorRT runtime

### C++/ROS (Controller)

- **Build:** `catkin_make`, C++11
- **ROS packages:** `roscpp`, `nav_msgs`, `geometry_msgs`, `sensor_msgs`, `tf`, `tf2_ros`, `nodelet`, `cv_bridge`
- **Libraries:** Eigen3, Armadillo
- **Vendored:** Boost odeint (~200 header files, in-tree)

### C++/CUDA (Simulator)

- **Build:** `catkin_make`, C++17 + CUDA
- **Libraries:** Eigen3, PCL, OpenCV, yaml-cpp, OpenMP

### Docker Data Generator

- **Base:** `nvidia/cuda:12.4.1-devel-ubuntu22.04` (build), `-runtime-` (runtime)
- **Libraries:** CMake, PCL, OpenCV, Eigen3, yaml-cpp, OpenMP

### No Lock Files

No `poetry.lock`, `Pipfile.lock`, `yarn.lock`, or `package-lock.json`. Dependencies are pinned inline in `requirements.txt`.

### CUDA Version Inconsistency

PyTorch targets **CUDA 11.8** (`cu118`), while Docker uses **CUDA 12.4.1**. Both work with the same driver via forward compatibility — but this is fragile.

---

## Domain Glossary

### Core Concepts

| Term               | Definition                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| **YOPO**           | You Only Plan Once — one-stage motion planner                                                           |
| **Motion Primitive** | One of 15 candidate 5th-order polynomial trajectories                                                  |
| **Primitive Anchor** | Fixed spherical direction forming a 5×3×1 angular lattice                                              |
| **Delta Yaw/Pitch**  | Per-primitive offset (±15°) predicted by network, added to anchor direction                            |
| **Poly5Solver**       | Closed-form 5th-order polynomial coefficient solver from boundary conditions                           |
| **ESDF**              | Euclidean Signed Distance Field — 3D voxel grid of distance to nearest obstacle                       |
| **NWU**               | North-West-Up inertial coordinate frame                                                                |
| **Rbp**               | Rotation matrix from Primitive frame to Body frame                                                     |
| **PositionCommand**   | ROS message: position + velocity + acceleration + yaw + yaw_dot + trajectory status                   |
| **Score**             | Network-predicted scalar cost per primitive; `argmin(score)` selects best trajectory at inference      |

### Loss Functions

| Loss             | Formula / Description                                                 | Weight |
| ---------------- | --------------------------------------------------------------------- | ------ |
| **Smoothness**   | ∫jerk² dt over trajectory, denormalized by `vel_scale⁵`               | 10.0   |
| **Acceleration** | ∫acc² dt, denormalized by `vel_scale³`                                | 1.0    |
| **Safety**       | `exp(-(d - d₀) / r)` with d₀=1.2m, r=0.6m, sampled at 30 points     | 1.0    |
| **Guidance**     | L1 of trajectory projection onto goal direction + 0.5× perpendicular | 0.15   |

### Key Constants

| Constant             | Value       | Meaning                               |
| -------------------- | ----------- | ------------------------------------- |
| `vel_max_train`      | 6.0 m/s     | Max training velocity                 |
| `acc_max_train`      | 6.0 m/s²    | Max training acceleration             |
| `radio_range`        | 5.0 m       | Primitive radial reach                |
| `segment_time`       | ~1.667 s    | 2×radio_range/vel_max                 |
| `d₀` (safety margin) | 1.2 m       | Safe obstacle clearance               |
| `r` (decay)          | 0.6 m       | Safety cost sharpness                 |
| `voxel_size`         | 0.2 m       | ESDF resolution                       |
| `control_dt`         | 0.02 s      | 50 Hz control loop                    |
| `depth_resolution`   | 160×96      | Network input size                    |
| `horizon_num`        | 5           | Primitive grid columns                |
| `vertical_num`       | 3           | Primitive grid rows                   |
| `radio_num`          | 1           | Primitive grid radial levels          |
| `traj_num`           | 15          | Total primitives                      |
| `observation_dim`    | 9           | vel(3)+acc(3)+goal(3) in body frame  |
| `output_dim`         | 10          | pos(3)+vel(3)+acc(3)+score per primitive |

### Frame Suffix Conventions

| Suffix | Meaning                                   | Example          |
| ------ | ----------------------------------------- | ---------------- |
| `_b`   | Body frame                                | `vel_b`, `goal_b`  |
| `_w`   | World frame (NWU)                         | `pos_w`, `vel_w`   |
| `_p`   | Primitive frame                           | `end_vel_p`        |
| `_c`   | Camera frame                              | `vel_c`            |
| `Rbp`  | Rotation Primitive→Body                   | `lattice_Rbp_list` |

---

## Documentation Index

### Existing Documentation

| File                                           | Lines | Quality     | Content                                    |
| ---------------------------------------------- | ----- | ----------- | ------------------------------------------ |
| `README.md`                                    | 228   | Good        | Project overview, install, test steps      |
| `PrimitivesAnalysis.md`                        | 541   | Excellent   | Deep technical analysis of primitive system |
| `Simulator/src/readme.md`                      | 109   | Good        | Simulator build/run/config reference        |
| `Controller/src/readme.md`                     | 35    | Adequate    | Controller build + rosservice commands     |
| `YOPO/readme.md`                               | 5     | Poor        | Placeholder only                            |
| `YOPO/config/traj_opt.yaml`                    | 52    | Good        | Well-commented parameter reference          |
| `Simulator/src/config/config.yaml`             | 87    | Good        | Well-commented simulator parameters         |
| `docker/data-gen/config/config.yaml`           | 87    | Good        | Mirror of simulator config                  |
| `Controller/src/so3_control/config/gains*.yaml` | 3 files | Adequate   | PID/attitude control gains                 |

### Documentation Gaps

- **No API reference** — no Sphinx or auto-generated docs
- **No ADRs** — no architecture decision records
- **No contributing guide** — no `CONTRIBUTING.md`
- **No changelog** — no version history
- **No C++ Doxygen** — Controller and Simulator have no annotated headers
- **No testing documentation** — no test running instructions
- **No security policy** — no `SECURITY.md`
- **Missing hardware files** — `hardware/` directory and `hardware_list.pdf` referenced but absent

---

## License

MIT License, 2024 — TJU-Aerial-Robotics, Tianjin University

**Papers:**
- YOPO: You Only Plan Once (RA-L 2024)
- YOPOv2-Tracker: End-to-end agile tracking from perception to action

BibTeX available in `README.md`.
