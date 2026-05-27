# YOPO Primitives Analysis

## 1. System Overview

YOPO is a neural-network-based motion planner for aggressive quadrotor flight. It generates 15 candidate 5th-order polynomial trajectories ("primitives") from a single depth image. Each primitive is parameterized by start and end boundary conditions `(pos, vel, acc)`, where start conditions come from the current vehicle state and end conditions are predicted by a CNN conditioned on depth and a predefined angular lattice. The lowest-cost primitive is selected and executed.

---

## 2. Input: Ray-Casted Depth Image

YOPO's neural network does **not** consume raw rays. The simulator pre-renders depth images via ray casting, and these images become the network input.

### 2.1 Ray Casting in the Simulator

**File**: `Simulator/src/src/sensor_simulator.cpp`

Camera intrinsics (from `Simulator/src/include/sensor_simulator.h`):

| Parameter | Value |
|-----------|-------|
| `fx, fy`  | 80.0  |
| `cx`      | 80.0  |
| `cy`      | 45.0  |
| Image size | 160 × 90 |

For each pixel `(u, v)`, a ray direction in camera frame is computed:

```
d_cam = (1, -(u-cx)/fx, -(v-cy)/fy)
```

This ray is rotated to world frame via the drone's attitude, then `getIntersectedVoxelIndices()` on a PCL octree returns the first obstacle hit distance. The result is a 16-bit millimeter depth image.

For batch dataset generation, a CUDA-accelerated ray caster (`Simulator/src/src/dataset_generator.cpp`) is used.

### 2.2 Depth Image Preprocessing

**File**: `YOPO/policy/yopo_dataset.py:86-87`

```python
image = cv2.imread(self.img_list[item], -1).astype(np.float32)
image = np.expand_dims(cv2.resize(image, (self.width, self.height),
    interpolation=cv2.INTER_NEAREST) / 65535.0, axis=0)
```

Steps:
1. Load 16-bit PNG depth as float32
2. Resize from 160×90 → **160×96** (network input shape)
3. Normalize from `[0, 65535]` → `[0, 1]`

**Config values**: `YOPO/config/traj_opt.yaml:17-18`
```yaml
image_height: 96
image_width: 160
```

---

## 3. Primitive Anchors: Predefined Deterministic Angular Lattice

The 15 primitive anchors form a fixed spherical grid whose angles are **fully predefined and deterministic** — the network does not predict anchor directions. The network only predicts small offsets (`delta_yaw`, `delta_pitch`) relative to these anchor directions, plus a radial distance adjustment.

### 3.1 Lattice Configuration

**File**: `YOPO/config/traj_opt.yaml:21-28`

```yaml
horizon_num: 5
vertical_num: 3
horizon_camera_fov: 90.0      # degrees, total horizontal spread of anchors
vertical_camera_fov: 60.0     # degrees, total vertical spread of anchors
horizon_anchor_fov: 30.0      # degrees, allowed delta_yaw range → half = ±15°
vertical_anchor_fov: 30.0     # degrees, allowed delta_pitch range → half = ±15°
radio_range: 5.0              # meters, anchor radius → planning horizon = 10m
radio_num: 1                  # single radial layer
```

Total primitives: `traj_num = horizon_num × vertical_num × radio_num = 5 × 3 × 1 = 15`.  
See `YOPO/config/config.py:13`:
```python
self._data["traj_num"] = self._data['horizon_num'] * self._data['vertical_num'] * self._data["radio_num"]
```

### 3.2 Anchor Angle Computation (Deterministic)

**File**: `YOPO/policy/primitive.py:47-56`

```python
if self.horizon_num == 1:
    direction_diff = 0
else:
    direction_diff = (self.horizon_fov / 180.0 * torch.pi) / self.horizon_num

if self.vertical_num == 1:
    altitude_diff = 0
else:
    altitude_diff = (self.vertical_fov / 180.0 * torch.pi) / self.vertical_num
radio_diff = self.radio_range / self.radio_num
```

| Derived Parameter | Formula | Value |
|-------------------|---------|-------|
| `direction_diff` | `90° / 5` | **18° = 0.314 rad** |
| `altitude_diff` | `60° / 3` | **20° = 0.349 rad** |
| `radio_diff` | `5.0 / 1` | **5.0 m** |

### 3.3 Anchor Position Computation

**File**: `YOPO/policy/primitive.py:62-77`

```python
for h in range(0, self.radio_num):
    for i in range(0, self.vertical_num):
        for j in range(0, self.horizon_num):
            search_radio = (h + 1) * radio_diff
            alpha = -direction_diff * (self.horizon_num - 1) / 2 + j * direction_diff
            beta  = -altitude_diff * (self.vertical_num - 1) / 2 + i * altitude_diff

            pos_node = [cos(beta) * cos(alpha) * search_radio,
                        cos(beta) * sin(alpha) * search_radio,
                        sin(beta) * search_radio]
```

Each primitive anchor `i` is defined by `(alpha_i, beta_i, search_radio)`:

```
alpha_i = -18° × (5-1)/2 + j × 18° = -36° + j × 18°
beta_i  = -20° × (3-1)/2 + i × 20° = -20° + i × 20°
```

**The full 15-primitive lattice** (grid index layout, bottom-left origin, per `primitive.py:31-39`):

| Grid ID | Image ID | `j` | `i` | `alpha` (yaw, °) | `beta` (pitch, °) |
|---------|----------|-----|------|-------------------|---------------------|
| 0  | 14 | 0 | 0 | −36° | −20° |
| 1  | 13 | 1 | 0 | −18° | −20° |
| 2  | 12 | 2 | 0 |   0° | −20° |
| 3  | 11 | 3 | 0 | +18° | −20° |
| 4  | 10 | 4 | 0 | +36° | −20° |
| 5  |  9 | 0 | 1 | −36° |   0° |
| 6  |  8 | 1 | 1 | −18° |   0° |
| 7  |  7 | 2 | 1 |   0° |   0° |
| 8  |  6 | 3 | 1 | +18° |   0° |
| 9  |  5 | 4 | 1 | +36° |   0° |
| 10 |  4 | 0 | 2 | −36° | +20° |
| 11 |  3 | 1 | 2 | −18° | +20° |
| 12 |  2 | 2 | 2 |   0° | +20° |
| 13 |  1 | 3 | 2 | +18° | +20° |
| 14 |  0 | 4 | 2 | +36° | +20° |

Note: `Image ID = convert_ImageGrid_LatticeID(Grid ID) = 14 - Grid ID` (`primitive.py:104-105`). The lattice is enumerated bottom-to-top, right-to-left, but the image grid (network output) uses the opposite order.

### 3.4 Anchor Rotation Matrices (Body → Primitive)

**File**: `YOPO/policy/primitive.py:76-77`

```python
Rotation = R.from_euler('ZYX', [alpha, -beta, 0.0], degrees=False)
lattice_Rbp_list.append(torch.tensor(Rotation.as_matrix()))
```

Each primitive defines a rotation matrix `R_bp` that transforms vectors from the primitive frame to the body frame, using ZYX Euler angles `[alpha, -beta, 0]` (yaw-pitch-roll).

### 3.5 Network Prediction Offsets (Non-Deterministic Part)

**File**: `YOPO/policy/primitive.py:83-84`

```python
self.yaw_diff = 0.5 * self.horizon_anchor_fov / 180.0 * torch.pi   # = 0.262 rad = 15°
self.pitch_diff = 0.5 * self.vertical_anchor_fov / 180.0 * torch.pi # = 0.262 rad = 15°
```

The network predicts normalized offset values `pred ∈ [-1, 1]` that are scaled by these half-ranges:

```python
delta_yaw   = pred[0] * yaw_diff     # ∈ [−15°, +15°]
delta_pitch = pred[1] * pitch_diff   # ∈ [−15°, +15°]
radio       = (pred[2] + 1.0) * radio_range  # ∈ [0, 10] m
```

**The anchor angles are predefined and fixed. The network does NOT predict which direction to fly — it only predicts bounded deltas around each anchor.**

### 3.6 Coverage and Overlap Analysis

The two FOV parameters serve distinct roles:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `horizon_camera_fov` | 90° | Defines anchor spacing: `direction_diff = 90° / 5 = 18°` |
| `vertical_camera_fov` | 60° | Defines anchor spacing: `altitude_diff = 60° / 3 = 20°` |
| `horizon_anchor_fov` | 30° | Defines delta range: each anchor can be adjusted by ±15° |
| `vertical_anchor_fov` | 30° | Defines delta range: each anchor can be adjusted by ±15° |

**Adjacent primitives intentionally overlap; there are no coverage gaps:**

```
Yaw coverage (anchor range ± delta):
  Anchor -36°:  [-51°, -21°]  ─╮ 12° overlap
  Anchor -18°:  [-33°,  -3°]  ─╯────╮ 12° overlap
  Anchor   0°:  [-15°, +15°]  ─╮────╯────╮ 12° overlap
  Anchor +18°:  [ +3°, +33°]  ─╯─────────╯────╮ 12° overlap
  Anchor +36°:  [+21°, +51°]  ─╯──────────────╯
```

```
Pitch coverage (anchor range ± delta):
  Anchor -20°:  [-35°, -5°]  ─╮ 10° overlap
  Anchor   0°:  [-15°, +15°] ─╯────╮ 10° overlap
  Anchor +20°:  [ +5°, +35°] ──────╯
```

Overlap size per axis:

| Axis  | Anchor step `S` | Per-anchor span `2Δ` | Overlap zone |
|-------|-----------------|-----------------------|---------------|
| Yaw   | 18°             | 30° (2 × 15°)         | 12° (30° − 18°) |
| Pitch | 20°             | 30° (2 × 15°)         | 10° (30° − 20°) |

**Edge primitives extend beyond the nominal lattice boundaries** (defined by `horizon_camera_fov/2 = ±45°` and `vertical_camera_fov/2 = ±30°`):

| Edge anchor | Delta range | Extreme reach | Beyond lattice boundary by |
|-------------|-------------|----------------|----------------------------|
| Yaw −36° | −15° | −51° | 6° |
| Yaw +36° | +15° | +51° | 6° |
| Pitch −20° | −15° | −35° | 5° |
| Pitch +20° | +15° | +35° | 5° |

This is **not a bug**: the `camera_fov` values are only used to compute anchor spacing. The `anchor_fov` values independently bound the delta offsets, and the network can safely predict large offsets for edge primitives — there is simply no further anchor to overlap with, and the cost function will penalize infeasible trajectories.

---

## 4. Trajectory Representation

Each candidate trajectory is a set of three independent 5th-order polynomials — one per Cartesian axis `(x, y, z)` — satisfying boundary conditions at `t=0` and `t=T`.

### 4.1 Polynomial Form

**File**: `YOPO/policy/poly_solver.py:4-15`

```python
p(t) = A[0] + A[1]·t + A[2]·t² + A[3]·t³ + A[4]·t⁴ + A[5]·t⁵
v(t) = A[1] + 2A[2]·t + 3A[3]·t² + 4A[4]·t³ + 5A[5]·t⁴
a(t) = 2A[2] + 6A[3]·t + 12A[4]·t² + 20A[5]·t³
j(t) = 6A[3] + 24A[4]·t + 60A[5]·t²
```

### 4.2 Boundary Value Problem (Closed-Form)

**File**: `YOPO/policy/poly_solver.py:6-15`

```python
State_Mat = np.array([pos0, vel0, acc0, pos1, vel1, acc1])
t = Tf
Coef_inv = np.array([
    [1,   0,   0,              0,              0,              0],
    [0,   1,   0,              0,              0,              0],
    [0,   0,   1/2,            0,              0,              0],
    [-10/t³, -6/t², -3/(2t),  10/t³,         -4/t²,          1/(2t)],
    [15/t⁴,  8/t³,  3/(2t²), -15/t⁴,          7/t³,         -1/t²],
    [-6/t⁵, -3/t⁴, -1/(2t³),  6/t⁵,         -3/t⁴,          1/(2t³)]
])
self.A = np.dot(Coef_inv, State_Mat)   # [6] coefficients per axis
```

This is a strictly closed-form solution — no iterative optimization at inference time.

### 4.3 Start Boundary Conditions

The start conditions `(pos0, vel0, acc0)` come from the current vehicle state (from odometry), transformed into world frame (NWU).

### 4.4 End Boundary Conditions (Network Output)

The end conditions `(pos1, vel1, acc1)` are predicted by the network. The raw network output is:

**File**: `YOPO/policy/state_transform.py:12-51`

| Output Channel | Range | In Primitive Frame | Denormalization |
|---------------|-------|-------------------|-----------------|
| `pred[0]` | `[-1, 1]` → `tanh` | `delta_yaw ∈ [−15°, +15°]` | `× yaw_diff` |
| `pred[1]` | `[-1, 1]` → `tanh` | `delta_pitch ∈ [−15°, +15°]` | `× pitch_diff` |
| `pred[2]` | `[-1, 1]` → `tanh` | `radio ∈ [0, 10] m` | `(+ 1.0) × radio_range` |
| `pred[3:6]` | `[-1, 1]` → `tanh` | `end_vel_p` (primitive frame) | `× vel_max` |
| `pred[6:9]` | `[-1, 1]` → `tanh` | `end_acc_p` (primitive frame) | `× acc_max` |

End position is computed by converting `(yaw+delta_yaw, pitch+delta_pitch, radio)` from spherical to Cartesian in body frame:

```python
cos_pitch = cos(pitch + delta_pitch)
end_pos_b.x = cos_pitch * cos(yaw + delta_yaw) * radio
end_pos_b.y = cos_pitch * sin(yaw + delta_yaw) * radio
end_pos_b.z = sin(pitch + delta_pitch) * radio
```

End velocity and acceleration are denormalized in primitive frame, then rotated to body frame via `Rbp`:

```python
end_vel_b = Rbp @ end_vel_p
end_acc_b = Rbp @ end_acc_p
```

### 4.5 Segment Time

**File**: `YOPO/config/config.py:12`

```python
self._data["sgm_time"] = 2 * self._data["radio_range"] / self._data["vel_max_train"]
```

At test time, adjusted for speed ratio (`primitive.py:11`):

```python
self.segment_time = cfg["sgm_time"] / ratio
# where ratio = velocity / vel_max_train
```

For `vel_max_train=6 m/s` and `radio_range=5 m`: `T = 2×5/6 ≈ 1.67 s`.

---

## 5. Complete Data Flow

### 5.1 Training Pipeline

**File**: `YOPO/policy/yopo_dataset.py:83-107` (dataset)  
**File**: `YOPO/policy/yopo_network.py:42-55` (inference wrapper)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SIMULATOR (offline)                                          │
│    Ray-cast each pixel → depth image [160×90, uint16 mm]        │
│    Log aircraft pose → positions.csv, quaternions.csv             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. DATASET LOADER (YOPODataset.__getitem__)                     │
│    load PNG depth → float32, resize 160×96, normalize → [0,1] │
│    load pose → R_WB rotation matrix                             │
│    sample random vel_w, acc_w (lognormal for vx, normal for     │
│      vy/vz/acc) → rotate to body: vel_b = R_Bw @ vel_w         │
│    sample random goal_w (normal yaw σ=20°, pitch σ=10°)        │
│      → rotate to body: goal_b = R_Bw @ goal_w                   │
│    output: (depth[1,96,160], pos_w[3], rot_wb[3,3],            │
│             obs=[vel_b, acc_b, goal_b][9])                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. STATE PREPROCESSING (StateTransform)                         │
│                                                                 │
│   3a. normalize_obs(obs):                                       │
│       vel_b /= vel_max       → [-1, 1]                          │
│       acc_b /= acc_max       → [-1, 1]                          │
│       goal_b /= ||goal_b|| clamped to goal_length               │
│                                                                 │
│   3b. prepare_input(obs):                                       │
│       obs = [vel_b, acc_b, goal_b] in body frame [B, 9]        │
│       For each of 15 primitives:                                │
│         obs_p = Rbp @ obs       (body → primitive frame)       │
│       Output: [B, 9, 3, 5]      ← 9 channels in 3×5 grid       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. NEURAL NETWORK (YopoNetwork.forward)                         │
│                                                                 │
│   depth [B, 1, 96, 160]                                        │
│     → ResNet18 backbone → features [B, 64, 3, 5]               │
│                                                                 │
│   obs in primitive frame [B, 9, 3, 5]                           │
│     → concatenate → [B, 73, 3, 5]                               │
│     → 3× 1×1 conv head → [B, 10, 3, 5]                         │
│                                                                 │
│   output split:                                                 │
│     endstate [B, 9, 3, 5]  ← tanh(output[:, :9])               │
│     score    [B, 1, 3, 5]   ← softplus(output[:, 9])            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. POSTPROCESSING (StateTransform.pred_to_endstate)             │
│                                                                 │
│   For each of 15 primitives (i=0..14):                          │
│     delta_yaw   = pred[0] × yaw_diff   (range: ±15°)            │
│     delta_pitch = pred[1] × pitch_diff  (range: ±15°)           │
│     radio       = (pred[2]+1) × radio_range  (range: [0,10]m)  │
│                                                                 │
│     anchor_yaw, anchor_pitch = getAngleLattice(i)   (predefined)│
│     end_pos_b = spherical_to_cartesian(                         │
│       anchor_yaw + delta_yaw,                                   │
│       anchor_pitch + delta_pitch,                               │
│       radio)                                                    │
│                                                                 │
│     end_vel_p = pred[3:6] × vel_max   (primitive frame)        │
│     end_acc_p = pred[6:9] × acc_max   (primitive frame)        │
│     end_vel_b = Rbp[i] @ end_vel_p   → body frame              │
│     end_acc_b = Rbp[i] @ end_acc_p   → body frame              │
│                                                                 │
│   Output: endstate [B, 9, 3, 5] in body frame                   │
│           score    [B, 3, 5]                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. LOSS COMPUTATION (training only)                              │
│                                                                 │
│   For each primitive:                                            │
│     start = (pos_w, vel_w, acc_w)         (world frame)         │
│     end   = transform_body2world(endstate)                      │
│     build Poly5Solver(start, end, T)                            │
│                                                                 │
│     C_smooth = ws × ∫jerk² dt + wa × ∫acc² dt                  │
│              = ws × dᵀ·R_J·d + wa × dᵀ·R_A·d                    │
│     C_safety = wc × Σ exp(-(dᵢ - d₀) / r)   (30 sample pts)    │
│       collision avoidance: ESDF distance dᵢ to nearest obstacle  │
│       d₀ = 1.2m safe margin, r = 0.6m decay. cost → 0 if        │
│       dᵢ ≫ d₀, explodes exponentially if dᵢ < d₀ or inside obs  │
│     C_guide  = wg × (||goal - traj_along||₁ + 0.5·||traj_perp||₂)│
│                                                                 │
│     total_loss = C_smooth + C_safety + C_guide                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. RUNTIME EXECUTION (test_yopo_ros.py)                         │
│                                                                 │
│   7a. Receive depth image + odometry from ROS                   │
│   7b. Extract vel_w, acc_w, goal_w from odometry                │
│       → rotate to body frame (vel_b, acc_b, goal_b)             │
│       → normalize → prepare_input → network forward             │
│   7c. Select primitive with minimum score                       │
│   7d. Build Poly5Solver for best primitive                      │
│   7e. At each control tick (50 Hz, dt=0.02s):                   │
│         evaluate p(t), v(t), a(t) at current t                 │
│         compute yaw from velocity + goal direction              │
│         publish PositionCommand(pos, vel, acc, yaw, yaw_dot)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Key Parameters Summary

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Planning horizon | `2 × radio_range` | **10.0 m** | `traj_opt.yaml:27` |
| Max training velocity | `vel_max_train` | **6.0 m/s** | `traj_opt.yaml:6` |
| Max training acceleration | `acc_max_train` | **6.0 m/s²** | `traj_opt.yaml:7` |
| Segment time | `T` | **1.667 s** | `config.py:12`, `2×5.0/6.0` |
| Number of primitives | `traj_num` | **15** (3×5×1) | `config.py:13` |
| Horizontal anchor spread | `horizon_camera_fov` | **90°** | `traj_opt.yaml:23` |
| Vertical anchor spread | `vertical_camera_fov` | **60°** | `traj_opt.yaml:24` |
| Anchor yaw step | `direction_diff` | **18°** (0.314 rad) | `primitive.py:50` |
| Anchor pitch step | `altitude_diff` | **20°** (0.349 rad) | `primitive.py:55` |
| Yaw offset range | `±yaw_diff` | **±15°** (0.262 rad) | `primitive.py:83` |
| Pitch offset range | `±pitch_diff` | **±15°** (0.262 rad) | `primitive.py:84` |
| Radial range | `radio_range` | **[0, 10] m** | `traj_opt.yaml:27` |
| Depth image size | — | **160 × 96** | `traj_opt.yaml:17-18` |
| Control rate | — | **50 Hz** (dt=0.02s) | `test_yopo_ros.py` |
| Network output channels | — | **10** (9 state + 1 score) | `yopo_network.py:19` |
| Backbone hidden dim | `hidden_state` | **64** | `yopo_network.py:20` |
| Collision safe distance | `d0` | **1.2 m** | `traj_opt.yaml:31` |
| Collision decay rate | `r` | **0.6 m** | `traj_opt.yaml:32` |
| Collision weight | `wc` | **1.0** | `traj_opt.yaml:13` |
| ESDF eval samples | — | **30** pts per trajectory | `safety_loss.py:23` |
| ESDF voxel resolution | — | **0.2 m** | `safety_loss.py:28` |

---

## 7. Frame Conventions

| Frame | Axes | Origin | Usage |
|-------|------|--------|-------|
| **World** `W` | NWU (North-West-Up) | Inertial origin | Absolute position, ESDF map |
| **Body** `B` | ZYX: X=forward, Y=left, Z=up | Drone CoM | Odometry input, control output |
| **Primitive** `Pᵢ` | Z points toward anchor `i` | Body origin | Network predicts offsets in this frame |
| **Camera** `C` | Z=forward (into depth), X=right, Y=down | Camera sensor | Ray casting |

World ↔ Body transforms handled by `scipy.spatial.transform.Rotation` with ZYX Euler convention.

**File**: `YOPO/policy/state_transform.py:120-144`

```python
def rotate_body2world(rot_wb, pos_b):
    pos_w = torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)
    return pos_w

def transform_body2world(rot_wb, t_w, pos_b):
    return rotate_body2world(rot_wb, pos_b) + t_w
```

---

## 8. Summary

```
               SIMULATOR (offline / test-time replay)
                        │
               ray-cast per pixel
               PCL octree intersection
                        │
                        ▼
              depth image [160×90]
                        │
                  resize + normalize
                        │
        ┌───────────────▼───────────────┐
        │     current state (odometry)   │
        │  vel_b, acc_b, goal_b [B,9]   │
        └───────────────┬───────────────┘
                        │
           normalize + rotate to each
           of 15 predefined primitive frames
                        │
        ┌───────────────▼───────────────┐
        │       YopoNetwork forward      │
        │  ResNet18(depth) ⊕ obs → head  │
        │        endstate + score        │
        └───────────────┬───────────────┘
                        │
        pred_to_endstate: denormalize
        delta_yaw/pitch + spherical→Cart
        rotate vel/acc to body frame
                        │
        ┌───────────────▼───────────────┐
        │    15 candidate end states     │
        │    (pos, vel, acc) in body     │
        └───────────────┬───────────────┘
                        │
          select min-score primitive
                        │
        ┌───────────────▼───────────────┐
        │   Poly5Solver(start, end, T)   │
        │  3 independent 5th-order polys │
        │   evaluate at 50 Hz control    │
        └───────────────┬───────────────┘
                        │
                  PositionCommand
                 (pos, vel, acc, yaw)
                        │
                        ▼
                 Flight Controller
```

**Angles are predefined.** The 15 anchor directions `(alpha_i, beta_i)` are computed once at initialization from `horizon_camera_fov=90°` and `vertical_camera_fov=60°`, uniformly spaced in 5 horizontal and 3 vertical steps. The network only predicts bounded offsets `delta_yaw ∈ [−15°, +15°]` and `delta_pitch ∈ [−15°, +15°]` relative to each anchor.

**Trajectories are 5th-order polynomials** solved in closed form from `(pos, vel, acc)` at t=0 and t=T. No iterative optimization is needed at runtime — the `Coef_inv` matrix maps boundary conditions directly to polynomial coefficients.
