
# You Only Plan Once

Original Paper: [You Only Plan Once: A Learning-Based One-Stage Planner With Guidance Learning](https://ieeexplore.ieee.org/document/10528860)

Improvements and Applications: [YOPOv2-Tracker: An End-to-End Agile Tracking and Navigation Framework from Perception to Action](https://arxiv.org/html/2505.06923v1)

Video of the paper: [YouTube](https://youtu.be/m7u1MYIuIn4), [bilibili](https://www.bilibili.com/video/BV15M4m1d7j5)

Some realworld experiment: [YouTube](https://youtu.be/LHvtbKmTwvE), [bilibili](https://www.bilibili.com/video/BV1jBpve5EkP)

<table>
  <tr>
    <td align="center" width="38.5%"><img src="docs/realworld_1.gif" alt="Fig1" width="100%"></td>
    <td align="center" width="38.5%"><img src="docs/realworld_2.gif" alt="Fig2" width="100%"></td>
    <td align="center" width="23.0%"><img src="docs/platform.gif" alt="Fig3" width="100%"></td>
  </tr>
</table>

**Faster and Simpler:** The code is greatly simplified and refactored in Python/PyTorch. We also replaced the simulator with our CUDA-accelerated randomized environment, which is faster, lightweight, and boundless. For the stable version consistent with our paper, please refer to the [main](https://github.com/TJU-Aerial-Robotics/YOPO/tree/main) branch.

### Hardware

Our drone hardware is also open-source — see [hardware_list.pdf](https://github.com/TJU-Aerial-Robotics/YOPO/releases/latest) and the [Release page](https://github.com/TJU-Aerial-Robotics/YOPO/releases/tag/hardware) for SolidWorks files of the carbon fiber frame.

## Introduction:
We propose **a learning-based planner for autonomous navigation in obstacle-dense environments** which integrates (i) perception and mapping, (ii) front-end path searching, and (iii) back-end optimization of classical methods into a single network. 

**Learning-based Planner:** Considering the multi-modal nature of the navigation problem and to avoid local minima around initial values, our approach adopts a set of motion primitives as anchor to cover the searching space, and predicts the offsets and scores of primitives for further improvement (like the one-stage object detector YOLO). 

**Training Strategy:** Compared to giving expert demonstrations as labels in imitation learning or exploring by trial-and-error in reinforcement learning, we directly back-propagate the gradients of trajectory costs (e.g. from ESDF) to the weights of network, which is simple, straightforward, accurate, and sequence-independent (free of online simulator interaction or rendering).

<table>
    <tr>
        <td align="center" style="border: none;"><img src="docs/primitive_trajectories.png" alt="Fig1" style="width: 80%;"></td>
        <td align="center" style="border: none;"><img src="docs/predicted_trajectories.png" alt="Fig2" style="width: 80%;"></td>
		<td align="center" style="border: none;"><img src="docs/proposed_guidance_learning.png" alt="Fig3" style="width: 100%;"></td>
    </tr>
    <tr>
        <td align="center" style="border: none;">primitive anchors</td>
        <td align="center" style="border: none;">predicted traj and scores</td>
		<td align="center" style="border: none;">learning method</td>
    </tr>
</table>


## Installation

The project uses Python >=3.12 with `uv` for dependency management. Controller and Simulator require ROS Noetic (Ubuntu 20.04).

**1. Clone the Code**
```
git clone --depth 1 git@github.com:TJU-Aerial-Robotics/YOPO.git
cd YOPO
```

**2. Install Python Dependencies**
```
uv sync
```

**3. Build Simulator Components** (optional — only needed for simulation testing)

Build the controller and dynamics simulator:
```
cd Controller
source /opt/ros/noetic/setup.bash
catkin_make
```
Build the environment and sensors simulator (see [Simulator README](Simulator/src/readme.md) for CUDA issues):
```
cd Simulator
source /opt/ros/noetic/setup.bash
catkin_make
```

## Test the Policy

Pre-trained weights are available at `YOPO/saved/YOPO_1/epoch50.pth`.

**1. Start the Controller and Dynamics Simulator**

See [Controller README](Controller/src/readme.md) for details.
```
cd Controller
source devel/setup.bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
```

**2. Start the Environment and Sensors Simulator**

See [Simulator README](Simulator/src/readme.md) for details. Configure sensor and environment via [config.yaml](Simulator/src/config/config.yaml).
```
cd Simulator
source devel/setup.bash
rosrun sensor_simulator sensor_simulator_cuda
```

**3. Start the YOPO Planner**

Configure flight speed in [traj_opt.yaml](YOPO/config/traj_opt.yaml). Pre-trained weights are at 6 m/s (0–6 m/s range). More models at [Releases](https://github.com/TJU-Aerial-Robotics/YOPO/releases).

```
uv run yopo train --trial=1 --epoch=50
```

**4. Visualization**

Start RViz to visualize the images and trajectory:
```
rviz -d Simulator/src/rviz.rviz
```

Left: Random Forest (maze_type=5); Right: 3D Perlin (maze_type=1).
<p align="center">
    <img src="docs/new_env.gif" alt="new_env" />
</p>

You can click the `2D Nav Goal` on RVIZ as the goal (the map is infinite so the goal is freely), just like the following GIF (Flightmare Simulator).

<p align="center">
    <img src="docs/click_in_rviz.gif" alt="click_in_rviz" />
</p>


## Train the Policy

**1. Data Collection**

Generate dataset via Docker (requires NVIDIA GPU). Collects ~100,000 samples in 1–2 minutes:
```
yopo data-gen
```
Data is saved to `./dataset/data/`. Configure via [config.yaml](docker/data-gen/config/config.yaml).

**2. Train the Policy**
```
uv run yopo train
```
~1 hour on RTX 3080 + i9-12900K for 50 epochs on 100,000 samples. On hybrid CPU architectures, bind to P-cores:
```
taskset -c 1,2,3,4 uv run yopo train
```
Monitor training logs:
```
uv run tensorboard --logdir=./saved
```
<p align="center">
    <img src="docs/train_log.png" alt="train_log" width="100%"/>
</p>

Configure trajectory optimization in [traj_opt.yaml](YOPO/config/traj_opt.yaml).


## TensorRT Deployment

TensorRT inference: ~1 ms (ResNet-14) to ~5 ms (ResNet-18) on NVIDIA Orin NX.

**1. Install TensorRT Tools**
```
uv pip install nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
uv pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git
```

**2. PyTorch to TensorRT**
```
uv run yopo trt --trial=1 --epoch=50
```

**3. Deploy**

See `uv run yopo trt --help` and the [Controller README](Controller/src/readme.md) for real-world deployment instructions.


## RKNN Deployment

Experimental: RK3566 (~20 ms with ResNet-14, INT8 quantization). RK3566/RK3588 guide coming soon.

## Citation

```
@article{YOPO,
  title={You Only Plan Once: A Learning-based One-stage Planner with Guidance Learning},
  author={Lu, Junjie and Zhang, Xuewei and Shen, Hongming and Xu, Liwen and Tian, Bailing},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```