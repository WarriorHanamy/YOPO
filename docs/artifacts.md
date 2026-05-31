# YOPO Artifact Contracts

> Canonical reference for all pipeline I/O artifacts.
> Corresponding Pydantic schemas: `YOPO/schema.py`

---

## Input Artifacts

### Training Config

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `YOPO/config/traj_opt.yaml` (default)                              |
| Schema       | `YOPOConfig`                                                       |
| Format       | Flat YAML (compatible with legacy `config.py`)                     |
| Created by   | User, edited manually                                              |
| Consumed by  | `YOPO.policy.*` via `from YOPO.schema import config` singleton     |
| Lifecycle    | Define once per experiment; mutable via `_apply_config_to_singleton` |

### Sweep Configs

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `<user-defined>/` (e.g. `sweep_configs/exp_vmax6.yaml`)            |
| Schema       | `YOPOConfig`                                                       |
| Format       | Flat YAML                                                          |
| Naming       | `{experiment_name}.yaml`                                           |
| Created by   | User                                                               |
| Transferred  | `scp` to remote `{RemoteTarget.configs_path}/`                     |
| Consumed by  | `yopo sweep --config-dir` (local) or container sweep (remote)      |
| Provenance   | Copied to `saved/{stem}/config.yaml` after training                 |

### Dataset

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `dataset/data/` (local), `{RemoteTarget.dataset_path}/` (remote)   |
| Schema       | `DatasetManifest`                                                  |
| Contents     | `pointcloud-{0..N-1}.ply`, `pose-{0..N-1}.csv`, `{0..N-1}/img_*.png` |
| Created by   | `yopo data-gen` or `make data` (Docker `dataset_generator`)        |
| Mount        | `-v {host}:{container}`: host `→ /output` in data-gen, `→ /app/dataset/data` in train |
| Lifecycle    | Generated once, cached across deployments (volume mount)           |
| Constraint   | Must exist before training; entrypoint checks `.ply` files          |

### Pre-trained Checkpoint (optional)

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `YOPO/saved/YOPO_{trial}/epoch{epoch}.pth`                         |
| Schema       | `Path`                                                             |
| Created by   | Previous training run                                              |
| Consumed by  | `yopo train --pretrained 1 --trial N --epoch E`                    |
| Constraint   | `torch.load(weights_only=True)`                                     |

---

## Build Artifacts

### Docker Image

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Tag          | `yopo-train:latest` (configurable)                                 |
| Schema       | `DockerImageSpec`                                                  |
| Built by     | `yopo docker build` or `yopo deploy run`                           |
| Dockerfile   | `docker/train/Dockerfile` (multi-stage: data-gen + Python env)      |
| Context      | Project root                                                       |
| Size         | ~6-8 GB                                                            |

### Docker Image Tar

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `yopo-train.tar` (configurable via `DockerImageSpec.tar_path`)      |
| Schema       | `DockerImageSpec`                                                  |
| Created by   | `docker save -o {tar_path} {tag}`                                  |
| Transferred  | `scp` to remote, then `docker load -i`                             |
| Lifecycle    | Build once → export → send → load on remote (idempotent)           |

---

## Output Artifacts

### Checkpoint

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `{saved_base}/{experiment_name}/epoch{epoch}.pth`                  |
| Format       | PyTorch `state_dict`                                               |
| Created by   | `YopoTrainer.save_model()` (on exit or interval)                   |
| Consumed by  | `yopo train --pretrained`, `yopo trt`, inference                   |

### Config Copy (provenance)

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `{saved_base}/{experiment_name}/config.yaml`                       |
| Created by   | `cmd_sweep` copies source YAML after training                       |
| Purpose      | Reproducibility: exact config that produced the checkpoint          |

### TensorBoard Logs

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `{saved_base}/{experiment_name}/events.out.tfevents.*`             |
| Created by   | `torch.utils.tensorboard.SummaryWriter`                             |
| Consumed by  | `tensorboard --logdir {saved_base}`                                |

### TRT Model (optional)

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `yopo_trt.pth` (configurable via `--output`)                       |
| Created by   | `yopo trt` (requires `torch2trt` + NVIDIA TensorRT)                 |
| Consumed by  | `Controller/` ROS node (real-time inference on Jetson)              |

---

## Deploy Artifacts

### Pipeline Config

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Path         | `deploy.yaml`                                                      |
| Schema       | `DeployConfig`                                                     |
| Format       | Nested YAML                                                        |
| Created by   | User                                                               |
| Consumed by  | `yopo deploy {run,status,gather} --config deploy.yaml`             |

Example:
```yaml
image:
  tag: yopo-train:latest
  tar_path: yopo-train.tar
remote:
  host: user@10.0.0.5
  port: 22
  dataset_path: /data/yopo/dataset
  saved_path: /data/yopo/saved
  configs_path: /data/yopo/configs
sweep_config_dir: sweep_configs/
epochs: 50
batch_size: 16
lr: 0.00015
```

---

## Directory Layout (Remote Server)

```
/data/yopo/                      ← root on remote
├── dataset/                     ← {RemoteTarget.dataset_path}
│   ├── pointcloud-0.ply
│   ├── pointcloud-1.ply
│   ├── pose-0.csv
│   ├── 0/img_0.png
│   └── ...
├── saved/                       ← {RemoteTarget.saved_path}
│   ├── exp_vmax6/
│   │   ├── epoch50.pth
│   │   └── config.yaml
│   └── exp_vmax8/
│       └── ...
└── configs/                     ← {RemoteTarget.configs_path}
    ├── exp_vmax6.yaml
    └── exp_vmax8.yaml
```

## Pipeline Lifecycle Summary

```
[local]                  [remote]
───────                  ───────
deploy.yaml  ──────────▶ consumed by deploy commands
sweep_configs/*.yaml ──▶ scp ──▶ /data/yopo/configs/
yopo-train.tar ────────▶ scp ──▶ docker load
                                          │
                                          ▼
                              docker run -d yopo-train
                              ├─ dataset miss → generate
                              ├─ dataset hit  → skip
                              └─ sweep --config-dir /app/configs
                                          │
                                          ▼
                              /data/yopo/saved/exp_*/
                                          │
                              scp ◀───────┘
                                          │
                              /local/results/exp_*/
```
