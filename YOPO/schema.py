"""YOPO configuration schema and artifact contracts.

All config parameters are defined here with Pydantic v2 validation.
The module-level ``config`` singleton is loaded from the default YAML file
on first import. Paths are resolved relative to the YOPO/ package root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)
from ruamel.yaml import YAML

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _package_root() -> Path:
    """Absolute path to the YOPO/ package directory."""
    return Path(__file__).resolve().parent


def _resolve(path: str | Path) -> Path:
    """Resolve a relative path against the package root."""
    return (_package_root() / path).resolve()


# ---------------------------------------------------------------------------
# Sub-config models
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    """Dataset location and image dimensions."""

    dataset_path: Path = Field(
        default_factory=lambda: Path('../dataset/data'),
        description='Path relative to YOPO/ package root',
    )
    image_height: int = Field(96, ge=1, description='Input depth image height [px]')
    image_width: int = Field(160, ge=1, description='Input depth image width [px]')

    @property
    def resolved(self) -> Path:
        return _resolve(self.dataset_path)


class CostWeights(BaseModel):
    """Loss weight coefficients."""

    wg: float = Field(0.15, ge=0.0, description='Guidance (goal approach) weight')
    ws: float = Field(10.0, ge=0.0, description='Smoothness (jerk) weight')
    wa: float = Field(1.0, ge=0.0, description='Acceleration weight')
    wc: float = Field(1.0, ge=0.0, description='Collision (safety) weight')


class LatticeConfig(BaseModel):
    """Motion primitive grid parameters."""

    horizon_num: int = Field(5, ge=1, description='Number of horizontal primitives')
    vertical_num: int = Field(3, ge=1, description='Number of vertical primitives')
    horizon_camera_fov: float = Field(
        90.0, gt=0, le=180, description='Horizontal camera FOV [deg]'
    )
    vertical_camera_fov: float = Field(60.0, gt=0, le=180, description='Vertical camera FOV [deg]')
    horizon_anchor_fov: float = Field(
        30.0, gt=0, le=180, description='Horiz. anchor adjustment range [deg]'
    )
    vertical_anchor_fov: float = Field(
        30.0, gt=0, le=180, description='Vert. anchor adjustment range [deg]'
    )
    radio_range: float = Field(5.0, gt=0, description='Planning horizon radius [m]')
    radio_num: int = Field(1, ge=1, description='Number of radio layers (only 1 supported)')


class SafetyConfig(BaseModel):
    """ESDF collision cost function parameters."""

    d0: float = Field(1.2, gt=0, description='Safe distance threshold [m]')
    r: float = Field(0.6, gt=0, description='Cost fall-off scale [m]')


class StateSamplingConfig(BaseModel):
    """Random state sampling distribution for imitation learning targets."""

    vx_mean_unit: float = 0.4
    vy_mean_unit: float = 0.0
    vz_mean_unit: float = 0.0
    vx_std_unit: float = Field(2.0, gt=0)
    vy_std_unit: float = Field(0.45, gt=0)
    vz_std_unit: float = Field(0.3, gt=0)
    ax_mean_unit: float = 0.0
    ay_mean_unit: float = 0.0
    az_mean_unit: float = 0.0
    ax_std_unit: float = Field(0.5, gt=0)
    ay_std_unit: float = Field(0.5, gt=0)
    az_std_unit: float = Field(0.3, gt=0)
    goal_pitch_std: float = Field(10.0, ge=0, description='Goal pitch std [deg]')
    goal_yaw_std: float = Field(20.0, ge=0, description='Goal yaw std [deg]')


class MapConfig(BaseModel):
    """ESDF map boundary expansion."""

    map_expand_min: Annotated[
        tuple[float, float, float],
        Field(
            default=(0.0, 0.0, 0.2),
            description='Min expansion beyond point cloud bounds [m]',
        ),
    ] = (0.0, 0.0, 0.2)
    map_expand_max: Annotated[
        tuple[float, float, float],
        Field(
            default=(0.0, 0.0, 6.0),
            description='Max expansion (z-up to avoid sky-as-obstacle) [m]',
        ),
    ] = (0.0, 0.0, 6.0)


# ---------------------------------------------------------------------------
# Artifact models (I/O contracts)
# ---------------------------------------------------------------------------


class DatasetManifest(BaseModel):
    """Resolved dataset layout validated at training start."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    maps: list[Path]  # pointcloud-*.ply files for ESDF
    poses: list[Path]  # pose-*.csv files
    image_dirs: list[Path]  # 0/, 1/, ... sub-directories with depth PNGs

    @classmethod
    def discover(cls, data_root: Path) -> DatasetManifest:
        """Scan a dataset directory and build a manifest."""
        import glob as _glob

        data_root = data_root.resolve()
        maps = sorted(Path(p) for p in _glob.glob(str(data_root / 'pointcloud-*.ply')))
        poses = sorted(Path(p) for p in _glob.glob(str(data_root / 'pose-*.csv')))
        image_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit())
        return cls(root=data_root, maps=maps, poses=poses, image_dirs=image_dirs)

    @property
    def num_maps(self) -> int:
        return len(self.maps)


class TrainOutput(BaseModel):
    """Output artifacts produced by a training run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    log_dir: Path
    checkpoint: Path | None = None


class TRTOutput(BaseModel):
    """Output artifact from TensorRT conversion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trt_file: Path


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class YOPOConfig(BaseModel):
    """Top-level YOPO configuration with validation and computed fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- training envelope --
    velocity: float = Field(6.0, gt=0, description='Velocity in testing [m/s] (modifiable)')
    vel_max_train: float = Field(
        6.0,
        gt=0,
        description='Max velocity used during training [m/s] — determines '
        'derived parameters via denormalization',
    )
    acc_max_train: float = Field(6.0, gt=0, description='Max acceleration in training [m/s²]')

    # -- sub-config blocks --
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    cost_weights: CostWeights = Field(default_factory=CostWeights)
    lattice: LatticeConfig = Field(default_factory=LatticeConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    sampling: StateSamplingConfig = Field(default_factory=StateSamplingConfig)
    map: MapConfig = Field(default_factory=MapConfig)

    @model_validator(mode='after')
    def _validate_radio(self) -> YOPOConfig:
        if self.lattice.radio_num != 1:
            raise ValueError('radio_num only supports 1 currently')
        return self

    # -- computed fields (mirror old config.py derived values) ----------

    @computed_field
    @property
    def goal_length(self) -> float:
        """Planning horizon: 2 × radio_range [m]."""
        return 2.0 * self.lattice.radio_range

    @computed_field
    @property
    def sgm_time(self) -> float:
        """Single segment time: horizon / max velocity [s]."""
        return 2 * self.lattice.radio_range / self.vel_max_train

    @computed_field
    @property
    def traj_num(self) -> int:
        """Total number of lattice primitives (V × H × R)."""
        return self.lattice.horizon_num * self.lattice.vertical_num * self.lattice.radio_num

    @property
    def vel_scale(self) -> float:
        """Denormalization scale = vel_max_train / 1.0."""
        return self.vel_max_train / 1.0

    # -- artifact resolution --------------------------------------------

    def resolve_dataset(self) -> DatasetManifest:
        """Scan the configured dataset path and return a manifest."""
        return DatasetManifest.discover(self.dataset.resolved)

    def resolve_log_dir(self, base_log_dir: Path | None = None) -> Path:
        """Find next auto-incremented YOPO_N log subdirectory."""
        base = base_log_dir or _resolve('saved')
        base.mkdir(parents=True, exist_ok=True)
        existing = [
            int(p.name.split('_')[1])
            for p in base.iterdir()
            if p.is_dir() and p.name.startswith('YOPO_') and p.name.split('_')[1].isdigit()
        ]
        next_n = max(existing, default=-1) + 1
        path = base / f'YOPO_{next_n}'
        path.mkdir(parents=True, exist_ok=False)
        return path

    def resolve_checkpoint_path(self, trial: int, epoch: int) -> Path:
        """Resolve the path to a specific saved checkpoint."""
        return _resolve('saved') / f'YOPO_{trial}' / f'epoch{epoch}.pth'

    # -- YAML round-trip ------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> YOPOConfig:
        """Load config from a YAML file, filling defaults for missing keys."""
        path = Path(path)
        if not path.is_absolute():
            path = _resolve(path)
        data = YAML().load(path.read_text())
        # Map flat YAML keys into nested sub-config dicts
        nested = cls._nest(data)
        return cls.model_validate(nested)

    def to_yaml(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        path = Path(path)
        if not path.is_absolute():
            path = _resolve(path)
        # Flatten nested models back to flat dict for YAML
        flat = self._flatten()
        with open(path, 'w') as f:
            YAML().dump(flat, f)

    @staticmethod
    def _nest(data: dict) -> dict:
        """Convert flat YAML dict (old format) into nested model structure."""
        nested: dict = {}
        _ns = nested
        for section, keys in [
            ('dataset', ['dataset_path', 'image_height', 'image_width']),
            ('cost_weights', ['wg', 'ws', 'wa', 'wc']),
            (
                'lattice',
                [
                    'horizon_num',
                    'vertical_num',
                    'horizon_camera_fov',
                    'vertical_camera_fov',
                    'horizon_anchor_fov',
                    'vertical_anchor_fov',
                    'radio_range',
                    'radio_num',
                ],
            ),
            ('safety', ['d0', 'r']),
            (
                'sampling',
                [
                    'vx_mean_unit',
                    'vy_mean_unit',
                    'vz_mean_unit',
                    'vx_std_unit',
                    'vy_std_unit',
                    'vz_std_unit',
                    'ax_mean_unit',
                    'ay_mean_unit',
                    'az_mean_unit',
                    'ax_std_unit',
                    'ay_std_unit',
                    'az_std_unit',
                    'goal_pitch_std',
                    'goal_yaw_std',
                ],
            ),
            ('map', ['map_expand_min', 'map_expand_max']),
        ]:
            sub = {}
            for k in keys:
                if k in data:
                    sub[k] = data.pop(k)
            if sub:
                nested[section] = sub
        nested.update(data)  # remaining top-level keys
        return nested

    def _flatten(self) -> dict:
        """Flatten nested models to a flat YAML-friendly dict."""
        d = self.model_dump(
            mode='json',
            exclude={'goal_length', 'sgm_time', 'traj_num'},
        )
        out: dict = {}
        for key, val in d.items():
            if isinstance(val, dict):
                out.update(val)
            else:
                out[key] = val
        return {k: v for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_YAML = _resolve('config/traj_opt.yaml')

_yaml = YAML()
_yaml.default_flow_style = False

config: YOPOConfig

if _DEFAULT_YAML.exists():
    raw = _yaml.load(_DEFAULT_YAML.read_text())
    config = YOPOConfig.model_validate(YOPOConfig._nest(raw))
else:
    config = YOPOConfig()
