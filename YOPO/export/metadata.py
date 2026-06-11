"""Generate EXPORT_RULE-compliant metadata YAML files.

Produces model_spec.yaml, observations_metadata.yaml, action_metadata.yaml
in the artifact directory.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ruamel.yaml import YAML

_yaml = YAML()
_yaml.default_flow_style = False
_yaml.indent(mapping=2, sequence=4, offset=2)


def generate_revision() -> str:
    """Produce revision string ``{YYYYMMDDTHHMMSSZ}-{commit8}``.

    @return Revision string, e.g. ``20260611T120000Z-abc12345``
    """
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    try:
        repo = Path(__file__).resolve().parent.parent.parent
        commit = subprocess.run(
            ['git', 'rev-parse', '--short=8', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=str(repo),
        ).stdout.strip()
    except Exception:
        commit = '00000000'
    return f'{timestamp}-{commit}'


def generate_metadata(
    config,
    output_dir: str | Path,
    revision: str,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """Write all three metadata YAML files into output_dir.

    @param[in] config  YOPOConfig instance
    @param[in] output_dir  Artifact directory
    @param[in] revision  Revision string (from generate_revision)
    @param[in] formats  List of exported formats (default: ['onnx'])
    @return Dict of metadata file paths
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_model_spec(out, config, revision, formats or ['onnx'])
    _write_observations_metadata(out, config)

    return {
        'model_spec': out / 'model_spec.yaml',
        'observations_metadata': out / 'observations_metadata.yaml',
    }


def _write_model_spec(out: Path, config, revision: str, formats: list[str]) -> None:
    model_file = 'model.onnx'
    model_format = 'onnx'
    model_opset = 18

    if 'torchscript' in formats and 'onnx' not in formats:
        model_file = 'model.pt'
        model_format = 'torchscript'
        model_opset = None

    spec = {
        'schema': 'c5pro.io/neural-policy/v1',
        'identity': {
            'name': 'yopo',
            'revision': revision,
            'task': None,
        },
        'model': {
            'file': model_file,
            'format': model_format,
            'opset': model_opset,
        },
        'input': [
            {
                'name': 'depth_image',
                'shape': [1, 1, 96, 160],
                'dtype': 'float32',
                'unit': 'm',
                'range': [0.04, 20.0],
                'layout_ref': None,
            },
            {
                'name': 'obs_grid',
                'shape': [1, 9, config.lattice.vertical_num, config.lattice.horizon_num],
                'dtype': 'float32',
                'unit': None,
                'range': None,
                'layout_ref': 'observations_metadata.yaml',
            },
        ],
        'output': [
            {
                'name': 'endstate',
                'shape': [1, 9, config.lattice.vertical_num, config.lattice.horizon_num],
                'dtype': 'float32',
            },
            {
                'name': 'score',
                'shape': [1, config.lattice.vertical_num, config.lattice.horizon_num],
                'dtype': 'float32',
            },
        ],
        'control': {
            'semantics': 'trajectory',
            'elements': [
                {'index': i, 'name': n, 'unit': u, 'range': None}
                for i, (n, u) in enumerate(
                    [
                        ('position.x', 'm'),
                        ('position.y', 'm'),
                        ('position.z', 'm'),
                        ('velocity.x', 'm/s'),
                        ('velocity.y', 'm/s'),
                        ('velocity.z', 'm/s'),
                        ('acceleration.x', 'm/s^2'),
                        ('acceleration.y', 'm/s^2'),
                        ('acceleration.z', 'm/s^2'),
                    ]
                )
            ],
            'constraints_ref': None,
        },
        'frequency': {'hz': 50},
    }

    with open(out / 'model_spec.yaml', 'w') as f:
        _yaml.dump(spec, f)
    print(f'  model_spec.yaml -> {out / "model_spec.yaml"}')


def _write_observations_metadata(out: Path, config) -> None:
    observations = [
        {
            'name': 'vel_primitive',
            'dim': 3,
            'offset': 0,
            'unit': 'm/s',
            'range': [-config.vel_max_train, config.vel_max_train],
        },
        {
            'name': 'acc_primitive',
            'dim': 3,
            'offset': 3,
            'unit': 'm/s^2',
            'range': [-config.acc_max_train, config.acc_max_train],
        },
        {
            'name': 'goal_primitive',
            'dim': 3,
            'offset': 6,
            'unit': None,
            'range': None,
        },
    ]

    with open(out / 'observations_metadata.yaml', 'w') as f:
        _yaml.dump(observations, f)
    print(f'  observations_metadata.yaml -> {out / "observations_metadata.yaml"}')
