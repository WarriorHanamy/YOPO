"""ONNX, TorchScript, and TensorRT export for YOPO network."""

from pathlib import Path

import torch

from YOPO.policy import YopoNetwork

from .metadata import generate_metadata, generate_revision


def _dummy_inputs(config, device):
    depth = torch.zeros([1, 1, 96, 160], dtype=torch.float32, device=device)
    obs = torch.zeros(
        [1, 9, config.lattice.vertical_num, config.lattice.horizon_num],
        dtype=torch.float32,
        device=device,
    )
    return depth, obs


def _load_policy(config, checkpoint_path: str | Path, device: str) -> YopoNetwork:
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    policy = YopoNetwork()
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy


def export_onnx(
    config,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    opset: int = 18,
    verbose: bool = False,
) -> Path:
    """Export YOPO policy to ONNX format.

    @param[in] config  YOPOConfig instance
    @param[in] checkpoint_path  Path to .pth state_dict
    @param[in] output_dir  Output directory for model_spec.yaml + model.onnx
    @param[in] opset  ONNX opset version (default 18)
    @param[in] verbose  ONNX export verbose logging
    @return Path to exported model.onnx
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = _load_policy(config, checkpoint_path, device)
    depth, obs = _dummy_inputs(config, device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    onnx_path = out / 'model.onnx'
    torch.onnx.export(
        net,
        (depth, obs),
        onnx_path,
        input_names=['depth_image', 'obs_grid'],
        output_names=['endstate', 'score'],
        dynamic_axes={
            'depth_image': {0: 'batch'},
            'obs_grid': {0: 'batch'},
            'endstate': {0: 'batch'},
            'score': {0: 'batch'},
        },
        opset_version=opset,
        verbose=verbose,
        external_data=False,
    )

    print(f'  ONNX -> {onnx_path}')
    return onnx_path


def export_torchscript(config, checkpoint_path: str | Path, output_dir: str | Path) -> Path:
    """Trace YOPO policy to TorchScript.

    @param[in] config  YOPOConfig instance
    @param[in] checkpoint_path  Path to .pth state_dict
    @param[in] output_dir  Output directory
    @return Path to exported model.pt
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = _load_policy(config, checkpoint_path, device)
    depth, obs = _dummy_inputs(config, device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts_path = out / 'model.pt'
    traced = torch.jit.trace(net, (depth, obs))
    traced.save(str(ts_path))

    print(f'  TorchScript -> {ts_path}')
    return ts_path


def export_trt(
    config, checkpoint_path: str | Path, output_dir: str | Path, *, fp16: bool = True
) -> Path:
    """Convert YOPO policy to TensorRT engine via torch2trt.

    Requires nvidia-tensorrt (NGC) + torch2trt (github.com/NVIDIA-AI-IOT/torch2trt).

    @param[in] config  YOPOConfig instance
    @param[in] checkpoint_path  Path to .pth state_dict
    @param[in] output_dir  Output directory
    @param[in] fp16  Enable FP16 inference
    @return Path to exported TRT engine file
    """
    try:
        from torch2trt import torch2trt
    except ImportError:
        raise ImportError(
            'torch2trt not found. Install:\n'
            '  pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com\n'
            '  git clone https://github.com/NVIDIA-AI-IOT/torch2trt && cd torch2trt && python setup.py install'
        )

    if not torch.cuda.is_available():
        raise RuntimeError('TensorRT requires CUDA')

    device = 'cuda'
    net = _load_policy(config, checkpoint_path, device)
    depth, obs = _dummy_inputs(config, device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_trt = torch2trt(net, [depth, obs], fp16_mode=fp16)

    engine_path = out / 'model.engine'
    torch.save(model_trt.state_dict(), str(engine_path))
    print(f'  TensorRT -> {engine_path}')
    return engine_path
