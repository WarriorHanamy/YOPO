"""Benchmark latency across PyTorch, TorchScript, ONNX Runtime, and TensorRT."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from YOPO.policy import YopoNetwork


@dataclass
class BenchmarkResult:
    """Latency statistics for one backend."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    speedup: float = 0.0  # relative to pytorch eager


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    pytorch: BenchmarkResult | None = None
    torchscript: BenchmarkResult | None = None
    onnx: BenchmarkResult | None = None
    tensorrt: BenchmarkResult | None = None

    _order: list[str] = field(
        default_factory=lambda: ['pytorch', 'torchscript', 'onnx', 'tensorrt']
    )

    def print(self) -> None:
        """Pretty-print benchmark results."""
        print()
        print('=' * 72)
        print(f'  Benchmark Report')
        print('=' * 72)
        header = f'  {"Backend":<16} {"Mean(ms)":>10} {"Std(ms)":>10} {"Min(ms)":>10} {"Max(ms)":>10} {"Speedup":>8}'
        print(header)
        print('  ' + '-' * 68)
        for key in self._order:
            r: BenchmarkResult | None = getattr(self, key, None)
            if r is None:
                print(f'  {key:<16} {"--":>10} {"--":>10} {"--":>10} {"--":>10} {"--":>8}')
            else:
                print(
                    f'  {key:<16} {r.mean_ms:>8.3f}   {r.std_ms:>8.3f}   {r.min_ms:>8.3f}   {r.max_ms:>8.3f}'
                    f'  {r.speedup:>6.1f}x'
                )
        print('=' * 72)


def _measure_torch(model, depth, obs, warmup: int, iterations: int) -> BenchmarkResult:
    """Measure torch/TorchScript model latency."""
    for _ in range(warmup):
        model(depth, obs)
    if depth.device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model(depth, obs)
        if depth.device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return BenchmarkResult(
        mean_ms=float(torch.tensor(times).mean().item()),
        std_ms=float(torch.tensor(times).std().item()),
        min_ms=float(min(times)),
        max_ms=float(max(times)),
    )


def _measure_onnx(session, depth_np, obs_np, warmup: int, iterations: int) -> BenchmarkResult:
    """Measure ONNX Runtime latency."""
    for _ in range(warmup):
        session.run(None, {'depth_image': depth_np, 'obs_grid': obs_np})

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {'depth_image': depth_np, 'obs_grid': obs_np})
        times.append((time.perf_counter() - t0) * 1000)

    return BenchmarkResult(
        mean_ms=float(torch.tensor(times).mean().item()),
        std_ms=float(torch.tensor(times).std().item()),
        min_ms=float(min(times)),
        max_ms=float(max(times)),
    )


def _numpy_inputs(config):
    import numpy as np

    depth = np.zeros((1, 1, 96, 160), dtype=np.float32)
    obs = np.zeros(
        (1, 9, config.lattice.vertical_num, config.lattice.horizon_num), dtype=np.float32
    )
    return depth, obs


def _torch_inputs(config, device):
    depth = torch.zeros([1, 1, 96, 160], dtype=torch.float32, device=device)
    obs = torch.zeros(
        [1, 9, config.lattice.vertical_num, config.lattice.horizon_num],
        dtype=torch.float32,
        device=device,
    )
    return depth, obs


def run_benchmark(
    config,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    warmup: int = 100,
    iterations: int = 500,
    device: str | None = None,
) -> BenchmarkReport:
    """Benchmark all available backends from an exported artifact directory.

    @param[in] config  YOPOConfig instance
    @param[in] output_dir  Artifact directory containing model.onnx, model.pt, model.engine
    @param[in] checkpoint_path  Optional checkpoint for PyTorch eager benchmark
    @param[in] warmup  Number of warmup iterations
    @param[in] iterations  Number of measured iterations
    @param[in] device  Override device (default: cuda if available else cpu)
    @return BenchmarkReport
    """
    out = Path(output_dir)
    report = BenchmarkReport()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth, obs = _torch_inputs(config, device)

    # ---- PyTorch eager (state_dict if available, else TorchScript) ----
    ts_path = out / 'model.pt'
    if checkpoint_path:
        state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
        model = YopoNetwork()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        report.pytorch = _measure_torch(model, depth, obs, warmup, iterations)

    # ---- TorchScript ----
    if ts_path.exists():
        ts_model = torch.jit.load(str(ts_path), map_location=device)
        ts_model.eval()
        report.torchscript = _measure_torch(ts_model, depth, obs, warmup, iterations)
    elif checkpoint_path and report.pytorch is not None:
        # Fallback: trace from eager model for TorchScript benchmark
        ts_model = torch.jit.trace(model, (depth, obs))
        report.torchscript = _measure_torch(ts_model, depth, obs, warmup, iterations)

    # ---- ONNX Runtime ----
    onnx_path = out / 'model.onnx'
    if onnx_path.exists():
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(onnx_path))
            depth_np, obs_np = _numpy_inputs(config)
            report.onnx = _measure_onnx(session, depth_np, obs_np, warmup, iterations)
        except ImportError:
            print('  [benchmark] onnxruntime not installed, skipping ONNX')

    # ---- TensorRT ----
    engine_path = out / 'model.engine'
    if engine_path.exists() and device == 'cuda':
        try:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(str(engine_path), map_location='cuda'))
            model_trt.eval()
            depth, obs = _torch_inputs(config, device)
            report.tensorrt = _measure_torch(model_trt, depth, obs, warmup, iterations)
        except ImportError:
            print('  [benchmark] torch2trt not installed, skipping TensorRT')
        except Exception as e:
            print(f'  [benchmark] TensorRT error: {e}')

    # Compute speedup relative to PyTorch eager
    if report.pytorch is not None:
        pytorch_mean = report.pytorch.mean_ms
        for r in [report.torchscript, report.onnx, report.tensorrt]:
            if r is not None and pytorch_mean > 0:
                r.speedup = pytorch_mean / r.mean_ms

    return report
