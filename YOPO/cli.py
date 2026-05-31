"""YOPO CLI entry point."""

import argparse
import os
import random
import sys

from YOPO.schema import YOPOConfig, config


def configure_random_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _load_config(yaml_config: str | None) -> YOPOConfig:
    if yaml_config:
        return YOPOConfig.from_yaml(yaml_config)
    return config


def cmd_train(args: argparse.Namespace) -> None:
    from YOPO.policy import YopoTrainer

    cfg = _load_config(args.yaml_config)
    configure_random_seed(0)

    checkpoint_path = ''
    if args.pretrained:
        checkpoint_path = str(cfg.resolve_checkpoint_path(args.trial, args.epoch))

    trainer = YopoTrainer(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        loss_weight=[1.0, 1.0],
        tensorboard_path=cfg.resolve_log_dir(),
        checkpoint_path=checkpoint_path,
        save_on_exit=True,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    trainer.train(epoch=args.epochs)
    print('Run YOPO Finish!')


def cmd_trt(args: argparse.Namespace) -> None:
    import time

    import numpy as np
    import torch

    from YOPO.policy import YopoNetwork

    try:
        from torch2trt import torch2trt
    except ImportError:
        sys.exit(
            'torch2trt not found. Install with: uv pip install torch2trt '
            '(requires nvidia-tensorrt)'
        )

    cfg = _load_config(args.yaml_config)
    weight = str(cfg.resolve_checkpoint_path(args.trial, args.epoch))

    print('Loading Network...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(weight, weights_only=True)
    policy = YopoNetwork()
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()

    depth = np.zeros(shape=[1, 1, 96, 160], dtype=np.float32)
    obs = np.zeros(
        shape=[1, 9, cfg.lattice.vertical_num, cfg.lattice.horizon_num],
        dtype=np.float32,
    )
    depth_in = torch.from_numpy(depth).to(device)
    obs_in = torch.from_numpy(obs).to(device)

    print('TensorRT Transfer...')
    model_trt = torch2trt(policy, [depth_in, obs_in], fp16_mode=True)
    torch.save(model_trt.state_dict(), args.output)

    print('Evaluation...')
    traj_trt, score_trt = model_trt(depth_in, obs_in)
    traj, score = policy(depth_in, obs_in)
    torch.cuda.synchronize()

    torch_start = time.time()
    traj, score = policy(depth_in, obs_in)
    torch.cuda.synchronize()
    torch_end = time.time()

    trt_start = time.time()
    traj_trt, score_trt = model_trt(depth_in, obs_in)
    torch.cuda.synchronize()
    trt_end = time.time()

    traj_error = torch.mean(torch.abs(traj - traj_trt))
    score_error = torch.mean(torch.abs(score - score_trt))

    print(
        f'Torch Latency: {1000 * (torch_end - torch_start):.3f} ms, '
        f'TensorRT Latency: {1000 * (trt_end - trt_start):.3f} ms, '
        f'Transfer Trajectory Error: {traj_error.item():.6f}, '
        f'Transfer Score Error: {score_error.item():.6f}'
    )


def cmd_visualize(args: argparse.Namespace) -> None:
    from YOPO.policy import YOPODataset

    _load_config(args.yaml_config)
    dataset = YOPODataset()
    dataset._plot_sample_distribution()


def cmd_validate(args: argparse.Namespace) -> None:

    cfg = _load_config(args.yaml_config)
    manifest = cfg.resolve_dataset()

    print(f'Config OK ({len(cfg.model_dump())} fields)')
    print(f'  velocity    = {cfg.velocity} m/s')
    print(f'  traj_num    = {cfg.traj_num}')
    print(f'  goal_length = {cfg.goal_length:.1f} m')
    print(f'  sgm_time    = {cfg.sgm_time:.3f} s')
    print(f'  vel_scale   = {cfg.vel_scale:.2f}')
    print(
        f'  cost:      wg={cfg.cost_weights.wg:.2f} '
        f'ws={cfg.cost_weights.ws:.2f} '
        f'wa={cfg.cost_weights.wa:.2f} '
        f'wc={cfg.cost_weights.wc:.2f}'
    )
    print(
        f'  lattice:    {cfg.lattice.horizon_num}x{cfg.lattice.vertical_num}'
        f'  radio={cfg.lattice.radio_range:.1f}m'
        f'  radio_num={cfg.lattice.radio_num}'
    )
    print(f'  safety:    d0={cfg.safety.d0:.2f}m  r={cfg.safety.r:.2f}m')
    print()
    print(f'Dataset: {manifest.root}')
    print(f'  maps:       {manifest.num_maps} (.ply)')
    print(f'  image dirs: {len(manifest.image_dirs)}')
    print(f'  pose files: {len(manifest.poses)} (.csv)')

    for i in range(manifest.num_maps):
        ok = all(
            [
                manifest.maps[i].exists(),
                manifest.poses[i].exists() if i < len(manifest.poses) else False,
                manifest.image_dirs[i].exists() if i < len(manifest.image_dirs) else False,
            ]
        )
        status = 'OK' if ok else 'MISSING'
        print(
            f'  [{status}] map {i}: '
            f'ply={"Y" if manifest.maps[i].exists() else "N"} '
            f'csv={"Y" if (i < len(manifest.poses) and manifest.poses[i].exists()) else "N"} '
            f'imgs={"Y" if (i < len(manifest.image_dirs) and manifest.image_dirs[i].exists()) else "N"}'
        )


def _add_global_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        '--yaml-config',
        type=str,
        default=None,
        help='Path to YAML config override (default: builtin traj_opt.yaml)',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='YOPO training CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train', help='Train the YOPO network')
    _add_global_args(p_train)
    p_train.add_argument('--pretrained', type=int, default=0, help='use pre-trained model')
    p_train.add_argument('--trial', type=int, default=1, help='trial number of checkpoint')
    p_train.add_argument('--epoch', type=int, default=50, help='epoch number of checkpoint')
    p_train.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    p_train.add_argument('--batch-size', type=int, default=16)
    p_train.add_argument('--lr', type=float, default=1.5e-4, help='learning rate')
    p_train.add_argument('--wandb', action='store_true', help='log to Weights & Biases')
    p_train.add_argument('--wandb-project', type=str, default='yopo', help='wandb project name')
    p_train.add_argument('--wandb-name', type=str, default=None, help='wandb run name')

    p_trt = sub.add_parser('trt', help='Convert checkpoint to TensorRT')
    _add_global_args(p_trt)
    p_trt.add_argument('--trial', type=int, default=1)
    p_trt.add_argument('--epoch', type=int, default=50)
    p_trt.add_argument('--output', type=str, default='yopo_trt.pth', help='output file name')

    p_vis = sub.add_parser('visualize', help='Plot dataset sampling distributions')
    _add_global_args(p_vis)

    p_val = sub.add_parser('validate', help='Validate config and dataset artifacts')
    _add_global_args(p_val)

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'trt':
        cmd_trt(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    elif args.command == 'validate':
        cmd_validate(args)
