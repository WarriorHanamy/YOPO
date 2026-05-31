"""YOPO CLI entry point."""

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from YOPO.schema import DeployConfig, DockerImageSpec, RemoteTarget, YOPOConfig, config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def _apply_config_to_singleton(cfg: YOPOConfig) -> None:
    for field_name in cfg.model_fields:
        if field_name in cfg.model_computed_fields:
            continue
        setattr(config, field_name, getattr(cfg, field_name))


def cmd_train(args: argparse.Namespace) -> None:
    from YOPO.policy import YopoTrainer

    cfg = _load_config(args.yaml_config)
    if args.yaml_config:
        _apply_config_to_singleton(cfg)
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


def cmd_sweep(args: argparse.Namespace) -> None:
    from YOPO.policy import YopoTrainer

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        sys.exit(f'Config directory not found: {config_dir}')
    yaml_files = sorted(config_dir.glob('*.yaml'))
    if not yaml_files:
        sys.exit(f'No .yaml files found in {config_dir}')

    saved_base = Path(args.saved_dir).resolve()
    saved_base.mkdir(parents=True, exist_ok=True)
    print(f'Sweep: {len(yaml_files)} configs, output -> {saved_base}')

    for yf in yaml_files:
        stem = yf.stem
        out_dir = saved_base / stem
        if out_dir.exists():
            print(f'  [{stem}] SKIP (output directory already exists)')
            continue

        print(f'  [{stem}] Loading config...')
        cfg = YOPOConfig.from_yaml(yf)
        _apply_config_to_singleton(cfg)
        configure_random_seed(0)

        checkpoint_path = ''
        if args.pretrained:
            checkpoint_path = str(cfg.resolve_checkpoint_path(args.trial, args.epoch))

        trainer = YopoTrainer(
            learning_rate=args.lr,
            batch_size=args.batch_size,
            loss_weight=[1.0, 1.0],
            tensorboard_path=str(saved_base),
            checkpoint_path=checkpoint_path,
            save_on_exit=True,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name or stem,
            saved_subdir=stem,
        )
        trainer.train(epoch=args.epochs)

        shutil.copy2(yf, out_dir / 'config.yaml')
        print(f'  [{stem}] Complete -> {out_dir}')

    print(f'Sweep complete ({len(yaml_files)} configs)')


def cmd_data_gen(args: argparse.Namespace) -> None:
    image_tag = 'yopo-data-gen:latest'
    data_gen_dir = _PROJECT_ROOT / 'docker' / 'data-gen'
    output_dir = Path(args.output_dir).resolve()

    if not args.skip_build:
        print(f'Building Docker image {image_tag}...')
        subprocess.run(
            ['docker', 'build', '-t', image_tag, str(data_gen_dir)],
            check=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Generating dataset -> {output_dir}...')
    subprocess.run(
        [
            'docker',
            'run',
            '--gpus',
            'all',
            '--rm',
            '-v',
            f'{output_dir}:/output',
            image_tag,
        ],
        check=True,
    )
    print(f'Dataset saved to {output_dir / "data"}')


def cmd_docker_build(args: argparse.Namespace) -> None:
    tag = args.tag or 'yopo-train:latest'
    dockerfile = _PROJECT_ROOT / 'docker' / 'train' / 'Dockerfile'
    print(f'Building {tag}...')
    subprocess.run(
        ['docker', 'build', '-f', str(dockerfile), '-t', tag, str(_PROJECT_ROOT)],
        check=True,
    )
    print(f'Image {tag} built successfully')


def cmd_docker_export(args: argparse.Namespace) -> None:
    tag = args.tag or 'yopo-train:latest'
    output = args.output or 'yopo-train.tar'
    print(f'Exporting {tag} -> {output}...')
    subprocess.run(
        ['docker', 'save', '-o', output, tag],
        check=True,
    )
    size_mb = Path(output).stat().st_size / (1024 * 1024)
    print(f'Exported {output} ({size_mb:.1f} MB)')


def cmd_docker_send(args: argparse.Namespace) -> None:
    tar_path = Path(args.tar)
    if not tar_path.exists():
        sys.exit(f'Tar file not found: {tar_path}')

    remote = f'{args.host}:{args.remote_path}'
    print(f'Sending {tar_path} -> {remote}...')
    scp_cmd = ['scp']
    if args.port:
        scp_cmd += ['-P', str(args.port)]
    scp_cmd += [str(tar_path), remote]
    subprocess.run(scp_cmd, check=True)

    if args.load:
        tag = args.load_tag or 'yopo-train:latest'
        remote_file = f'{args.remote_path.rstrip("/")}/{tar_path.name}'
        print(f'Loading image on {args.host}...')
        ssh_cmd = ['ssh']
        if args.port:
            ssh_cmd += ['-p', str(args.port)]
        ssh_cmd += [args.host, f'docker load -i {remote_file}']
        subprocess.run(ssh_cmd, check=True)
        print(f'Image {tag} loaded on {args.host}')


# ---------------------------------------------------------------------------
# deploy helpers
# ---------------------------------------------------------------------------


def _ssh_command(remote: RemoteTarget, cmd: str) -> list[str]:
    ssh = ['ssh']
    if remote.port != 22:
        ssh += ['-p', str(remote.port)]
    ssh += [remote.host, cmd]
    return ssh


def _scp_to_remote(remote: RemoteTarget, src_paths: list[str], dst: str) -> list[str]:
    scp = ['scp']
    if remote.port != 22:
        scp += ['-P', str(remote.port)]
    return scp + src_paths + [f'{remote.host}:{dst}']


def _scp_from_remote(remote: RemoteTarget, src: str, dst: str) -> list[str]:
    scp = ['scp', '-r']
    if remote.port != 22:
        scp += ['-P', str(remote.port)]
    return scp + [f'{remote.host}:{src}', dst]


# ---------------------------------------------------------------------------
# deploy phases
# ---------------------------------------------------------------------------


def _phase_build_image(image: DockerImageSpec) -> None:
    tag = image.tag
    dockerfile = _PROJECT_ROOT / 'docker' / 'train' / 'Dockerfile'
    print(f'[Phase 1/4] Build image: {tag}')
    subprocess.run(
        ['docker', 'build', '-f', str(dockerfile), '-t', tag, str(_PROJECT_ROOT)],
        check=True,
    )


def _phase_send_image(remote: RemoteTarget, image: DockerImageSpec) -> None:
    tar = image.tar_path
    if not tar.exists():
        print(f'[Phase 2/4] Export image -> {tar}')
        subprocess.run(['docker', 'save', '-o', str(tar), image.tag], check=True)

    print(f'[Phase 2/4] Send image to {remote.host}')
    subprocess.run(_scp_to_remote(remote, [str(tar)], '/tmp/'), check=True)

    remote_tar = f'/tmp/{tar.name}'
    print('[Phase 2/4] Load image on remote')
    subprocess.run(_ssh_command(remote, f'docker load -i {remote_tar}'), check=True)


def _phase_send_configs(remote: RemoteTarget, sweep_config_dir: Path) -> None:
    yaml_files = sorted(sweep_config_dir.glob('*.yaml'))
    if not yaml_files:
        sys.exit(f'No .yaml files found in {sweep_config_dir}')
    print(f'[Phase 3/4] Send {len(yaml_files)} configs -> {remote.host}:{remote.configs_path}')
    subprocess.run(
        _ssh_command(remote, f'mkdir -p {remote.configs_path}'),
        check=True,
    )
    subprocess.run(
        _scp_to_remote(remote, [str(f) for f in yaml_files], f'{remote.configs_path}/'),
        check=True,
    )


def _phase_launch_training(cfg: DeployConfig) -> str:
    print(f'[Phase 4/4] Launch training on {cfg.remote.host}')
    docker_cmd = (
        f'docker run -d --gpus all '
        f'-v {cfg.remote.dataset_path}:/app/dataset/data '
        f'-v {cfg.remote.saved_path}:/app/YOPO/saved '
        f'-v {cfg.remote.configs_path}:/app/configs '
        f'{cfg.image.tag} '
        f'sweep --config-dir /app/configs --epochs {cfg.epochs} '
        f'--batch-size {cfg.batch_size} --lr {cfg.lr}'
    )
    result = subprocess.run(
        _ssh_command(cfg.remote, docker_cmd),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# deploy commands
# ---------------------------------------------------------------------------


def cmd_deploy_run(args: argparse.Namespace) -> None:
    from YOPO.schema import DeployConfig

    cfg = DeployConfig.from_yaml(args.config)

    if not cfg.skip_build:
        _phase_build_image(cfg.image)
    else:
        print('[Phase 1/4] Build: SKIP (skip_build=true)')

    if not cfg.skip_image_send:
        _phase_send_image(cfg.remote, cfg.image)
    else:
        print('[Phase 2/4] Send image: SKIP (skip_image_send=true)')

    if not cfg.skip_configs_send:
        _phase_send_configs(cfg.remote, cfg.sweep_config_dir)
    else:
        print('[Phase 3/4] Send configs: SKIP (skip_configs_send=true)')

    container_id = _phase_launch_training(cfg)

    print()
    print('Pipeline launched. Training is running asynchronously on remote.')
    print(f'  Container: {container_id}')
    print(f'  Check:     yopo deploy status --config {args.config}')
    print(f'  Gather:    yopo deploy gather --config {args.config}')
    print(f'  Logs:      ssh {cfg.remote.host} "docker logs -f {container_id}"')


def cmd_deploy_status(args: argparse.Namespace) -> None:
    from YOPO.schema import DeployConfig

    cfg = DeployConfig.from_yaml(args.config)
    tag = cfg.image.tag

    result = subprocess.run(
        _ssh_command(
            cfg.remote,
            f'docker ps -a --filter ancestor={tag} '
            "--format '{{.ID}} {{.Status}} {{.Names}}'",
        ),
        capture_output=True,
        text=True,
    )

    lines = [line for line in result.stdout.strip().split('\n') if line]
    if not lines:
        print(f'No containers found for image {tag}')
        return

    for line in lines:
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        cid = parts[0]
        status = parts[1] if len(parts) > 1 else '?'
        name = parts[2] if len(parts) > 2 else '-'

        icon = '●' if 'Up' in status else '○'
        print(f'{icon} {cid[:12]}  {name}  {status}')

        log_result = subprocess.run(
            _ssh_command(cfg.remote, f'docker logs --tail 15 {cid}'),
            capture_output=True,
            text=True,
        )
        tail = log_result.stdout.strip()
        if tail:
            print(f'  {tail[-200:]}')
        print()


def cmd_deploy_gather(args: argparse.Namespace) -> None:
    from YOPO.schema import DeployConfig

    cfg = DeployConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        _ssh_command(
            cfg.remote,
            f'docker ps --filter ancestor={cfg.image.tag} -q',
        ),
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print('WARNING: Training container(s) still running. Gathering partial results.')

    print(f'Gathering {cfg.remote.host}:{cfg.remote.saved_path} -> {output_dir}')
    subprocess.run(
        _scp_from_remote(cfg.remote, cfg.remote.saved_path + '/', str(output_dir)),
        check=True,
    )

    if args.clean:
        print(f'Cleaning remote: {cfg.remote.saved_path}')
        subprocess.run(
            _ssh_command(cfg.remote, f'rm -rf {cfg.remote.saved_path}'),
            check=True,
        )

    print(f'Results in {output_dir}')


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
    if args.yaml_config:
        _apply_config_to_singleton(cfg)
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

    if args.yaml_config:
        _apply_config_to_singleton(_load_config(args.yaml_config))
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


def _add_train_args(subparser: argparse.ArgumentParser) -> None:
    _add_global_args(subparser)
    subparser.add_argument('--pretrained', type=int, default=0, help='use pre-trained model')
    subparser.add_argument('--trial', type=int, default=1, help='trial number of checkpoint')
    subparser.add_argument('--epoch', type=int, default=50, help='epoch number of checkpoint')
    subparser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    subparser.add_argument('--batch-size', type=int, default=16)
    subparser.add_argument('--lr', type=float, default=1.5e-4, help='learning rate')
    subparser.add_argument('--wandb', action='store_true', help='log to Weights & Biases')
    subparser.add_argument('--wandb-project', type=str, default='yopo', help='wandb project name')
    subparser.add_argument('--wandb-name', type=str, default=None, help='wandb run name')


def main() -> None:
    parser = argparse.ArgumentParser(description='YOPO training CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    # ---- train ----
    p_train = sub.add_parser('train', help='Train the YOPO network')
    _add_train_args(p_train)

    # ---- sweep ----
    p_sweep = sub.add_parser('sweep', help='Run training over multiple YAML configs')
    p_sweep.add_argument(
        '--config-dir', type=str, required=True, help='Directory with YAML config files'
    )
    p_sweep.add_argument('--pretrained', type=int, default=0)
    p_sweep.add_argument('--trial', type=int, default=1)
    p_sweep.add_argument('--epoch', type=int, default=50)
    p_sweep.add_argument('--epochs', type=int, default=50)
    p_sweep.add_argument('--batch-size', type=int, default=16)
    p_sweep.add_argument('--lr', type=float, default=1.5e-4)
    p_sweep.add_argument(
        '--saved-dir', type=str, default='YOPO/saved', help='Base output directory'
    )
    p_sweep.add_argument('--wandb', action='store_true')
    p_sweep.add_argument('--wandb-project', type=str, default='yopo')
    p_sweep.add_argument('--wandb-name', type=str, default=None)

    # ---- data-gen ----
    p_data = sub.add_parser('data-gen', help='Generate training dataset via Docker')
    p_data.add_argument(
        '--output-dir',
        type=str,
        default=str(_PROJECT_ROOT / 'dataset'),
        help='Host directory to mount (default: project_root/dataset)',
    )
    p_data.add_argument('--skip-build', action='store_true', help='Skip Docker image build')

    # ---- docker ----
    p_docker = sub.add_parser('docker', help='Docker image management')
    docker_sub = p_docker.add_subparsers(dest='docker_cmd', required=True)

    p_dk_build = docker_sub.add_parser('build', help='Build yopo-train Docker image')
    p_dk_build.add_argument(
        '--tag', type=str, default=None, help='Image tag (default: yopo-train:latest)'
    )

    p_dk_export = docker_sub.add_parser('export', help='Export image to tar')
    p_dk_export.add_argument('--tag', type=str, default=None)
    p_dk_export.add_argument('-o', '--output', type=str, default='yopo-train.tar')

    p_dk_send = docker_sub.add_parser('send', help='SCP tar to remote and optionally docker load')
    p_dk_send.add_argument('host', type=str, help='Remote host (user@ip)')
    p_dk_send.add_argument('--port', type=int, default=22)
    p_dk_send.add_argument('--tar', type=str, default='yopo-train.tar')
    p_dk_send.add_argument('--remote-path', type=str, default='/tmp/')
    p_dk_send.add_argument('--load', action='store_true', help='docker load on remote after SCP')
    p_dk_send.add_argument('--load-tag', type=str, default=None)

    # ---- deploy ----
    p_deploy = sub.add_parser('deploy', help='End-to-end deployment pipeline')
    deploy_sub = p_deploy.add_subparsers(dest='deploy_cmd', required=True)

    p_dep_run = deploy_sub.add_parser('run', help='Build, send, and launch training')
    p_dep_run.add_argument('--config', type=str, required=True, help='Path to deploy.yaml')

    p_dep_status = deploy_sub.add_parser('status', help='Check remote training status')
    p_dep_status.add_argument('--config', type=str, required=True, help='Path to deploy.yaml')

    p_dep_gather = deploy_sub.add_parser('gather', help='Pull results from remote')
    p_dep_gather.add_argument('--config', type=str, required=True, help='Path to deploy.yaml')
    p_dep_gather.add_argument(
        '--output-dir', type=str, default='results', help='Local output directory'
    )
    p_dep_gather.add_argument(
        '--clean', action='store_true', help='Remove remote saved path after gathering'
    )

    # ---- trt ----
    p_trt = sub.add_parser('trt', help='Convert checkpoint to TensorRT')
    _add_global_args(p_trt)
    p_trt.add_argument('--trial', type=int, default=1)
    p_trt.add_argument('--epoch', type=int, default=50)
    p_trt.add_argument('--output', type=str, default='yopo_trt.pth', help='output file name')

    # ---- visualize ----
    p_vis = sub.add_parser('visualize', help='Plot dataset sampling distributions')
    _add_global_args(p_vis)

    # ---- validate ----
    p_val = sub.add_parser('validate', help='Validate config and dataset artifacts')
    _add_global_args(p_val)

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'sweep':
        cmd_sweep(args)
    elif args.command == 'data-gen':
        cmd_data_gen(args)
    elif args.command == 'docker':
        if args.docker_cmd == 'build':
            cmd_docker_build(args)
        elif args.docker_cmd == 'export':
            cmd_docker_export(args)
        elif args.docker_cmd == 'send':
            cmd_docker_send(args)
    elif args.command == 'deploy':
        if args.deploy_cmd == 'run':
            cmd_deploy_run(args)
        elif args.deploy_cmd == 'status':
            cmd_deploy_status(args)
        elif args.deploy_cmd == 'gather':
            cmd_deploy_gather(args)
    elif args.command == 'trt':
        cmd_trt(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    elif args.command == 'validate':
        cmd_validate(args)
