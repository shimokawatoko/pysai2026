# Based on Genesis v1.4.5 examples/locomotion/go2_eval.py (Apache License 2.0, https://github.com/Genesis-Embodied-AI/Genesis)
# Changes from the original are described in README.md.
import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to load")
    # [変更] ビューアの代わりにヘッドレスで録画する（macOS ではビューアにメインスレッドの制約があるため）
    parser.add_argument("--seconds", type=float, default=10.0, help="録画する長さ [s]")
    parser.add_argument("--cmd-vx", type=float, default=None, help="前進速度の指令を固定する [m/s]")
    parser.add_argument("--friction", type=float, default=None, help="床と足の摩擦係数（例 0.1 でツルツル）")
    parser.add_argument("--out", type=str, default=None, help="出力 mp4（既定: ../videos/<exp>_ckpt<N>.mp4）")
    args = parser.parse_args()
    tag = f"_mu{args.friction:g}" if args.friction is not None else ""
    tag += f"_vx{args.cmd_vx:g}" if args.cmd_vx is not None else ""
    out = args.out or f"../videos/{args.exp_name}_ckpt{args.ckpt}{tag}.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    if args.friction is not None:
        env_cfg["friction"] = args.friction
    if args.cmd_vx is not None:
        command_cfg["lin_vel_x_range"] = [args.cmd_vx, args.cmd_vx]

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        add_camera=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    n_steps = int(args.seconds / env.dt)
    n_falls = 0
    env.cam.start_recording(save_to_filename=out, fps=int(1 / env.dt))
    with torch.no_grad():
        for _ in range(n_steps):
            actions = policy(obs_dict)  # 推論: 観測 → 12関節の目標角（の差分）
            obs_dict, rews, dones, infos = env.step(actions)
            n_falls += int(dones[0] and not infos["time_outs"][0])
            # カメラを犬の斜め後ろに付いていかせる
            x, y = env.base_pos[0, 0].item(), env.base_pos[0, 1].item()
            env.cam.set_pose(pos=(x + 1.5, y - 2.0, 0.9), lookat=(x, y, 0.3))
    env.cam.stop_recording()

    print(f"ckpt {args.ckpt}: {args.seconds:.0f}秒で 転倒(リセット) {n_falls} 回, 最後の x 位置 {env.base_pos[0, 0].item():.2f} m")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

"""
# evaluation
python go2_eval.py -e go2-walking --ckpt 100  # ../videos/go2-walking_ckpt100.mp4
"""
