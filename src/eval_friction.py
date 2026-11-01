"""床の摩擦を変えたときの転倒回数を比べる（録画なし）。摩擦ごとに1体ずつ、同時に動かす。

    python eval_friction.py -e go2-stand-v1 --ckpt 499
    python eval_friction.py -e go2-stand-slip --ckpt 799
"""

import argparse
import os
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env

FRICTIONS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    # 床は最小値に固定し、ロボット側の倍率で環境ごとの摩擦を決める（go2_env.py の friction_range と同じ仕組み）
    env_cfg.pop("friction", None)
    env_cfg["friction_range"] = [min(FRICTIONS), min(FRICTIONS)]

    env = Go2Env(len(FRICTIONS), env_cfg, obs_cfg, reward_cfg, command_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    del env.env_cfg["friction_range"]  # 以後リセットで引き直さない
    ratio = torch.tensor(FRICTIONS, dtype=gs.tc_float, device=gs.device)[:, None]
    env.robot.set_friction_ratio(ratio.expand(-1, env.robot.n_links))

    obs = env.reset()
    falls = torch.zeros(len(FRICTIONS), dtype=torch.int)
    with torch.no_grad():
        for _ in range(int(args.seconds / env.dt)):
            obs, _, dones, infos = env.step(policy(obs))
            falls += (dones & (infos["time_outs"] == 0)).cpu().int()

    print(f"{args.exp_name} ckpt {args.ckpt}: {args.seconds:.0f}秒あたりの転倒回数")
    for mu, n in zip(FRICTIONS, falls.tolist()):
        print(f"  摩擦 {mu:4.2f}: {n:3d} 回")


if __name__ == "__main__":
    main()
