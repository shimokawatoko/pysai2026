# Based on Genesis v1.4.5 examples/locomotion/go2_train.py (Apache License 2.0, https://github.com/Genesis-Embodied-AI/Genesis)
# Changes from the original are described in README.md.
import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env


def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_stand_cfgs():
    """[変更] 2足立ち（後ろ足で立ってバランスを保つ）。公式の設定から報酬と終了条件だけを変える。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # 終了条件: 前後・左右10度で終了をやめ、「倒れた（胴体が低い）」「横に大きく傾いた」で終了
    del env_cfg["termination_if_pitch_greater_than"]
    del env_cfg["termination_if_roll_greater_than"]
    env_cfg["termination_if_base_height_below"] = 0.15  # [m]
    env_cfg["termination_if_side_gravity_greater_than"] = 0.7  # 横に約45度

    reward_cfg["stand_height_range"] = [0.30, 0.50]  # [m] 四つ足の高さ → 立ったときの高さ
    reward_cfg["feet_air_target"] = 0.10  # [m]
    reward_cfg["reward_scales"] = {
        "upright": 1.0,  # 胴体を起こす
        "stand_height": 1.0,  # 胴体を高く
        "front_feet_air": 0.5,  # 前足を浮かせる
        "tracking_lin_vel": 0.5,  # 指令（速度0）＝その場にとどまる
        "tracking_ang_vel": 0.2,  # 回らない
        "lin_vel_z": -0.2,  # 跳ねない（立ち上がる動きを妨げないよう公式の -1.0 から弱めた）
        "action_rate": -0.01,  # なめらかに
    }

    # 指令は「止まっている」（前後・横・旋回すべて 0）
    command_cfg["lin_vel_x_range"] = [0.0, 0.0]
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_stand_slip_cfgs():
    """[変更] 2足立ち＋床の摩擦のランダム化（0.05＝氷〜1.0＝普通の床）。報酬は2足立ちと同じ。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_cfgs()
    env_cfg["friction_range"] = [0.05, 1.0]
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_biped_cfgs():
    """[変更] 2足歩行 v1。2足立ちの報酬に「世界座標での前進速度の追従」を足す（歩き方は教えない）。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_cfgs()
    reward_cfg["contact_height"] = 0.03  # [m] 足先がこれより低ければ接地（feet_air_time 用）
    reward_cfg["air_time_target"] = 0.25  # [s]
    reward_cfg["reward_scales"] = {
        "upright": 1.0,
        "stand_height": 0.5,  # 歩くときは多少低くてもよいので弱める
        "front_feet_air": 0.5,
        "tracking_lin_vel_world": 1.5,  # 主役。立っているだけの点数より大きくする
        "tracking_ang_vel_world": 0.3,  # まっすぐ進む
        "lin_vel_z": -0.2,
        "action_rate": -0.01,
    }
    # 前進速度の指令を 0〜0.5 m/s でランダムに（4秒ごとに引き直し）。0 なら立ち止まる
    command_cfg["lin_vel_x_range"] = [0.0, 0.5]
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_stand_natural_cfgs():
    """[変更] 2足立ち v3。ガニ股・膝の伸ばし切り（関節限界に寄りかかる姿勢）を矯正する。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_cfgs()

    reward_cfg["stand_height_range"] = [0.28, 0.42]  # 高さへの圧力を弱める（膝を伸ばし切らせない）
    reward_cfg["hip_neutral_sigma"] = 0.20  # [rad] 約11度。hip が開くほど減点
    reward_cfg["knee_target"] = -1.40  # [rad] 約 -80度（軽く曲げた姿勢）
    reward_cfg["knee_sigma"] = 0.35  # [rad]
    reward_cfg["feet_width_target"] = 0.15  # [m] 左右それぞれの足の横位置（腰幅）
    reward_cfg["feet_width_sigma"] = 0.06  # [m]
    reward_cfg["reward_scales"] = {
        "upright": 1.0,
        "stand_height": 0.3,  # 1.0 → 0.3
        "front_feet_air": 0.5,
        "hip_neutral": 1.0,  # 足を開かない
        "knee_bend": 0.5,  # 膝を曲げる
        "feet_width": 0.5,  # 足を腰幅に
        "tracking_lin_vel": 0.5,
        "tracking_ang_vel": 0.2,
        "lin_vel_z": -0.2,
        "action_rate": -0.01,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_stand_natural2_cfgs():
    """[変更] 2足立ち v4。v3 の「片脚だけ広げて平均点を稼ぐ」を潰す（悪い方の脚で採点＋横傾き）。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_natural_cfgs()

    reward_cfg["posture_worst_leg"] = True  # 左右の平均ではなく、悪い方の脚で採点
    reward_cfg["hip_neutral_sigma"] = 0.18  # [rad] 少し厳しく
    reward_cfg["level_sigma"] = 0.10  # 横傾き 0 に近いほど加点
    reward_cfg["reward_scales"]["hip_neutral"] = 1.5
    reward_cfg["reward_scales"]["knee_bend"] = 0.8
    reward_cfg["reward_scales"]["feet_width"] = 0.8
    reward_cfg["reward_scales"]["level"] = 0.5
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_stand_natural3_cfgs():
    """[変更] 2足立ち v5。左右対称を明示的に報酬化し、腰幅の自然な姿勢へ寄せる。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_natural2_cfgs()

    reward_cfg["hip_neutral_sigma"] = 0.15  # [rad]
    reward_cfg["feet_width_target"] = 0.15  # [m]
    reward_cfg["feet_width_sigma"] = 0.06  # [m]
    reward_cfg["hip_sym_sigma"] = 0.10  # [rad] 左右の hip の和を 0 に
    reward_cfg["feet_sym_sigma"] = 0.04  # [m] 左右の足の横位置の和を 0 に
    reward_cfg["reward_scales"]["hip_neutral"] = 2.0
    reward_cfg["reward_scales"]["feet_width"] = 1.0
    reward_cfg["reward_scales"]["hip_sym"] = 2.0
    reward_cfg["reward_scales"]["feet_sym"] = 1.0
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_stand_natural4_cfgs():
    """[変更] 2足立ち v6。hip/足幅の報酬を線形の山にして、遠くても勾配が残るようにする。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_natural3_cfgs()

    reward_cfg["hip_neutral_range"] = 0.50  # [rad] 28.6度。ここより内側なら線形に加点
    reward_cfg["feet_width_range"] = 0.20  # [m] 目標 0.15 m から 0.20 m 以内なら加点
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_stand_natural5_cfgs():
    """[変更] 2足立ち v6。遠くても勾配が残る二次罰則で、ガニ股と膝の伸ばし切りを潰す。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_natural4_cfgs()

    reward_cfg["knee_min"] = -1.20  # [rad] これより伸びていたら罰則（約 -69度）
    reward_cfg["reward_scales"]["hip_penalty"] = -5.0
    reward_cfg["reward_scales"]["feet_width_penalty"] = -40.0
    reward_cfg["reward_scales"]["knee_penalty"] = -5.0
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_stand_natural6_cfgs():
    """[変更] 2足立ち v7。姿勢報酬を直立度でゲートし、しゃがみ逃げを防ぐ。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_natural5_cfgs()

    reward_cfg["posture_gate"] = True
    reward_cfg["reward_scales"]["upright"] = 5.0  # 直立を最優先
    reward_cfg["reward_scales"]["front_feet_air"] = 1.0
    reward_cfg["reward_scales"]["hip_penalty"] = -2.0
    reward_cfg["reward_scales"]["feet_width_penalty"] = -20.0
    reward_cfg["reward_scales"]["knee_penalty"] = -3.0
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_biped3_cfgs():
    """[変更] 2足歩行 v8。v7 の自然な立ち姿勢から、移動平均した速度追従で歩かせる。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_natural6_cfgs()

    reward_cfg["vel_filter_tau"] = 0.5  # [s] 速度の移動平均
    reward_cfg["tracking_sigma"] = 0.05
    reward_cfg["contact_height"] = 0.03  # [m]
    reward_cfg["air_time_target"] = 0.15  # [s]
    # 立ち止まり報酬（速度0への追従）を外し、移動平均の速度追従を主役にする
    scales = dict(reward_cfg["reward_scales"])
    scales.pop("tracking_lin_vel")
    scales.pop("tracking_ang_vel")
    scales["tracking_lin_vel_ema"] = 3.0
    scales["tracking_ang_vel_ema"] = 0.5
    scales["feet_air_time"] = 1.0
    reward_cfg["reward_scales"] = scales

    command_cfg["lin_vel_x_range"] = [0.2, 0.4]
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_biped4_cfgs():
    """[変更] 2足歩行 v9。立ち止まりの取り分を 0 にする「前進進捗」報酬を主役にする。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_biped3_cfgs()

    scales = dict(reward_cfg["reward_scales"])
    scales["tracking_lin_vel_ema"] = 1.0  # 横方向のズレだけ弱く見る
    scales["progress"] = 6.0  # 前進の達成率（立ち止まり 0 → 6.0）
    # 立ち止まりが得にならないよう、姿勢報酬は控えめにする
    scales.update(
        {
            "hip_neutral": 0.5,
            "knee_bend": 0.3,
            "feet_width": 0.3,
            "hip_sym": 0.5,
            "feet_sym": 0.3,
            "level": 0.3,
            "front_feet_air": 0.5,
        }
    )
    reward_cfg["reward_scales"] = scales
    reward_cfg["air_time_target"] = 0.10  # [s] 短い一歩でも拾う
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_biped5_cfgs():
    """[変更] 2足歩行 v10。片脚バランスを避け、両足で跳ぶホッピングで前進する。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_biped4_cfgs()

    scales = dict(reward_cfg["reward_scales"])
    scales["progress"] = 6.0
    scales["flight"] = 4.0  # 両足が浮いている時間
    scales["feet_air_time"] = 1.0
    reward_cfg["reward_scales"] = scales
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_biped6_cfgs():
    """[変更] 2足歩行 v11。ホップを前進と結びつける（滞空報酬を前進速度でゲート、後退に罰則）。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_biped5_cfgs()

    reward_cfg["flight_gate_speed"] = 0.15  # [m/s] この速度までは滞空報酬を比例配分
    reward_cfg["backward_speed"] = 0.20  # [m/s] この速度で後退罰則が満点
    scales = dict(reward_cfg["reward_scales"])
    scales["progress"] = 8.0
    scales["backward"] = -3.0
    reward_cfg["reward_scales"] = scales
    command_cfg["lin_vel_x_range"] = [0.3, 0.5]
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_biped7_cfgs():
    """[変更] 2足歩行 v12。往復ホップを打ち消すため、リセットからの平均前進速度で評価する。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_biped6_cfgs()

    reward_cfg["flight_gate_avg"] = True  # 滞空報酬も平均前進速度でゲート
    scales = dict(reward_cfg["reward_scales"])
    scales["progress"] = 0.0  # 瞬間の進捗は使わない
    scales["progress_cum"] = 8.0  # 累積の前進距離 / 経過時間
    scales["backward"] = -2.0
    reward_cfg["reward_scales"] = scales
    reward_cfg["reward_scales"]["progress_cum"] = 8.0
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_step_cfgs():
    """[変更] その場足踏み（v13）。片足支持と足の振り上げを学ばせる（前進指令はまだ無し）。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_stand_natural6_cfgs()

    reward_cfg["contact_height"] = 0.03  # [m]
    reward_cfg["air_time_target"] = 0.10  # [s]
    scales = dict(reward_cfg["reward_scales"])
    scales.update({"single_support": 3.0, "swing": 2.0, "feet_air_time": 1.0})
    reward_cfg["reward_scales"] = scales
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_step2_cfgs():
    """[変更] その場足踏み v14。片足立ちのまま固まらず、接地足を交互に入れ替える。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_step_cfgs()

    reward_cfg["support_hold_time"] = 0.6  # [s] 同じ足でこれ以上立ち続けたら罰則
    scales = dict(reward_cfg["reward_scales"])
    scales.update(
        {
            "single_support": 1.0,
            "swing": 0.5,
            "feet_air_time": 2.0,
            "support_switch": 2.0,
            "support_hold_penalty": -2.0,
        }
    )
    reward_cfg["reward_scales"] = scales
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_step3_cfgs():
    """[変更] その場足踏み v15。踏み替え検出のバグを直した版。"""
    return get_step2_cfgs()


def get_step4_cfgs():
    """[変更] その場足踏み v16。両足接地にも罰則を入れ、踏み替えを強く評価する。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_step3_cfgs()

    reward_cfg["support_hold_time"] = 0.40  # [s]
    reward_cfg["swing_time_target"] = 0.15  # [s]
    scales = dict(reward_cfg["reward_scales"])
    scales.update(
        {
            "single_support": 2.0,
            "swing": 0.5,
            "feet_air_time": 1.0,
            "support_switch": 5.0,
            "support_hold_penalty": -3.0,
            "double_support_penalty": -2.0,
            "swing_time": 1.0,
        }
    )
    reward_cfg["reward_scales"] = scales
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_biped2_cfgs():
    """[変更] 2足歩行 v2。v1 の「立ち止まっても前進の報酬が 75% もらえる」を潰す。"""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_biped_cfgs()
    reward_cfg["tracking_sigma"] = 0.05  # 速度のずれに厳しく（立ち止まりの点 0.75 → 0.06）
    reward_cfg["reward_scales"]["feet_air_time"] = 1.0  # 後ろ足を上げて踏み出したら加点
    command_cfg["lin_vel_x_range"] = [0.3, 0.5]  # 必ず進む指令（0 を引かない）
    return env_cfg, obs_cfg, reward_cfg, command_cfg


TASKS = {
    "walk": get_cfgs,
    "stand": get_stand_cfgs,
    "stand-slip": get_stand_slip_cfgs,
    "stand-natural": get_stand_natural_cfgs,
    "stand-natural2": get_stand_natural2_cfgs,
    "stand-natural3": get_stand_natural3_cfgs,
    "stand-natural4": get_stand_natural4_cfgs,
    "stand-natural5": get_stand_natural5_cfgs,
    "stand-natural6": get_stand_natural6_cfgs,
    "biped3": get_biped3_cfgs,
    "biped4": get_biped4_cfgs,
    "biped5": get_biped5_cfgs,
    "biped6": get_biped6_cfgs,
    "biped7": get_biped7_cfgs,
    "step": get_step_cfgs,
    "step2": get_step2_cfgs,
    "step3": get_step3_cfgs,
    "step4": get_step4_cfgs,
    "biped": get_biped_cfgs,
    "biped2": get_biped2_cfgs,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--max-iterations", type=int, default=101, help="Number of learning iterations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="続きから学習するモデル（例 logs/go2-stand-v1/model_499.pt）")
    parser.add_argument("--task", choices=TASKS, default="walk", help="walk=公式の前進歩行 / stand=2足立ち")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = TASKS[args.task]()
    train_cfg = get_train_cfg(args.exp_name)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # Training throughput comes from the 4096 parallel environments, which need a GPU.
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume:
        runner.load(args.resume)  # [変更] 学習済みの方策から続けて学習（ファインチューニング）

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
