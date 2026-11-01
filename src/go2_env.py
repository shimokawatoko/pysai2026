# Based on Genesis v1.4.5 examples/locomotion/go2_env.py (Apache License 2.0, https://github.com/Genesis-Embodied-AI/Genesis)
# Changes from the original are described in README.md.
import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, add_camera=False):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
                # can save memory. Violating this condition will raise an exception.
                max_collision_pairs=20,
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.plane = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        # [変更] 録画用のカメラ（推論時だけ使う。build の前に置く必要がある）
        self.cam = None
        if add_camera:
            self.cam = self.scene.add_camera(res=(1280, 720), pos=(2.0, -2.0, 1.0), lookat=(0.0, 0.0, 0.3), fov=40, GUI=False)

        self.scene.build(n_envs=num_envs)

        # [変更] 摩擦係数（ツルツルの床）。接触の摩擦は「2物体の大きい方」が使われるので、床とロボットの両方に設定する
        if "friction" in self.env_cfg:
            self.plane.set_friction(self.env_cfg["friction"])
            self.robot.set_friction(self.env_cfg["friction"])
        # [変更] 摩擦のランダム化（ドメインランダム化）。床は範囲の下限に固定し、ロボット側の倍率を環境ごとに
        # ランダムにする → 実際の摩擦 = max(床, ロボット×倍率) = ロボット×倍率 が範囲内で一様にばらつく
        elif "friction_range" in self.env_cfg:
            self.plane.set_friction(self.env_cfg["friction_range"][0])
            self.robot.set_friction(1.0)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        # [変更] 前足のすね（足先はすねの先 0.213 m。Genesis は足のリンクをすねに合体させている）
        self.front_calf_links_idx = [self.robot.get_link(name).idx_local for name in ("FL_calf", "FR_calf")]
        self.rear_calf_links_idx = [self.robot.get_link(name).idx_local for name in ("RL_calf", "RR_calf")]
        # [変更] 姿勢の矯正用: 後ろ足の hip / calf の関節番号（dof_pos は joint_names の順）
        self.rear_hip_dof_idx = torch.tensor(
            [self.env_cfg["joint_names"].index(n) for n in ("RL_hip_joint", "RR_hip_joint")],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.rear_calf_dof_idx = torch.tensor(
            [self.env_cfg["joint_names"].index(n) for n in ("RL_calf_joint", "RR_calf_joint")],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.rear_air_time = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.foot_offset = torch.tensor([0.0, 0.0, -0.213], dtype=gs.tc_float, device=gs.device)

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # initialize buffers
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.vel_ema = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.start_x = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.avg_v_x = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.support_side = torch.full((self.num_envs,), -1, dtype=gs.tc_int, device=gs.device)
        self.support_time = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.switched = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.swing_time = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = dict()  # extra information for logging

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # [変更] 世界座標の速度の移動平均。前後の揺れ（ロッキング）で瞬間速度を稼ぐのを防ぐ
        if "vel_filter_tau" in self.reward_cfg:
            alpha = math.exp(-self.dt / self.reward_cfg["vel_filter_tau"])
            world_vel = torch.cat((self.robot.get_vel()[:, :2], self.robot.get_ang()[:, 2:3]), dim=1)
            self.vel_ema = alpha * self.vel_ema + (1.0 - alpha) * world_vel

        # [変更] 片足支持の追跡（踏み替え報酬用）
        if "contact_height" in self.reward_cfg:
            contact = self._feet_z(self.rear_calf_links_idx) < self.reward_cfg["contact_height"]
            n_contact = contact.sum(dim=1)
            # 片足支持のときだけ「いまどちらの足か」を更新する。
            # 両足接地を挟んでも直前の支持足を覚えておく（踏み替えを取りこぼさない）
            side = torch.where(
                n_contact == 1, contact.to(dtype=gs.tc_int).argmax(dim=1), torch.full_like(n_contact, -1)
            )
            valid = side != -1
            prev = self.support_side
            switched = valid & (prev != -1) & (side != prev)
            self.switched = switched.to(dtype=gs.tc_float)
            self.support_time = torch.where(
                valid & (side == prev),
                self.support_time + self.dt,
                torch.where(valid, torch.zeros_like(self.support_time), self.support_time),
            )
            self.support_side = torch.where(valid, side, prev)
            # 足が浮いている時間（連続報酬用）
            self.swing_time = torch.where(contact, torch.zeros_like(self.swing_time), self.swing_time + self.dt)

        # [変更] リセットからの平均前進速度（往復のホップを打ち消す）
        elapsed = torch.clamp(self.episode_length_buf.to(dtype=gs.tc_float) * self.dt, min=0.5)
        self.avg_v_x = (self.base_pos[:, 0] - self.start_x) / elapsed

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # [変更] 終了条件を設定で選べるようにした（2足立ちでは前後10度の制限を外す）
        if "termination_if_pitch_greater_than" in self.env_cfg:
            self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        if "termination_if_roll_greater_than" in self.env_cfg:
            self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        # 胴体が直立に近いとオイラー角は不安定になるので、重力の向き（横成分）と高さで転倒を判定する
        if "termination_if_side_gravity_greater_than" in self.env_cfg:
            self.reset_buf |= torch.abs(self.projected_gravity[:, 1]) > self.env_cfg["termination_if_side_gravity_greater_than"]
        if "termination_if_base_height_below" in self.env_cfg:
            self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_base_height_below"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # Compute timeout
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        # reset state
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.vel_ema.zero_()
            self.start_x.zero_()
            self.avg_v_x.zero_()
            self.support_side.fill_(-1)
            self.support_time.zero_()
            self.switched.zero_()
            self.swing_time.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.vel_ema.masked_fill_(envs_idx[:, None], 0.0)
            self.start_x.masked_fill_(envs_idx, 0.0)
            self.avg_v_x.masked_fill_(envs_idx, 0.0)
            self.support_side.masked_fill_(envs_idx, -1)
            self.support_time.masked_fill_(envs_idx, 0.0)
            self.switched.masked_fill_(envs_idx, 0.0)
            self.swing_time.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        # random sample command upon reset
        self._resample_commands(envs_idx)

        if envs_idx is None:
            self.rear_air_time.zero_()
        else:
            self.rear_air_time.masked_fill_(envs_idx[:, None], 0.0)

        # [変更] リセットした環境ごとに床の滑りやすさを引き直す
        if "friction_range" in self.env_cfg and "friction" not in self.env_cfg:
            idx = torch.arange(self.num_envs, device=gs.device) if envs_idx is None else envs_idx.nonzero().squeeze(-1)
            if len(idx) > 0:
                lo, hi = self.env_cfg["friction_range"]
                ratio = lo + (hi - lo) * torch.rand((len(idx), 1), dtype=gs.tc_float, device=gs.device)
                self.robot.set_friction_ratio(ratio.expand(-1, self.robot.n_links), envs_idx=idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # Reward functions.
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    # ---- [変更] 2足立ち（後ろ足で立つ）用の報酬 ----
    def _reward_upright(self):
        # 胴体の起き具合。projected_gravity の x 成分は 水平=0、頭が真上=-1、頭が真下=+1
        # → (1 - g_x) / 2 は 頭が真下=0、水平=0.5、真上=1。二乗して 水平=0.25、真上=1
        return torch.square((1.0 - self.projected_gravity[:, 0]) / 2.0)

    def _reward_stand_height(self):
        # 胴体の高さ。lo 以下で 0、hi 以上で 1、その間は直線
        lo, hi = self.reward_cfg["stand_height_range"]
        return torch.clip((self.base_pos[:, 2] - lo) / (hi - lo), 0.0, 1.0)

    def _feet_z(self, calf_links_idx):
        # 足先の高さ（すねの位置 + すねの向きに沿って 0.213 m 先）。(n_envs, 2)
        pos = self.robot.get_links_pos(links_idx_local=calf_links_idx)
        quat = self.robot.get_links_quat(links_idx_local=calf_links_idx)
        return (pos + transform_by_quat(self.foot_offset.expand_as(pos), quat))[..., 2]

    def _reward_front_feet_air(self):
        # 前足の足先の高さ。地面=0、feet_air_target 以上で 1（2本の平均）
        foot_z = self._feet_z(self.front_calf_links_idx)
        return torch.clip(foot_z / self.reward_cfg["feet_air_target"], 0.0, 1.0).mean(dim=1)

    # ---- [変更] 姿勢の矯正用（ガニ股・膝の伸ばし切りをやめさせる）----
    def _feet_rel(self, calf_links_idx):
        # 足先の胴体座標系での位置 (n_envs, 2, 3)。x=前, y=左, z=上（胴体基準）
        pos = self.robot.get_links_pos(links_idx_local=calf_links_idx)
        quat = self.robot.get_links_quat(links_idx_local=calf_links_idx)
        foot = pos + transform_by_quat(self.foot_offset.expand_as(pos), quat)
        rel = foot - self.base_pos[:, None, :]
        return transform_by_quat(rel, inv_quat(self.base_quat)[:, None, :])

    def _reward_hip_neutral(self):
        # 股関節の開き（横方向）。0 に近いほど 1。既定は左右の平均、posture_worst_leg なら悪い方
        hip = self.dof_pos[:, self.rear_hip_dof_idx]
        if "hip_neutral_range" in self.reward_cfg:
            # 遠くても勾配が残る線形の山（限界付近でガウスが 0 になり学習が止まるのを防ぐ）
            rew = torch.clip(1.0 - hip.abs() / self.reward_cfg["hip_neutral_range"], 0.0, 1.0)
        else:
            rew = torch.exp(-torch.square(hip / self.reward_cfg["hip_neutral_sigma"]))
        rew = rew.min(dim=1).values if self.reward_cfg.get("posture_worst_leg") else rew.mean(dim=1)
        return rew * self._posture_gate()

    def _reward_knee_bend(self):
        # 膝を曲げたままにする。knee_target に近いほど 1。既定は左右の平均、posture_worst_leg なら悪い方
        calf = self.dof_pos[:, self.rear_calf_dof_idx]
        err = (calf - self.reward_cfg["knee_target"]) / self.reward_cfg["knee_sigma"]
        rew = torch.exp(-torch.square(err))
        rew = rew.min(dim=1).values if self.reward_cfg.get("posture_worst_leg") else rew.mean(dim=1)
        return rew * self._posture_gate()

    def _reward_feet_width(self):
        # 後ろ足の左右の開き。feet_width_target に近いほど 1。既定は左右の平均、posture_worst_leg なら悪い方
        y = self._feet_rel(self.rear_calf_links_idx)[..., 1].abs()
        if "feet_width_range" in self.reward_cfg:
            rew = torch.clip(1.0 - (y - self.reward_cfg["feet_width_target"]).abs() / self.reward_cfg["feet_width_range"], 0.0, 1.0)
        else:
            err = (y - self.reward_cfg["feet_width_target"]) / self.reward_cfg["feet_width_sigma"]
            rew = torch.exp(-torch.square(err))
        rew = rew.min(dim=1).values if self.reward_cfg.get("posture_worst_leg") else rew.mean(dim=1)
        return rew * self._posture_gate()

    def _reward_hip_sym(self):
        # 左右の hip が鏡像（hip_L + hip_R = 0）に近いほど 1
        hip = self.dof_pos[:, self.rear_hip_dof_idx]
        return torch.exp(-torch.square(hip.sum(dim=1) / self.reward_cfg["hip_sym_sigma"])) * self._posture_gate()

    def _reward_feet_sym(self):
        # 左右の足の横位置が対称（y_L + y_R = 0）に近いほど 1
        y = self._feet_rel(self.rear_calf_links_idx)[..., 1]
        return torch.exp(-torch.square(y.sum(dim=1) / self.reward_cfg["feet_sym_sigma"])) * self._posture_gate()

    def _posture_gate(self):
        # 直立しているときだけ姿勢報酬を与える（しゃがんで罰則を逃れるのを防ぐ）
        if not self.reward_cfg.get("posture_gate"):
            return 1.0
        upright = torch.square((1.0 - self.projected_gravity[:, 0]) / 2.0)
        return torch.clip(upright, 0.0, 1.0)

    def _reward_hip_penalty(self):
        # hip の開きの二乗罰則（限界付近でも勾配が残る）
        return torch.mean(torch.square(self.dof_pos[:, self.rear_hip_dof_idx]), dim=1)

    def _reward_feet_width_penalty(self):
        # 目標より広い足の間隔の二乗罰則
        y = self._feet_rel(self.rear_calf_links_idx)[..., 1].abs()
        return torch.mean(torch.square(torch.relu(y - self.reward_cfg["feet_width_target"])), dim=1)

    def _reward_knee_penalty(self):
        # 膝を伸ばし切る（calf が knee_min より上）ことへの二乗罰則
        calf = self.dof_pos[:, self.rear_calf_dof_idx]
        return torch.mean(torch.square(torch.relu(calf - self.reward_cfg["knee_min"])), dim=1)

    def _reward_level(self):
        # 胴体の横傾き（roll）。projected_gravity の y 成分が 0 に近いほど 1
        return torch.exp(-torch.square(self.projected_gravity[:, 1] / self.reward_cfg["level_sigma"])) * self._posture_gate()

    # ---- [変更] 2足歩行用の報酬 ----
    # 胴体が起きると胴体の x 軸は真上を向くので、速度・旋回は世界座標（地面基準）で測る
    def _reward_tracking_lin_vel_world(self):
        # 世界座標の水平速度が指令（x=前、y=横）に近いほど 1 に近い
        vel = self.robot.get_vel()[:, :2]
        err = torch.sum(torch.square(self.commands[:, :2] - vel), dim=1)
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel_world(self):
        # 世界座標の鉛直軸まわりの回転が指令に近いほど 1 に近い
        err = torch.square(self.commands[:, 2] - self.robot.get_ang()[:, 2])
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_single_support(self):
        # 後ろ足のうち、ちょうど1本だけが接地している（片足支持）
        contact = (self._feet_z(self.rear_calf_links_idx) < self.reward_cfg["contact_height"]).to(dtype=gs.tc_float)
        return (contact.sum(dim=1) == 1).to(dtype=gs.tc_float)

    def _reward_swing(self):
        # 上げている方の足の高さ（踏み出しの高さ）
        z = self._feet_z(self.rear_calf_links_idx)
        return torch.clip((z - self.reward_cfg["contact_height"]) / 0.05, 0.0, 1.0).max(dim=1).values

    def _reward_double_support_penalty(self):
        # 両足で立つことへの罰則（罰則逃れの「両足立ち」を防ぐ）
        contact = (self._feet_z(self.rear_calf_links_idx) < self.reward_cfg["contact_height"])
        return (contact.sum(dim=1) == 2).to(dtype=gs.tc_float)

    def _reward_swing_time(self):
        # 足が浮いている時間（連続報酬）。上げっぱなしは support_hold_penalty で抑える
        return torch.clip(self.swing_time / self.reward_cfg["swing_time_target"], 0.0, 1.0).sum(dim=1)

    def _reward_support_switch(self):
        # 支持足が入れ替わった瞬間に加点（その場足踏み）
        return self.switched

    def _reward_support_hold_penalty(self):
        # 同じ足で立ち続けることへの罰則
        return torch.relu(self.support_time - self.reward_cfg["support_hold_time"])

    def _reward_flight(self):
        # 後ろ足2本が同時に地面から離れている（ホップの滞空）。設定があれば前進速度でゲート
        z = self._feet_z(self.rear_calf_links_idx)
        rew = (z > self.reward_cfg["contact_height"]).all(dim=1).to(dtype=gs.tc_float)
        if "flight_gate_speed" in self.reward_cfg:
            v = self.avg_v_x if self.reward_cfg.get("flight_gate_avg") else self.vel_ema[:, 0]
            rew = rew * torch.clip(v / self.reward_cfg["flight_gate_speed"], 0.0, 1.0)
        return rew

    def _reward_progress_cum(self):
        # リセットからの平均前進速度（往復してその場に戻るホップは 0 点）
        cmd = torch.clamp(self.commands[:, 0], min=0.1)
        return torch.clip(self.avg_v_x / cmd, 0.0, 1.0)

    def _reward_backward(self):
        # 後退への罰則（前進指令なのに後ろへ進むのを止める）
        return torch.clip(-self.vel_ema[:, 0] / self.reward_cfg["backward_speed"], 0.0, 1.0)

    def _reward_progress(self):
        # 前進の進捗（指令速度に対する達成率）。立ち止まりは 0、指令どおりで 1。上限なしの頭打ち
        cmd = torch.clamp(self.commands[:, 0], min=0.1)
        return torch.clip(self.vel_ema[:, 0] / cmd, 0.0, 1.0)

    def _reward_tracking_lin_vel_ema(self):
        # 移動平均した世界座標の水平速度で追従を測る（前後ロッキング対策）
        err = torch.sum(torch.square(self.commands[:, :2] - self.vel_ema[:, :2]), dim=1)
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel_ema(self):
        err = torch.square(self.commands[:, 2] - self.vel_ema[:, 2])
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_feet_air_time(self):
        # 後ろ足が air_time_target 秒以上浮いてから着地した瞬間に点数（進む指令があるときだけ）
        contact = self._feet_z(self.rear_calf_links_idx) < self.reward_cfg["contact_height"]
        self.rear_air_time += self.dt
        first_contact = (self.rear_air_time > self.dt) & contact
        rew = torch.sum((self.rear_air_time - self.reward_cfg["air_time_target"]) * first_contact, dim=1)
        rew *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.rear_air_time *= ~contact
        return rew

