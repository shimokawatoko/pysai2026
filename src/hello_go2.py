"""Genesis の最小例: 地面に Go2 を置き、PD 制御で立たせて録画する（学習なし）。

    python src/hello_go2.py            # videos/hello_go2.mp4 を出力
    python src/hello_go2.py --no-pd    # 関節を制御しない（脱力して崩れる）

Genesis の基本の流れ:
  1. gs.init()            … 計算に使う装置（CPU / GPU）を決める
  2. gs.Scene()           … 物理世界を作る（時間刻み dt など）
  3. scene.add_entity()   … 物体（地面・ロボット）を置く
  4. scene.build()        … 物理エンジン用にコンパイルする（ここが少し重い）
  5. scene.step()         … 時間を dt だけ進める。これを繰り返す
"""

import argparse

import genesis as gs

# 立ったときの関節角度 [rad]（go2_train.py の default_joint_angles と同じ値）
STAND_POSE = {
    "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pd", action="store_true", help="関節を制御しない")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out", default="videos/hello_go2.mp4")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    dt = 0.02  # 50Hz（学習コードと同じ制御周期）
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=2), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0.0, 0.0, 0.42), quat=(1.0, 0.0, 0.0, 0.0))
    )
    cam = scene.add_camera(res=(640, 480), pos=(1.5, -1.5, 0.8), lookat=(0.0, 0.0, 0.25), fov=40, GUI=False)
    scene.build()

    # 関節名 → Genesis 内部の自由度番号
    joint_names = list(STAND_POSE.keys())
    dofs = [robot.get_joint(name).dofs_idx_local[0] for name in joint_names]
    print(f"Go2 の関節数: {len(dofs)}")

    if not args.no_pd:
        # PD 制御: トルク = kp * (目標角 - 現在角) - kd * 角速度
        # kp は「ばねの硬さ」、kd は「ダンパー（揺れを抑える）」。学習コードと同じ値
        robot.set_dofs_kp([20.0] * len(dofs), dofs)
        robot.set_dofs_kv([0.5] * len(dofs), dofs)
    robot.set_dofs_position([STAND_POSE[n] for n in joint_names], dofs)

    # 録画を始めると、以後 scene.step() のたびにカメラが自動で撮影する
    cam.start_recording(save_to_filename=args.out, fps=int(1 / dt))
    for _ in range(int(args.seconds / dt)):
        if not args.no_pd:
            robot.control_dofs_position([STAND_POSE[n] for n in joint_names], dofs)
        scene.step()
    cam.stop_recording()

    print(f"胴体の高さ（最後）: {robot.get_pos()[2].item():.3f} m")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
