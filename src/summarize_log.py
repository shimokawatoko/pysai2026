"""学習ログ（TensorBoard）から、報酬の推移を表で出す。

    python summarize_log.py logs/go2-stand-v1            # 表を表示
    python summarize_log.py logs/go2-stand-v1 --plot     # 学習曲線を logs/<exp>/curves.png に保存
"""

import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(log_dir):
    ea = EventAccumulator(glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))[0], size_guidance={"scalars": 0})
    ea.Reload()
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)] for tag in ea.Tags()["scalars"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    data = load(args.log_dir)
    keys = ["Train/mean_reward", "Train/mean_episode_length"] + sorted(k for k in data if k.startswith("Episode/rew_"))
    keys = [k for k in keys if k in data]
    short = [k.split("/")[-1].replace("rew_", "").replace("mean_", "") for k in keys]
    print("iter  " + "  ".join(f"{s[:12]:>12}" for s in short))
    steps = [s for s, _ in data[keys[0]]]
    for i, step in enumerate(steps):
        if step % args.every == 0 or i == len(steps) - 1:
            print(f"{step:4d}  " + "  ".join(f"{dict(data[k]).get(step, float('nan')):12.4f}" for k in keys))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for k, ax in zip(["Train/mean_reward", "Train/mean_episode_length"], axes):
            s, v = zip(*data[k])
            ax.plot(s, v)
            ax.set_title(k.split("/")[-1])
            ax.set_xlabel("iteration")
        fig.tight_layout()
        out = os.path.join(args.log_dir, "curves.png")
        fig.savefig(out, dpi=120)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
