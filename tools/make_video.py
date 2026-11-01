"""提出用の動画を組み立てる（字幕は Pillow で画像にして ffmpeg で重ねる）。

    .venv/bin/python tools/make_video.py   # → videos/PAI最終課題_j_shimokawatoko.mp4

素材の動画は src/go2_eval.py で録画したもの（videos/ 以下）。
"""

import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "videos")
OUT = os.path.join(V, "PAI最終課題_j_shimokawatoko.mp4")
W, H, FPS = 1280, 720, 30
FONT_B = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_R = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
BG = (14, 20, 32)
ACCENT = (120, 200, 255)

# (種類, 素材, 秒数, 文字)
SEGMENTS = [
    ("card", None, 4, ("Go2 に後ろ足で立つ動作を\n強化学習で学習させる",
                       "PAI 最終課題  j_shimokawatoko\nGenesis + PPO (rsl-rl)  ／  映像はすべて等倍速")),
    ("card", None, 4, ("1. 四足歩行（公式の報酬）",
                       "指令速度（前進 0.5 m/s）への追従\n胴体の高さを 0.30 m に保つ、四つ足の姿勢から離れない")),
    ("clip", "go2-walking_ckpt100.mp4", 6, "四足歩行：0.55 m/s で前進"),
    ("card", None, 5, ("2. 2足立ち",
                       "追加：胴体の起き具合、胴体の高さ、前足の高さ\n削除：胴体の高さを保つ、四つ足の姿勢\n終了条件：10 度傾く → 倒れる")),
    ("clip", "go2-stand-v1_ckpt499_20s.mp4", 6, "2足立ち：後ろ足を大きく開き、関節を可動範囲の端に当てて立つ"),
    ("card", None, 5, ("3. 2足立ち・姿勢修正",
                       "追加：股を閉じる、膝を曲げる、足を腰幅に置く、左右対称\n姿勢の点は、立っているときだけ入る")),
    ("clip", "go2-stand-natural6_ckpt1999_20s.mp4", 10, "姿勢修正：足を腰幅に置き、膝を曲げて立つ"),
    ("trio", [os.path.join(ROOT, "docs/img", f) for f in ("front_walk.png", "stand_front_before.png", "stand_front_after.png")],
     6, ("四足歩行（公式）", "2足立ち", "2足立ち・姿勢修正")),
    ("card", None, 4, ("4. 2足歩行",
                       "前進の指令を加えたが、歩くようにはならなかった")),
    ("clip", "go2-biped-v10_ckpt2798_vx0.4.mp4", 5, "両足で跳ぶ：前には進まない"),
    ("clip", "go2-step-v13_ckpt2398.mp4", 5, "片足で立つ：足を踏み替えない"),
    ("card", None, 5, ("まとめ",
                       "報酬と終了条件を変えるだけで、四足歩行から\n後ろ足で立つ動作を学習した。2足歩行はできなかった")),
]


def font(path, size):
    return ImageFont.truetype(path, size)


def card_png(title, body, path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.multiline_text((90, 170), title, font=font(FONT_B, 52), fill=ACCENT, spacing=18)
    d.multiline_text((90, 380 if "\n" in title else 300), body, font=font(FONT_R, 32), fill=(230, 235, 245), spacing=20)
    img.save(path)


def caption_png(text, path):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 90, W, H], fill=(0, 0, 0, 170))
    d.text((40, H - 66), text, font=font(FONT_B, 34), fill=(255, 255, 255))
    img.save(path)


def image_png(src, text, path):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    pic = Image.open(src).convert("RGB")
    pic.thumbnail((W - 80, H - 140))
    img.paste(pic, ((W - pic.width) // 2, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 90, W, H], fill=BG)
    d.text((40, H - 66), text, font=font(FONT_B, 34), fill=(255, 255, 255))
    img.save(path)


def trio_png(srcs, labels, path):
    # 3 枚の画像を横に並べ、下にラベルを付ける（正面からの比較）
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    w = (W - 80) // 3
    for k, (src, label) in enumerate(zip(srcs, labels)):
        pic = Image.open(src).convert("RGB")
        pic = pic.resize((w - 20, int(pic.height * (w - 20) / pic.width)))
        x = 40 + k * w + 10
        img.paste(pic, (x, 160))
        tw = d.textlength(label, font=font(FONT_B, 30))
        d.text((x + (pic.width - tw) / 2, 160 + pic.height + 24), label, font=font(FONT_B, 30), fill=(255, 255, 255))
    d.text((60, 70), "正面から見た姿勢", font=font(FONT_B, 40), fill=ACCENT)
    img.save(path)


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-preset", "medium", "-crf", "20"]


def main():
    tmp = tempfile.mkdtemp()
    parts = []
    for i, (kind, src, sec, text) in enumerate(SEGMENTS):
        png, seg = os.path.join(tmp, f"{i}.png"), os.path.join(tmp, f"{i}.mp4")
        if kind == "card":
            card_png(*text, png)
        elif kind == "image":
            image_png(src, text, png)
        elif kind == "trio":
            trio_png(src, text, png)
        if kind in ("card", "image", "trio"):
            run(["ffmpeg", "-y", "-loop", "1", "-t", str(sec), "-i", png, *ENC, seg])
        else:
            caption_png(text, png)
            run(["ffmpeg", "-y", "-t", str(sec), "-i", os.path.join(V, src), "-i", png, "-filter_complex",
                 f"[0:v]scale={W}:{H},fps={FPS}[v];[v][1:v]overlay=0:0", *ENC, seg])
        parts.append(seg)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    # YouTube 用に無音の音声トラックを付ける
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
         "-shortest", "-c:v", "copy", "-c:a", "aac", OUT])
    print(f"saved: {OUT}  ({sum(s[2] for s in SEGMENTS)} 秒)")


if __name__ == "__main__":
    main()
