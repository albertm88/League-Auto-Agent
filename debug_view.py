"""
debug_view.py — ScreenReader ROI 校准工具

用法：进游戏后在终端运行
    D:\anaconda3\envs\gguf_env\python.exe debug_view.py

截图保存至 ./temp/，日志保存至 ./log/debug_view.log
对照图片确认坐标是否覆盖正确位置，若有偏差告知 GitHub Copilot 调整。
"""

import logging
import os
import time

import cv2
import numpy as np
import mss

from view import ScreenReader

IMG_DIR = "./temp"
LOG_DIR = "./log"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "debug_view.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def main():
    # ── 截图 ──────────────────────────────────────────────────────────────
    log.info("开始截图...")
    with mss.mss() as sct:
        img = sct.grab({"top": 0, "left": 0, "width": 1920, "height": 1080})
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
    log.info(f"截图尺寸: {frame.shape[1]}×{frame.shape[0]}")

    sr = ScreenReader()
    ts = time.strftime("%H%M%S")

    # ── 保存所有 ROI 截图 ──────────────────────────────────────────────────
    regions = {
        "hp":      sr._HP,
        "mana":    sr._MANA,
        "skill_q": sr._SKILL["q"],
        "skill_w": sr._SKILL["w"],
        "skill_e": sr._SKILL["e"],
        "skill_r": sr._SKILL["r"],
        "spell_d": sr._SPELL["d"],
        "spell_f": sr._SPELL["f"],
        "gold":    sr._GOLD,
        "minimap": sr._MINIMAP,
        "enemy":   sr._ENEMY,
    }
    for name, roi in regions.items():
        crop = sr._crop(frame, roi)
        path = os.path.join(IMG_DIR, f"{ts}_{name}.png")
        cv2.imwrite(path, crop)
        log.debug(f"保存 ROI [{name}] → {path}")

    full_path = os.path.join(IMG_DIR, f"{ts}_full.png")
    cv2.imwrite(full_path, frame)
    log.info(f"原始截图 → {full_path}")

    # ── 在完整截图上画出所有 ROI 框 ──────────────────────────────────────
    annotated = frame.copy()
    h, w = frame.shape[:2]
    colors = {
        "hp": (0, 255, 0),        "mana": (255, 128, 0),
        "skill_q": (0, 200, 255), "skill_w": (0, 200, 255),
        "skill_e": (0, 200, 255), "skill_r": (0, 200, 255),
        "spell_d": (200, 0, 255), "spell_f": (200, 0, 255),
        "gold": (0, 215, 255),    "minimap": (255, 255, 0),
        "enemy": (0, 0, 255),
    }
    for name, roi in regions.items():
        x1, y1, x2, y2 = roi
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        color = colors.get(name, (255, 255, 255))
        cv2.rectangle(annotated, pt1, pt2, color, 2)
        cv2.putText(annotated, name, (pt1[0], pt1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    ann_path = os.path.join(IMG_DIR, f"{ts}_annotated.png")
    cv2.imwrite(ann_path, annotated)
    log.info(f"标注截图 → {ann_path}")

    # ── 打印并记录当前读数 ─────────────────────────────────────────────────
    state = sr.read_state(frame)
    labels = ["hp", "mana", "enemy_hp", "dist", "q", "w", "e", "r", "d", "f", "gold"]
    log.info("当前读数（值域 0~1）：")
    print("\n📊 当前读数（值域 0~1，0.000 可能表示 ROI 未对准）：")
    print("-" * 36)
    for label, val in zip(labels, state):
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        line = f"  {label:<10s} {val:.3f}  [{bar}]"
        print(line)
        log.info(line.strip())
    print("-" * 36)
    print(f"\n✅ 截图 → {os.path.abspath(IMG_DIR)}/")
    print(f"   日志 → {os.path.abspath(os.path.join(LOG_DIR, 'debug_view.log'))}")
    print(f"   最重要：{ts}_annotated.png")


def tune(full_img_path: str | None = None, overrides: dict | None = None):
    """
    微调模式：从 temp/ 中加载已有的完整截图（不重新截屏），
    用当前 view.py 里的 ROI 坐标（可临时覆盖）重新画框并保存 tuned_annotated.png。

    用法：
        python debug_view.py tune
        python debug_view.py tune ./temp/062636_full.png
        python debug_view.py tune --hp 0.354,0.963,0.480,0.978
        python debug_view.py tune --gold 0.445,0.968,0.548,0.990 --hp 0.36,0.963,0.49,0.978

    可覆盖的 ROI 名称：hp  mana  gold  minimap  enemy
                      skill_q  skill_w  skill_e  skill_r
                      spell_d  spell_f
    格式：x1,y1,x2,y2（相对坐标，逗号分隔，无空格）
    """
    # ── 找到要加载的截图 ──────────────────────────────────────────────────
    if full_img_path is None:
        # 自动取 temp/ 里最新的 *_full.png
        candidates = sorted(
            [f for f in os.listdir(IMG_DIR) if f.endswith("_full.png")]
        )
        if not candidates:
            print("❌ temp/ 中没有 *_full.png，请先运行一次截图模式（不带参数）")
            return
        full_img_path = os.path.join(IMG_DIR, candidates[-1])

    frame = cv2.imread(full_img_path)
    if frame is None:
        print(f"❌ 无法加载图片：{full_img_path}")
        return
    print(f"✅ 加载截图：{full_img_path}  ({frame.shape[1]}×{frame.shape[0]})")

    sr = ScreenReader()
    h, w = frame.shape[:2]

    regions = {
        "hp":      sr._HP,
        "mana":    sr._MANA,
        "skill_q": sr._SKILL["q"],
        "skill_w": sr._SKILL["w"],
        "skill_e": sr._SKILL["e"],
        "skill_r": sr._SKILL["r"],
        "spell_d": sr._SPELL["d"],
        "spell_f": sr._SPELL["f"],
        "gold":    sr._GOLD,
        "minimap": sr._MINIMAP,
        "enemy":   sr._ENEMY,
    }

    # 命令行临时覆盖坐标（不修改 view.py，只影响本次预览）
    if overrides:
        for key, val in overrides.items():
            if key in regions:
                regions[key] = val
                print(f"  ⚡ 覆盖 {key} → {val}")
    colors = {
        "hp": (0, 255, 0),        "mana": (255, 128, 0),
        "skill_q": (0, 200, 255), "skill_w": (0, 200, 255),
        "skill_e": (0, 200, 255), "skill_r": (0, 200, 255),
        "spell_d": (200, 0, 255), "spell_f": (200, 0, 255),
        "gold": (0, 215, 255),    "minimap": (255, 255, 0),
        "enemy": (0, 0, 255),
    }

    annotated = frame.copy()
    for name, roi in regions.items():
        x1, y1, x2, y2 = roi
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        color = colors.get(name, (255, 255, 255))
        cv2.rectangle(annotated, pt1, pt2, color, 2)
        cv2.putText(annotated, name, (pt1[0], pt1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 顺便把每块 ROI 单独保存一份，方便对照
    print("\n📐 当前 ROI 像素坐标（供参考）：")
    for name, roi in regions.items():
        x1, y1, x2, y2 = roi
        px = (int(x1*w), int(y1*h), int(x2*w), int(y2*h))
        print(f"  {name:<10s}  rel{roi}  px{px}")
        crop = sr._crop(frame, roi)
        cv2.imwrite(os.path.join(IMG_DIR, f"tune_{name}.png"), crop)

    out_path = os.path.join(IMG_DIR, "tuned_annotated.png")
    cv2.imwrite(out_path, annotated)
    print(f"\n✅ 标注图已保存 → {os.path.abspath(out_path)}")
    print("   各 ROI 截图 → temp/tune_*.png")
    print("\n改完 view.py 坐标后再跑：python debug_view.py tune")


def interactive(full_img_path: str | None = None):
    """
    交互式 ROI 编辑器：在截图上直接拖框，完成后自动写入 view.py。

    用法：
        python debug_view.py edit                        # 自动用最新截图
        python debug_view.py edit ./temp/062636_full.png # 指定截图

    操作方式：
        Tab / Shift+Tab  — 切换当前编辑的 ROI
        左键拖拽         — 在画面上重新框出当前 ROI 的位置
        S                — 保存所有 ROI 坐标到 view.py
        R                — 重置当前 ROI 到 view.py 原始值
        Q / Esc          — 退出（不保存）
    """
    # ── 找截图 ─────────────────────────────────────────────────────────────
    if full_img_path is None:
        candidates = sorted(
            [f for f in os.listdir(IMG_DIR) if f.endswith("_full.png")]
        )
        if not candidates:
            print("❌ temp/ 中没有 *_full.png，请先不带参数运行一次截图")
            return
        full_img_path = os.path.join(IMG_DIR, candidates[-1])

    frame = cv2.imread(full_img_path)
    if frame is None:
        print(f"❌ 无法加载：{full_img_path}")
        return

    H, W = frame.shape[:2]
    sr = ScreenReader()

    roi_names = ["hp", "mana", "skill_q", "skill_w", "skill_e", "skill_r",
                 "spell_d", "spell_f", "gold", "minimap", "enemy"]
    colors = {
        "hp": (0, 255, 0),        "mana": (255, 128, 0),
        "skill_q": (0, 200, 255), "skill_w": (0, 200, 255),
        "skill_e": (0, 200, 255), "skill_r": (0, 200, 255),
        "spell_d": (200, 0, 255), "spell_f": (200, 0, 255),
        "gold": (0, 215, 255),    "minimap": (255, 255, 0),
        "enemy": (0, 0, 255),
    }

    def get_roi(name):
        if name.startswith("skill_"):
            return sr._SKILL[name[-1]]
        if name.startswith("spell_"):
            return sr._SPELL[name[-1]]
        return getattr(sr, f"_{name.upper()}")

    # 把所有 ROI 转成像素坐标存起来（可编辑）
    rois = {}
    for n in roi_names:
        x1, y1, x2, y2 = get_roi(n)
        rois[n] = [int(x1*W), int(y1*H), int(x2*W), int(y2*H)]

    state = {"cur": 0, "drawing": False, "sx": 0, "sy": 0, "mx": 0, "my": 0}

    def draw():
        img = frame.copy()
        for i, n in enumerate(roi_names):
            x1, y1, x2, y2 = rois[n]
            c = colors[n]
            thick = 3 if i == state["cur"] else 1
            cv2.rectangle(img, (x1, y1), (x2, y2), c, thick)
            cv2.putText(img, n, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        # 正在拖拽时预览新框
        if state["drawing"]:
            cv2.rectangle(img, (state["sx"], state["sy"]),
                          (state["mx"], state["my"]), (255, 255, 255), 1)
        cur_name = roi_names[state["cur"]]
        x1, y1, x2, y2 = rois[cur_name]
        info = (f"[{state['cur']+1}/{len(roi_names)}] {cur_name}  "
                f"rel({x1/W:.3f},{y1/H:.3f},{x2/W:.3f},{y2/H:.3f})  "
                f"Tab=next  Drag=draw  S=save  R=reset  Q=quit")
        cv2.rectangle(img, (0, H-24), (W, H), (0, 0, 0), -1)
        cv2.putText(img, info, (6, H-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img

    def mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["sx"] = state["mx"] = x
            state["sy"] = state["my"] = y
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["mx"] = x
            state["my"] = y
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x1, y1 = min(state["sx"], x), min(state["sy"], y)
            x2, y2 = max(state["sx"], x), max(state["sy"], y)
            if x2 - x1 > 3 and y2 - y1 > 3:
                rois[roi_names[state["cur"]]] = [x1, y1, x2, y2]
                # 自动跳到下一个
                state["cur"] = (state["cur"] + 1) % len(roi_names)

    def save_to_view():
        """把编辑结果写回 view.py 的 ROI 常量"""
        view_path = os.path.join(os.path.dirname(__file__), "view.py")
        with open(view_path, "r", encoding="utf-8") as f:
            src = f.read()

        def fmt(px_roi):
            x1, y1, x2, y2 = px_roi
            return (round(x1/W, 3), round(y1/H, 3),
                    round(x2/W, 3), round(y2/H, 3))

        import re
        # HP / MANA / GOLD / MINIMAP / ENEMY — 保留第一个捕获组（属性名 + 空格 + =）
        for attr, name in [("_HP", "hp"), ("_MANA", "mana"), ("_GOLD", "gold"),
                            ("_MINIMAP", "minimap"), ("_ENEMY", "enemy")]:
            rel = fmt(rois[name])
            src = re.sub(
                rf"({attr}\s*=\s*)\([^)]+\)",
                lambda m, r=rel: m.group(1) + str(r),   # 保留 _HP = 格式，只替换元组
                src,
            )
        # SKILL dict
        for sk in ["q", "w", "e", "r"]:
            rel = fmt(rois[f"skill_{sk}"])
            src = re.sub(
                rf"('{sk}':\s*)\([^)]+\)",
                lambda m, r=rel, s=sk: f"'{s}': {r}",   # s=sk 捕获当前值，避免闭包陷阱
                src,
                count=1,
            )
        # SPELL dict
        for sp in ["d", "f"]:
            rel = fmt(rois[f"spell_{sp}"])
            src = re.sub(
                rf"('{sp}':\s*)\([^)]+\)",
                lambda m, r=rel, s=sp: f"'{s}': {r}",   # s=sp 捕获当前值
                src,
                count=1,
            )

        with open(view_path, "w", encoding="utf-8") as f:
            f.write(src)
        print("\n✅ 已写入 view.py：")
        for n in roi_names:
            print(f"  {n:<10s} → {fmt(rois[n])}")

    win = "ROI Editor  Tab=next  Drag=draw  S=save  R=reset  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(W, 1440), min(H, 810))
    cv2.setMouseCallback(win, mouse)

    print(f"\n🎯 交互式 ROI 编辑器启动  截图: {full_img_path}")
    print("   Tab/Shift+Tab 切换 ROI，左键拖拽重框，S 保存到 view.py，Q 退出\n")

    while True:
        cv2.imshow(win, draw())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == ord('\t'):  # Tab
            state["cur"] = (state["cur"] + 1) % len(roi_names)
        elif key == 245:  # Shift+Tab (某些系统)
            state["cur"] = (state["cur"] - 1) % len(roi_names)
        elif key in (ord('s'), ord('S')):
            save_to_view()
        elif key in (ord('r'), ord('R')):
            n = roi_names[state["cur"]]
            x1, y1, x2, y2 = get_roi(n)
            rois[n] = [int(x1*W), int(y1*H), int(x2*W), int(y2*H)]
            print(f"  ↩ 重置 {n}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "edit":
        interactive(args[1] if len(args) >= 2 else None)
    elif args and args[0] == "tune":
        args = args[1:]
        img_path = None
        overrides = {}
        i = 0
        while i < len(args):
            a = args[i]
            if a.startswith("--"):
                key = a[2:]
                if i + 1 < len(args):
                    try:
                        vals = tuple(float(v) for v in args[i + 1].split(","))
                        if len(vals) == 4:
                            overrides[key] = vals
                        else:
                            print(f"⚠️  {a} 需要 4 个值，收到 {len(vals)} 个，已忽略")
                    except ValueError:
                        print(f"⚠️  {a} 的值 '{args[i+1]}' 无法解析，已忽略")
                    i += 2
                else:
                    print(f"⚠️  {a} 缺少值，已忽略")
                    i += 1
            else:
                img_path = a
                i += 1
        tune(img_path, overrides if overrides else None)
    else:
        main()
