import ctypes
import sys
import logging
from agent import GameAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _check_admin():
    """LoL 以管理员运行时，脚本也需要管理员权限才能发送键鼠输入。"""
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("=" * 55)
            print("  \u26a0\ufe0f  当前非管理员权限！")
            print("  LoL 以管理员运行，脚本也需要管理员才能控制游戏。")
            print("  请右键 → '以管理员身份运行' 终端/VS Code。")
            print("=" * 55)
            resp = input("  输入 y 继续（操作可能无法送达游戏），其他退出: ").strip().lower()
            if resp != "y":
                sys.exit(1)
        else:
            print("✅ 管理员权限确认")
    except Exception:
        pass

if __name__ == "__main__":
    _check_admin()
    agent = GameAgent(real_game_mode=True)  # True=真实游戏  False=离线预热
    print("💡 退出方式：Ctrl+C（终端）或 F10（全局急停，无需切回终端）")
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n⚠️ Ctrl+C — 安全退出中...")
    finally:
        agent.stop()
