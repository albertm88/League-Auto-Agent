"""
view.py — 游戏视觉感知模块

包含两个类：
  ScreenReader      — 从截图读取 13 维归一化 state（真实游戏专用）
  VisualPerception  — 在 state 基础上附加 4 维 CV 语义特征（bot/real 通用）

13 维 state 与 env.py _normalize() 对齐（新增 2 维小地图坐标）：
  [hp, mana, enemy_hp, dist, q_cd, w_cd, e_cd, r_cd, d_cd, f_cd, gold,
   minimap_x, minimap_y]
  值域 [0, 1]
"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# pytesseract 可选（金币 OCR）
try:
    import pytesseract  # type: ignore
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
    _TESS_OK = True
except ImportError:
    _TESS_OK = False


# ─────────────────────────────────────────────────────────────────────────────
class ScreenReader:
    """
    从 LoL 游戏截图读取 13 维归一化 state。
    所有坐标为相对值 (x1/W, y1/H, x2/W, y2/H)，基于 1920×1080 标准 HUD。
    支持任意分辨率（自动缩放）。

    HUD 参考（1920×1080，HUD 缩放=默认 0）：
      HP  bar  ：画面左下，绿色血条
      Mana bar ：HP 条正下方，蓝色蓝条
      Q/W/E/R  ：底部中央技能栏，暗化 = CD 中
      D/F      ：技能栏左侧召唤师技能
      Gold     ：底部中央，技能栏正上方
      Minimap  ：右下角
    """

    # ── 相对坐标 ROI（基于 1920×1080 像素分析校准）────────────────────────────
    # HP 绿条：x=681~1094, y=1030~1045  → rel (0.355, 0.954, 0.570, 0.968)
    _HP   = (0.439, 0.951, 0.493, 0.969)
    # Mana 蓝条：x=681~1094, y=1049~1065 → rel (0.355, 0.971, 0.570, 0.986)
    _MANA = (0.44, 0.971, 0.493, 0.994)
    _SKILL    = {
        'q': (0.38, 0.875, 0.41, 0.945),
        'w': (0.415, 0.879, 0.445, 0.944),
        'e': (0.449, 0.878, 0.479, 0.946),
        'r': (0.484, 0.877, 0.515, 0.944),
    }
    _SPELL    = {
        'd': (0.523, 0.872, 0.545, 0.919),     # 默认 D=闪现
        'f': (0.549, 0.875, 0.573, 0.919),     # 默认 F=点燃/治疗
    }
    _GOLD = (0.609, 0.967, 0.647, 0.994)   # 金币数字（HUD 中右，技能栏右方）
    _MINIMAP = (0.857, 0.741, 0.998, 0.997)   # 右下角小地图
    _ENEMY = (0.057, 0.039, 0.849, 0.819)   # 上半部分（敌方头顶血条）
    # 等级数字：头像右下金色小圆圈，x=612~652, y=1048~1078
    _LEVEL = (0.323, 0.966, 0.331, 0.981)

    # ── CD 归一化上限 ─────────────────────────────────────────────────────────
    _MAX_SKILL_CD = 30.0
    _MAX_SPELL_CD = 300.0  # Flash ≈300s；治疗/点燃≈210s，取大值保守估计

    # ── 满血时敌方血条期望红色像素数（用于归一化） ──────────────────────────
    _ENEMY_HP_SCALE = 800

    def __init__(self):
        # 延迟引用同模块的 GameStateDetector，用于精确 enemy_hp 检测
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            self._detector = GameStateDetector()
        return self._detector

    # ─────────────────────────────────────────────────────────────────────────
    def _crop(self, frame: np.ndarray, rel: tuple) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = rel
        r = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        return r if r.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)

    def _bar_ratio(self, roi: np.ndarray, color: str) -> float:
        """
        计算色条填充比例。
          color='green' → HP 条 (HSV Hue 35-90)
          color='blue'  → MP 条 (HSV Hue 95-135)
        策略：找最右边仍有目标色的列位置 / 总列数。
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if color == 'green':
            mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))
        else:
            mask = cv2.inRange(hsv, np.array([95, 50, 50]), np.array([135, 255, 255]))
        col_sum = mask.sum(axis=0)
        filled = np.where(col_sum > 0)[0]
        if len(filled) == 0:
            # HP 条读不到绿色 = 0（死亡）
            # 蒂条读不到蓝色 = 英雄无蒂力条（盖伦等），返回 1.0（视为永远可释放）
            return 0.0 if color == 'green' else 1.0
        return float(filled[-1] + 1) / mask.shape[1]

    def _cd_ratio(self, icon: np.ndarray) -> float:
        """
        检测技能/召唤师图标的 CD 剩余比例 [0, 1]。
        返回值语义：
          -1.0 = 未学习（图标完全灰色/无彩色，技能不可用）
           0.0 = 就绪（已学习，无 CD）
          (0,1) = CD 中（部分灰色遮罩覆盖）
           1.0 = 完全 CD

        三态检测原理：
          1. 计算图标整体平均饱和度（avg_sat）和平均亮度（avg_val）
          2. 未学习图标：整体灰白色，饱和度极低（<25）且亮度中等偏低（<120）
             → 与CD中的暗色遮罩区分：未学习时亮度更均匀，CD遮罩有明暗分界线
          3. 就绪图标：保留原始美术彩色，平均饱和度较高（>50）
          4. CD 中：从下到上的灰色半透明遮罩，CD 区域低饱和低亮度

        注意：对于召唤师技能（D/F），不存在"未学习"状态，
        调用者可根据需求忽略 -1.0 返回值。
        """
        if icon.size == 0:
            return 0.0
        hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)  # 饱和度通道
        val = hsv[:, :, 2].astype(np.float32)  # 亮度通道
        total = float(sat.size)
        if total == 0:
            return 0.0

        avg_sat = float(sat.mean())
        avg_val = float(val.mean())

        # ── 未学习检测：整体饱和度极低 + 亮度不会太暗也不会太亮 ──
        # 未学习图标在 LoL 中呈灰白色（S<25, V 在 50-140 之间）
        # 与 CD 遮罩（S<40, V<80）的区别：未学习时整体亮度更均匀
        if avg_sat < 25 and 40 < avg_val < 150:
            # 额外验证：检查是否有明显的饱和度变化（CD有高低分界线，未学习则均匀灰）
            sat_std = float(sat.std())
            if sat_std < 20:   # 饱和度方差小 → 均匀灰色 → 未学习
                return -1.0

        # ── CD 比例计算 ──
        # CD 遮罩像素：低饱和（灰色，<40）且低亮度（<80）
        cd_px = float(((sat < 40) & (val < 80)).sum())
        raw = cd_px / total
        # 归一化：就绪图标仍有 ~15% 纯黑边框像素命中（基线噪声）
        # 完全 CD 图标约 ~80% 像素命中
        cd = max(0.0, min(1.0, (raw - 0.15) / 0.65))
        return cd

    def _read_gold(self, roi: np.ndarray) -> float:
        """
        OCR 读取金币数值，归一化到 [0, 1]（上限 10000）。
        需要 pytesseract；未安装时返回 0（不影响训练逻辑）。
        """
        if not _TESS_OK or roi.size == 0:
            return 0.0
        try:
            # 裁掉左侧 25% 区域（金币图标），只保留数字部分
            w = roi.shape[1]
            roi = roi[:, int(w * 0.25):]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # OTSU 自适应二值化（比固定阈值更稳健）
            _, thresh = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 放大以提升 OCR 精度
            thresh = cv2.resize(thresh, None, fx=3, fy=3,
                                interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(
                thresh,
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789',
                timeout=2,   # 防止 tesseract 进程挂起导致崩溃
            ).strip()
            # 游戏金币最多 5 位数（0-99999）
            if not text.isdigit() or len(text) < 1 or len(text) > 5:
                return 0.0
            gold = int(text)
            return min(gold / 10000.0, 1.0)
        except Exception as e:
            log.debug(f"gold OCR 失败: {e}")
            return 0.0

    def _read_level(self, roi: np.ndarray) -> int:
        """
        OCR 读取英雄等级数字（1–18）。
        头像右下角金色小圆圈内的白色数字。
        """
        if not _TESS_OK or roi.size == 0:
            return 0
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 放大到 300×300 提供足够像素给 OCR
            big = cv2.resize(gray, (300, 300), interpolation=cv2.INTER_CUBIC)
            # 固定阈值 160：只保留白色数字，排除金色圆圈背景
            _, thresh = cv2.threshold(big, 160, 255, cv2.THRESH_BINARY)
            # 闭操作填充游戏字体笔画间隙
            k_close = np.ones((7, 7), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
            thresh = cv2.copyMakeBorder(thresh, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=0) # type: ignore
            text = pytesseract.image_to_string(
                thresh,
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789',
                timeout=2,
            ).strip()
            if text.isdigit() and 1 <= int(text) <= 18:
                return int(text)
            return 0
        except Exception as e:
            log.debug(f"level OCR 失败: {e}")
            return 0

    def read_level(self, frame: np.ndarray) -> int:
        """
        从游戏截图读取当前英雄等级（1–18）。
        返回 0 表示读取失败。
        """
        return self._read_level(self._crop(frame, self._LEVEL))

    def _read_minimap(self, minimap: np.ndarray):
        """
        分析小地图，返回 (dist_norm, enemy_near)。
          dist_norm ∈ [0, 1]，对应游戏距离 [0, 1200]。
          enemy_near = 1.0 表示有敌方在 1200 范围内。

        原理：
          蓝色像素重心 = 己方位置
          红色像素重心 = 最近敌方位置
          像素距离 / 地图对角线 × 15000 → 游戏单位距离
        """
        if minimap.size == 0:
            return 0.5, 0.0
        h, w  = minimap.shape[:2]
        hsv   = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        # 己方蓝点
        blue_mask = cv2.inRange(hsv,
                                np.array([100, 120, 120]),
                                np.array([130, 255, 255]))
        # 敌方红点（含两段色调）
        r1 = cv2.inRange(hsv, np.array([0,   130, 100]), np.array([10,  255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 130, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)

        bm: np.ndarray = np.asarray(blue_mask, dtype=np.uint8)
        rm: np.ndarray = np.asarray(red_mask,  dtype=np.uint8)
        blue_pts = np.column_stack(np.where(bm > np.uint8(0)))
        red_pts  = np.column_stack(np.where(rm > np.uint8(0)))

        if len(blue_pts) == 0 or len(red_pts) == 0:
            return 0.5, 0.0

        my_rc  = blue_pts.mean(axis=0)  # (row, col)
        en_rc  = red_pts.mean(axis=0)
        px_dist = np.linalg.norm(my_rc - en_rc)
        diag    = np.sqrt(h ** 2 + w ** 2)
        # 小地图对角线 ≈ 地图对角线 15000 单位
        game_dist = (px_dist / diag) * 15000.0

        dist_norm  = float(min(game_dist / 1200.0, 1.0))
        enemy_near = 1.0 if game_dist < 1200 else 0.0
        return dist_norm, enemy_near

    def _read_minimap_position(self, minimap: np.ndarray) -> tuple:
        """
        从小地图提取己方英雄精确坐标 (x_norm, y_norm)。
        x_norm ∈ [0,1]: 0=地图最左(蓝方基地), 1=地图最右(红方基地)
        y_norm ∈ [0,1]: 0=地图最上(红方基地), 1=地图最下(蓝方基地)

        同时返回所有可见敌方位置列表 [(x, y), ...]。
        返回: (my_x, my_y, enemy_positions)
        """
        if minimap.size == 0:
            return 0.5, 0.5, []
        h, w = minimap.shape[:2]
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        # 己方蓝点
        blue_mask = cv2.inRange(hsv,
                                np.array([100, 120, 120]),
                                np.array([130, 255, 255]))
        bm = np.asarray(blue_mask, dtype=np.uint8)
        blue_pts = np.column_stack(np.where(bm > 0))

        if len(blue_pts) == 0:
            return 0.5, 0.5, []

        my_rc = blue_pts.mean(axis=0)  # (row, col)
        my_x = float(my_rc[1]) / w     # col → x
        my_y = float(my_rc[0]) / h     # row → y

        # 敌方红点（可能有多个敌人）
        r1 = cv2.inRange(hsv, np.array([0,   130, 100]), np.array([10,  255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 130, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)
        rm = np.asarray(red_mask, dtype=np.uint8)
        red_pts = np.column_stack(np.where(rm > 0))

        enemy_positions = []
        if len(red_pts) > 0:
            # 使用连通域分离多个敌人
            n_labels, labels = cv2.connectedComponents(rm)
            for lbl in range(1, n_labels):
                pts = np.column_stack(np.where(labels == lbl))
                if len(pts) < 3:
                    continue
                centroid = pts.mean(axis=0)
                ex = float(centroid[1]) / w
                ey = float(centroid[0]) / h
                enemy_positions.append((ex, ey))

        return my_x, my_y, enemy_positions

    def get_minimap_crop(self, frame: np.ndarray) -> np.ndarray:
        """返回小地图区域裁切图（BGR），供 VLM 分析使用。"""
        return self._crop(frame, self._MINIMAP)

    def _read_enemy_hp(self, region: np.ndarray) -> float:
        """
        从画面上半部分检测最近敌方血条比例。
        原理：LoL 敌方头顶血条 = 绿→黄→红色细条。
              取最大连续红/绿色水平线段长度 / 血条总长度。
        简化版：红色像素数 / 期望满血像素数。
        """
        if region.size == 0:
            return 1.0
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # 血条颜色：绿色（满）→ 黄色（半）→ 红色（残）
        green = cv2.inRange(hsv, np.array([35, 60, 60]),  np.array([90, 255, 255]))
        yellow= cv2.inRange(hsv, np.array([20, 60, 60]),  np.array([34, 255, 255]))
        red1  = cv2.inRange(hsv, np.array([0,  60, 60]),  np.array([19, 255, 255]))
        red2  = cv2.inRange(hsv, np.array([170,60, 60]),  np.array([180,255, 255]))
        combined = cv2.bitwise_or(
            cv2.bitwise_or(green, yellow),
            cv2.bitwise_or(red1, red2)
        )
        hp_px = int(np.asarray(combined).sum()) // 255
        return float(min(hp_px / self._ENEMY_HP_SCALE, 1.0))

    # ── 技能等级 / 升级光效检测 ───────────────────────────────────────────────

    # 可升级时图标出现的金色脉冲光晕（HSV 金/橙黄区间）
    _LEVELUP_GOLD_THRESH = 0.06   # 超过 6% 像素呈金色 → 判定可升级

    def _has_levelup_glow(self, roi: np.ndarray) -> bool:
        """
        检测技能图标 ROI 中是否存在可升级的金色光效。
        LoL 未学/可升技能图标会出现金黄色脉冲光晕（H=15–38, S>100, V>140）。
        与普通 CD 中（全黑覆盖）和就绪（彩色图标）在色调上可明确区分。
        """
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gold = cv2.inRange(hsv,
                           np.array([15, 100, 140]),
                           np.array([38, 255, 255]))
        ratio = float(np.count_nonzero(gold)) / gold.size
        return ratio > self._LEVELUP_GOLD_THRESH

    def read_levelup(self, frame: np.ndarray) -> dict:
        """
        检测各技能是否可升级（图标出现金色光效）。
        返回 {'q': bool, 'w': bool, 'e': bool, 'r': bool}

        升级触发：角色升级时游戏在对应技能图标上叠加金色动画；
                  检测到此光效即表示当前有技能点可分配。
        """
        return {
            sk: self._has_levelup_glow(self._crop(frame, self._SKILL[sk]))
            for sk in ('q', 'w', 'e', 'r')
        }

    def skill_available(self, state: np.ndarray, skill_levels: dict) -> dict:
        """
        综合技能等级和 CD 状态，判断各技能当前是否可释放。
          level == 0  → 未学，不可释放（返回 False）
          level >  0 且 cd_ratio < 0.15 → 就绪（返回 True）
          level >  0 且 cd_ratio >= 0.15 → CD 中（返回 False）

        state 索引（与 read_state 对齐）: q=4, w=5, e=6, r=7
        skill_levels: agent 维护的 {'q': int, 'w': int, 'e': int, 'r': int}
        """
        _CD_IDX = {'q': 4, 'w': 5, 'e': 6, 'r': 7}
        result = {}
        for sk, idx in _CD_IDX.items():
            lv  = skill_levels.get(sk, 0)
            cd  = float(state[idx]) if idx < len(state) else 1.0
            result[sk] = bool(lv > 0 and cd < 0.15)
        return result

    def read_skill_states(self, frame: np.ndarray) -> dict:
        """
        读取各技能的精确三态状态，返回字典。
        每个技能的值为:
          'not_learned' — 技能未学习（灰色图标）
          'ready'       — 技能已学习且可释放
          'on_cd'       — 技能已学习但在冷却中

        该方法供 agent 使用，不进入 state 向量。
        """
        result = {}
        for sk in ('q', 'w', 'e', 'r'):
            raw = self._cd_ratio(self._crop(frame, self._SKILL[sk]))
            if raw < 0:
                result[sk] = 'not_learned'
            elif raw < 0.10:
                result[sk] = 'ready'
            else:
                result[sk] = 'on_cd'
        for sp in ('d', 'f'):
            raw = self._cd_ratio(self._crop(frame, self._SPELL[sp]))
            if raw < 0.10:
                result[sp] = 'ready'
            else:
                result[sp] = 'on_cd'
        return result

    # ── 主接口 ────────────────────────────────────────────────────────────────
    def read_state(self, frame: np.ndarray) -> np.ndarray:
        """
        从游戏截图提取 13 维归一化 state [0, 1]。
        格式：
          [hp, mana, enemy_hp, dist, q, w, e, r, d_cd, f_cd, gold,
           minimap_x, minimap_y]
        """
        try:
            hp   = self._bar_ratio(self._crop(frame, self._HP),   'green')
            mana = self._bar_ratio(self._crop(frame, self._MANA),  'blue')

            # 技能 CD：-1=未学习，0=就绪，(0,1)=CD中
            # 对于 state 向量：未学习映射为 1.0（不可用 = 等同于满 CD）
            q_raw = self._cd_ratio(self._crop(frame, self._SKILL['q']))
            w_raw = self._cd_ratio(self._crop(frame, self._SKILL['w']))
            e_raw = self._cd_ratio(self._crop(frame, self._SKILL['e']))
            r_raw = self._cd_ratio(self._crop(frame, self._SKILL['r']))
            d_raw = self._cd_ratio(self._crop(frame, self._SPELL['d']))
            f_raw = self._cd_ratio(self._crop(frame, self._SPELL['f']))

            # 未学习(-1) → 1.0（满CD=不可用），其他保留原值
            q = 1.0 if q_raw < 0 else q_raw
            w = 1.0 if w_raw < 0 else w_raw
            e = 1.0 if e_raw < 0 else e_raw
            r = 1.0 if r_raw < 0 else r_raw
            d = max(0.0, d_raw)   # 召唤师技能不存在未学习
            f = max(0.0, f_raw)

            gold = self._read_gold(self._crop(frame, self._GOLD))

            dist_norm, _enemy_near = self._read_minimap(
                self._crop(frame, self._MINIMAP)
            )

            # 精确小地图坐标
            minimap_crop = self._crop(frame, self._MINIMAP)
            my_x, my_y, _enemy_pos = self._read_minimap_position(minimap_crop)

            # 通过 detect_units 精确检测最近敌方英雄 HP（取代全区域像素计数）
            units = self._get_detector().detect_units(frame)
            if units["enemies"]:
                enemy_hp = units["enemies"][0][2]
            else:
                enemy_hp = 0.0

            # HP 异常检测：蓝条满但血条为零 → ROI 可能未对准
            if hp < 0.01 and mana > 0.5:
                log.warning(
                    "⚠️  HP=0%% 但 MP>50%% — HP ROI 可能未对准！"
                    "请运行 ScreenReader().save_debug_crops(frame) 校准"
                )
                hp = 0.5   # 保守估计，避免误判死亡

            return np.array(
                [hp, mana, enemy_hp, dist_norm, q, w, e, r, d, f, gold,
                 my_x, my_y],
                dtype=np.float32,
            )
        except Exception as ex:
            log.warning(f"ScreenReader.read_state 失败，返回零向量: {ex}")
            return np.zeros(13, dtype=np.float32)

    # ROI 标注颜色（与 debug_view.py 保持一致，方便肉眼对照）
    _ROI_COLORS = {
        "hp":      (0, 255, 0),
        "mana":    (255, 128, 0),
        "skill_q": (0, 200, 255), "skill_w": (0, 200, 255),
        "skill_e": (0, 200, 255), "skill_r": (0, 200, 255),
        "spell_d": (200, 0, 255), "spell_f": (200, 0, 255),
        "gold":    (0, 215, 255),
        "minimap": (255, 255, 0),
        "enemy":   (0, 0, 255),
        "level":   (0, 255, 255),
    }

    def save_debug_crops(self, frame: np.ndarray, out_dir: str = "./debug_crops") -> None:
        """
        将每个 ROI 区域保存为图片，并生成彩色标注全图，用于校准坐标。
        进游戏后运行，对照 _annotated.png 确认每块区域是否覆盖正确的 HUD 位置。
        输出文件：
          {name}.png        — 各 ROI 单独截图
          _full.png         — 原始全屏截图
          _annotated.png    — 带彩色 ROI 框的标注图（最重要）
        """
        import os
        os.makedirs(out_dir, exist_ok=True)
        regions = {
            "hp":      self._HP,
            "mana":    self._MANA,
            "skill_q": self._SKILL['q'],
            "skill_w": self._SKILL['w'],
            "skill_e": self._SKILL['e'],
            "skill_r": self._SKILL['r'],
            "spell_d": self._SPELL['d'],
            "spell_f": self._SPELL['f'],
            "gold":    self._GOLD,
            "minimap": self._MINIMAP,
            "enemy":   self._ENEMY,
            "level":   self._LEVEL,
        }
        h, w = frame.shape[:2]

        # ── 单独 crop 图 ────────────────────────────────────────────────────
        for name, roi in regions.items():
            crop = self._crop(frame, roi)
            cv2.imwrite(os.path.join(out_dir, f"{name}.png"), crop)

        # ── 全图 ────────────────────────────────────────────────────────────
        cv2.imwrite(os.path.join(out_dir, "_full.png"), frame)

        # ── 彩色标注图（核心：直接看哪里框错了）─────────────────────────────
        annotated = frame.copy()
        for name, roi in regions.items():
            x1, y1, x2, y2 = roi
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            color = self._ROI_COLORS.get(name, (255, 255, 255))
            cv2.rectangle(annotated, pt1, pt2, color, 2)
            cv2.putText(annotated, name, (pt1[0], max(pt1[1] - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(os.path.join(out_dir, "_annotated.png"), annotated)

        # ── 打印读数 ────────────────────────────────────────────────────────
        state = self.read_state(frame)
        labels = [
            "hp", "mana", "enemy_hp", "dist", "q", "w", "e", "r", "d", "f", "gold",
            "minimap_x", "minimap_y"
        ]
        print(f"\n[ScreenReader] debug crops -> {os.path.abspath(out_dir)}")
        print("-" * 38)
        for label, val in zip(labels, state):
            bar = chr(9608) * int(val * 20) + chr(9617) * (20 - int(val * 20))
            print(f"  {label:<10s} {val:.3f}  [{bar}]")
        print("-" * 38)
        print(f"  最重要: _annotated.png  (确认 ROI 框位置)")


# ─────────────────────────────────────────────────────────────────────────────
class GameStateDetector:
    """
    游戏元状态检测器（纯 OpenCV，无 LLM 依赖）：

      1. is_game_loaded()      — 判断游戏 HUD 是否已加载完成
      2. detect_spawn_side()   — 判断出生在蓝方（左下）还是红方（右上）
      3. detect_death()        — 检测英雄是否处于死亡/等待复活状态
      4. detect_units()        — 检测视野内所有单位的血条位置和 HP 比例
         └─ 按颜色（红=敌、绿=友）和宽度（宽=英雄、窄=小兵）分类

    血条检测原理：
      LoL 单位头顶血条是水平细条（宽>>高，aspect > 4，高 2–15 px）。
      颜色分类：
        敌方英雄/小兵 → 红色外框（HSV H 0–10 / 170–180）
        友方英雄/小兵 → 绿色（HSV H 38–88）
      英雄 vs 小兵区分：血条像素宽度 > 40 px = 英雄，≤ 40 px = 小兵。
    """

    SIDE_UNKNOWN = "unknown"
    SIDE_BLUE    = "blue"   # 蓝色方：出生在地图左下，小地图左下角
    SIDE_RED     = "red"    # 红色方：出生在地图右上，小地图右上角

    # 小地图 ROI（与 ScreenReader._MINIMAP 对齐）
    _MINIMAP_ROI   = (0.857, 0.741, 0.998, 0.999)
    # 死亡/复活计时区域（屏幕中央偏下）
    _RESPAWN_ROI   = (0.35, 0.62, 0.65, 0.80)
    # 商店窗口中心区域（用于检测开局/复活时商店是否打开）
    _SHOP_CENTER_ROI = (0.12, 0.08, 0.88, 0.88)
    # 游戏视角区域（排除底部 HUD，约占画面高度 84%）
    _GAME_VIEW_YMAX = 0.84
    # 英雄血条最小宽度（px）
    _CHAMP_MIN_W   = 40

    def _crop(self, frame: np.ndarray, rel: tuple) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = rel
        r = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        return r if r.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)

    # ── 1. 游戏加载完成检测 ───────────────────────────────────────────────────
    def is_game_loaded(self, frame: np.ndarray, screen_reader: "ScreenReader") -> bool:
        """
        检测游戏 HUD 是否已可见。

        两个独立条件，任意一个满足即可：
          A. HP 绿条读值 > 0.5（HUD 血条可见）
          B. 小地图区域存在蓝色像素（己方头像出现在小地图上）

        不再依赖 Q_CD，因为泉水里技能未点亮时图标全黑，
        _cd_ratio 会返回接近 1.0，导致条件永远不成立。
        """
        try:
            state = screen_reader.read_state(frame)
            hp = float(state[0])
            if hp > 0.5:
                return True
        except Exception:
            pass

        # 备用：检测小地图蓝色像素
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self._MINIMAP_ROI
            mm = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
            if mm.size == 0:
                return False
            hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
            blue = cv2.inRange(hsv, np.array([100, 150, 100]), np.array([130, 255, 255]))
            return int(blue.sum()) // 255 > 20
        except Exception:
            return False

    # ── 2. 出生方检测 ─────────────────────────────────────────────────────────
    def detect_spawn_side(self, frame: np.ndarray) -> str:
        """
        通过小地图上己方蓝点重心位置判断出生方。
          蓝方：蓝点在小地图左下象限（row_norm > 0.55, col_norm < 0.45）
          红方：蓝点在小地图右上象限（row_norm < 0.45, col_norm > 0.55）
        """
        minimap = self._crop(frame, self._MINIMAP_ROI)
        if minimap.size == 0:
            return self.SIDE_UNKNOWN
        h, w = minimap.shape[:2]
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        # 己方蓝色圆点（纯蓝，高饱和度）
        blue_mask = cv2.inRange(
            hsv, np.array([100, 150, 100]), np.array([130, 255, 255])
        )
        bm  = np.asarray(blue_mask, dtype=np.uint8)
        pts = np.column_stack(np.where(bm > 0))   # shape (N, 2): [row, col]

        if len(pts) < 10:
            return self.SIDE_UNKNOWN

        row_norm = float(pts[:, 0].mean()) / h   # 0=上边缘，1=下边缘
        col_norm = float(pts[:, 1].mean()) / w   # 0=左边缘，1=右边缘

        if row_norm > 0.55 and col_norm < 0.45:
            return self.SIDE_BLUE   # 左下 → 蓝方
        if row_norm < 0.45 and col_norm > 0.55:
            return self.SIDE_RED    # 右上 → 红方
        return self.SIDE_UNKNOWN

    # ── 3. 死亡检测 ───────────────────────────────────────────────────────────
    def detect_death(self, frame: np.ndarray, hp: float) -> bool:
        """
        检测英雄是否处于死亡状态（等待复活）。
        三重确认（必须同时满足）：
          1. HP 读値 < 0.15
          2. 游戏视野平均亮度 < 70（死亡时屏幕整体变暗）
          3. 游戏视野平均饱和度 < 40（死亡时屏幕去色变灰）
        泉水/草地场景有丰富颜色（饱和度高），不会同时触发条件 2+3。
        """
        if hp >= 0.15:
            return False
        roi = self._crop(frame, self._RESPAWN_ROI)
        if roi.size == 0:
            return False
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_br  = float(np.asarray(gray,  dtype=np.float32).mean())
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_sat = float(hsv[:, :, 1].astype(np.float32).mean())
        log.debug(f"death_check: hp={hp:.3f} br={avg_br:.1f} sat={avg_sat:.1f}")
        # 死亡屏幕：画面暗 AND 颜色极少（两者缺一不可）
        return avg_br < 70 and avg_sat < 40

    def detect_shop_open(self, frame: np.ndarray) -> bool:
        """
        检测商店页面是否打开（启发式）：
          1) 中央大面板区域边缘密度较高（UI 文本/边框密集）
          2) 同时饱和度偏低（深色半透明面板）
        该检测仅用于“是否需要按 P 关闭商店”，宁可保守误判为 False。
        """
        roi = self._crop(frame, self._SHOP_CENTER_ROI)
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 160)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[:, :, 1].astype(np.float32).mean())
        return edge_ratio > 0.075 and sat_mean < 85.0

    # ── 4. 单位检测 ───────────────────────────────────────────────────────────
    def detect_units(self, frame: np.ndarray) -> dict:
        """
        检测游戏视角区域内所有单位的血条。

        返回字典（坐标均为画面归一化值 [0,1]）：
          enemies       : [(x, y, hp_ratio), ...]  敌方英雄，按距屏幕中心由近到远排序
          enemy_minions : [(x, y, hp_ratio), ...]  敌方小兵
          allies        : [(x, y, hp_ratio), ...]  友方英雄
          ally_minions  : [(x, y, hp_ratio), ...]  友方小兵

        hp_ratio = 1.0 满血，0.0 已死（残血时红色像素段短于总段）。
        """
        h, w  = frame.shape[:2]
        gmax  = int(h * self._GAME_VIEW_YMAX)
        game  = frame[:gmax, :]
        gh, gw = game.shape[:2]

        hsv = cv2.cvtColor(game, cv2.COLOR_BGR2HSV)

        # 敌方血条：红色（两段色相）
        r1 = cv2.inRange(hsv, np.array([0,   110, 80]),  np.array([10,  255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 110, 80]),  np.array([180, 255, 255]))
        enemy_mask = cv2.bitwise_or(r1, r2)

        # 友方血条：绿色
        ally_mask = cv2.inRange(hsv, np.array([38, 80, 60]), np.array([88, 255, 255]))

        # ── 遮罩已知 UI 区域，防止大量误检 ────────────────────────────────
        # 小地图（右下角）：包含大量红/绿色的英雄/建筑标记
        mm_x = int(gw * 0.84)
        mm_y = int(gh * 0.86)
        enemy_mask[mm_y:, mm_x:] = 0
        ally_mask[mm_y:, mm_x:]  = 0
        # 击杀通知栏（右上角）
        kf_x = int(gw * 0.75)
        kf_y = int(gh * 0.12)
        enemy_mask[:kf_y, kf_x:] = 0
        ally_mask[:kf_y, kf_x:]  = 0

        enemy_bars = self._extract_hp_bars(enemy_mask, gw, gh)
        ally_bars  = self._extract_hp_bars(ally_mask,  gw, gh)

        CW = self._CHAMP_MIN_W
        # LoL 最多 5v5，超出数量必为误检
        return {
            "enemies":       [(x, y, hp) for x, y, hp, bw in enemy_bars if bw >  CW][:5],
            "enemy_minions": [(x, y, hp) for x, y, hp, bw in enemy_bars if bw <= CW][:20],
            "allies":        [(x, y, hp) for x, y, hp, bw in ally_bars  if bw >  CW][:5],
            "ally_minions":  [(x, y, hp) for x, y, hp, bw in ally_bars  if bw <= CW][:20],
        }

    def _extract_hp_bars(
        self, mask: np.ndarray, frame_w: int, frame_h: int
    ) -> list:
        """
        从二值颜色 mask 提取血条列表。

        血条特征过滤：
          高度 2–16 px，宽度 15–180 px，宽/高比 > 4。
        同一行（±4 px）重复的条只保留最宽一条（避免英雄底框重复计数）。

        返回：[(x_norm, y_norm, hp_ratio, bar_px_width), ...]
        按距屏幕水平中心的距离升序排序（最近的单位在前）。
        """
        # 水平闭运算：连接同行因抗锯齿断开的像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results: list = []
        seen_rows: dict = {}   # row_key → index in results（保留最宽）

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 尺寸过滤
            if bh < 2 or bh > 16:
                continue
            if bw < 15 or bw > 180:
                continue
            if bw / max(bh, 1) < 4:
                continue
            if bw * bh < 50:
                continue   # 面积过小 → 噪点

            # HP 比例：原始 mask 中实际有色列的最右位置 / 总宽
            roi   = mask[y:y + bh, x:x + bw]
            cols  = np.asarray(roi, dtype=np.uint8).sum(axis=0)
            filled = np.where(cols > 0)[0]
            if len(filled) == 0:
                continue
            hp_ratio = float(filled[-1] + 1) / bw

            cx = float(x + bw / 2) / frame_w
            cy = float(y + bh / 2) / frame_h

            row_key = y // 4
            if row_key in seen_rows:
                # 同行已有记录，保留宽度更大的
                idx = seen_rows[row_key]
                if bw > results[idx][3]:
                    results[idx] = (cx, cy, hp_ratio, bw)
            else:
                seen_rows[row_key] = len(results)
                results.append((cx, cy, hp_ratio, bw))

        # 按距屏幕水平中心距离排序
        results.sort(key=lambda t: abs(t[0] - 0.5))
        return results

    def closest_enemy(self, units: dict) -> "tuple | None":
        """返回最近敌方英雄 (x, y, hp)，不存在返回 None。"""
        return units["enemies"][0] if units["enemies"] else None

    def unit_summary(self, units: dict) -> tuple:
        """
        提炼 4 维语义摘要，直接对齐 VLM 4 维格式：
          (enemy_near, enemy_low_hp, minion_nearby, can_kill)
        """
        enemies = units["enemies"]
        enemy_near   = 1.0 if enemies else 0.0
        closest_hp   = enemies[0][2] if enemies else 1.0
        enemy_low_hp = 1.0 if closest_hp < 0.30 else 0.0
        minion_nearby = 1.0 if units["enemy_minions"] else 0.0
        can_kill      = 1.0 if (enemy_near and closest_hp < 0.15) else 0.0
        return enemy_near, enemy_low_hp, minion_nearby, can_kill


# ─────────────────────────────────────────────────────────────────────────────
class VisualPerception:
    """
    纯 OpenCV 视觉感知模块，不依赖 LLM。
    frame 存在时输出 17 维 (13 state + 4 CV 语义)，否则输出 13 维。
    """

    def __init__(self, thinker=None):
        # thinker = DecisionThinker 实例；有 VLM 时用缓存，否则降级 CV
        self._thinker  = thinker
        self._detector = GameStateDetector()   # 单位检测

    def get_observation(self, state=None, frame=None) -> np.ndarray:
        base = np.array(state, dtype=np.float32)
        if frame is None:
            return base
        try:
            # 优先使用 Thread C 写入的 VLM 缓存，VLM 不可用时降级为纯 CV
            if self._thinker is not None and self._thinker.llm_available:
                vlm = self._thinker.get_vlm_cache()
                semantic = np.array([
                    vlm["enemy_near"], vlm["enemy_low_hp"],
                    vlm["in_danger"],  vlm["can_kill"],
                ], dtype=np.float32)
            else:
                semantic = self._extract_semantic(frame)
            return np.concatenate([base, semantic])
        except Exception as e:
            log.warning(f"视觉特征提取失败，降级为纯 state: {e}")
            return base

    # ── OpenCV 语义特征提取 ──────────────────────────────────────────────────

    def _extract_semantic(self, frame: np.ndarray) -> np.ndarray:
        """
        从游戏截图提取 4 维布尔特征（与 VLM 缓存格式对齐）:
          [enemy_near, enemy_low_hp, minion_nearby, can_kill]

        使用 GameStateDetector.detect_units() 精确识别血条，
        比原来的全区域像素计数更准确（区分英雄/小兵，知道各自HP比例）。
        """
        units = self._detector.detect_units(frame)
        en, el, mn, ck = self._detector.unit_summary(units)

        # in_danger 改为从友方侧感知（我方HUD已由ScreenReader读取，这里用enemy_near替代）
        h, w = frame.shape[:2]
        hud  = frame[int(h * 0.88):, :int(w * 0.25)]
        in_danger = self._detect_player_low_hp(hud)

        # 用 in_danger 覆盖 can_kill 的第三位（保持4维格式不变）
        return np.array([en, el, in_danger, ck], dtype=np.float32)

    def _detect_enemy_hp_bars(self, region: np.ndarray):
        """检测红色敌方血条。返回 (enemy_near, enemy_low_hp)。"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 120, 80]),   np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 80]), np.array([180, 255, 255]))
        red_px = int(cv2.bitwise_or(mask1, mask2).sum()) // 255

        enemy_near  = 1.0 if red_px > 200  else 0.0
        # 血条残血时红色像素很少（条变短），仍有一点点红
        enemy_low_hp = 1.0 if 30 < red_px < 300 else 0.0
        return enemy_near, enemy_low_hp

    def _detect_player_low_hp(self, hud: np.ndarray) -> float:
        """检测 HUD 区域是否出现大量红色（我方 HP 危险）。"""
        if hud.size == 0:
            return 0.0
        hsv = cv2.cvtColor(hud, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0,   100, 100]), np.array([15,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 100, 100]), np.array([180, 255, 255]))
        red_px = int(cv2.bitwise_or(mask1, mask2).sum()) // 255
        return 1.0 if red_px > 80 else 0.0
