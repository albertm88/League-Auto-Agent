# League-Auto-Agent  
用于《英雄联盟》的 LLM+PPO 模块架构  
[English](README.md)
## 📌 概述  

League-Auto-Agent 是一个实验性框架，它将大语言模型（LLM）与近端策略优化（PPO）强化学习相结合，用于创建能够自动游玩《英雄联盟》的智能体。系统通过 OCR 和基于像素的 UI 监控实时捕获游戏状态，将 13 维状态向量输入混合决策架构，以约 10 Hz 的频率从 8 个离散动作中选择执行。  

## 🧠 架构  

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   游戏画面捕获   │────▶│   状态编码       │────▶│   LLM + PPO     │
│  (OCR + 像素)   │     │  (13维向量)     │     │   策略网络      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   游戏控制       │◀────│   动作映射       │◀────│   动作输出      │
│  (键盘/鼠标)    │     │  (8个动作)      │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 核心组件  

| 文件 | 用途 |
|:-----|:--------|
| `main.py` | 入口文件；检查管理员权限并启动智能体。 |
| `agent.py` | 核心智能体循环，集成了感知、决策与控制。 |
| `env.py` | 环境封装（模拟或真实），定义状态/动作空间及奖励逻辑。 |
| `policy.py` | 用于动作选择的策略网络。 |
| `rl_model.py` | PPO 训练逻辑与模型定义。 |
| `think.py` | LLM 推理模块，用于高层策略决策。 |
| `view.py` | 屏幕捕获、OCR（Tesseract）以及基于像素的 UI 解析。 |
| `control.py` | 键盘/鼠标输入模拟（`pydirectinput`）。 |
| `action_space.py` | 动作空间定义（8 个离散动作）。 |
| `debug_view.py` | 可视化与调试工具。 |
| `test_behavior.py` / `test_system.py` | 单元测试与集成测试。 |

## 🎮 动作空间  

智能体每约 100 毫秒（10 Hz）从 **8 个离散动作** 中选择一个执行：  

| ID | 名称 | 描述 |
|:---|:-----|:------------|
| 0 | `MOVE` | 向敌人移动 |
| 1 | `ATTACK` | 普通攻击（若在范围内） |
| 2 | `Q` | 施放 Q 技能 |
| 3 | `W` | 施放 W 技能 |
| 4 | `E` | 施放 E 技能 |
| 5 | `R` | 施放 R 技能（终极技能） |
| 6 | `D` | 使用召唤师技能 1（例如闪现） |
| 7 | `F` | 使用召唤师技能 2（例如点燃） |

## 📊 状态表示  

环境将游戏状态编码为一个 **13 维归一化向量**（取值范围 `[0, 1]`）：  

| 索引 | 特征 | 归一化因子 |
|:------|:--------|:---------------------|
| 0 | 玩家生命值 | ÷ 150.0 |
| 1 | 玩家法力值 | ÷ 100.0 |
| 2 | 敌方生命值 | ÷ 150.0 |
| 3 | 与敌人距离 | ÷ 1200.0 |
| 4 | Q 技能冷却 | ÷ 300.0 |
| 5 | W 技能冷却 | ÷ 300.0 |
| 6 | E 技能冷却 | ÷ 300.0 |
| 7 | R 技能冷却 | ÷ 300.0 |
| 8 | 召唤师技能 D 冷却 | ÷ 300.0 |
| 9 | 召唤师技能 F 冷却 | ÷ 300.0 |
| 10 | 金币 | ÷ 10000.0 |
| 11 | 小地图 X 坐标（归一化） | 已在 [0, 1] 内 |
| 12 | 小地图 Y 坐标（归一化） | 已在 [0, 1] 内 |

> **注：** 金币提取使用 Tesseract OCR；若未配置 OCR，该维度默认为 `0`。

## 🔧 安装  

### 前置条件  
- Windows 10/11  
- Python 3.10+  
- CUDA 12.6（可选，用于 GPU 加速训练）  
- 《英雄联盟》客户端  
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)（用于读取金币）  

### 安装步骤  

```bash
# 1. 克隆仓库
git clone https://github.com/albertm88/League-Auto-Agent.git
cd League-Auto-Agent

# 2. 创建并激活 conda 环境
conda create -n league_agent python=3.10 -y
conda activate league_agent

# 3. 安装 Python 依赖
pip install -r requirements.txt
```

### 核心依赖  

| 包 | 版本 | 用途 |
|:--------|:--------|:--------|
| `torch` | 2.10.0+cu126 | 深度学习（PPO） |
| `opencv-python` | 4.9.0.80 | 图像处理与屏幕捕获 |
| `llama-cpp-python` | 0.3.36 | LLM 推理 |
| `numpy` | 1.26.4 | 数值运算 |
| `pydirectinput` | - | 键盘/鼠标模拟 |
| `mss` | - | 快速屏幕捕获 |
| `pillow` | - | 图像处理 |
| `keyboard` | - | 全局热键（F10 紧急停止） |
| `pytesseract` | 0.3.13 | 金币 OCR 读取 |

### Tesseract OCR 配置  

安装 Tesseract 后，在代码或环境变量中指定其路径：  

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## 🚀 使用方法  

### 运行智能体  

```bash
conda activate league_agent
python main.py
```

> ⚠️ **需要管理员权限：** 脚本必须以管理员身份运行，才能向游戏客户端发送输入。

### 紧急停止  
随时按下 **F10** 可立即停止所有操作（全局热键）。  

### 训练模式  

```python
from env import LoLEnv

env = LoLEnv()
state = env.reset()
for step in range(env.max_steps):
    action = policy.select_action(state)
    next_state, reward, done, _ = env.step(action)
    # 训练逻辑...
```

### 真实游戏模式  

```python
from agent import GameAgent

agent = GameAgent(real_game_mode=True)
agent.run()
```

## 📁 项目结构  

```
League-Auto-Agent/
├── main.py
├── agent.py
├── env.py
├── policy.py
├── rl_model.py
├── think.py
├── view.py              # 屏幕捕获 + OCR + 像素解析
├── control.py
├── action_space.py
├── debug_view.py
├── test_behavior.py
├── test_system.py
├── requirements.txt
├── model/               # 训练好的模型
├── debug_crops/         # 调试用图像裁剪
├── temp/
├── .gitignore
└── LICENSE
```

## 🔬 技术细节  

### 感知模块（当前实现）  

游戏状态通过以下方式提取：  
- **OCR**（`pytesseract`）用于金币及基于文本的 UI 元素。  
- **像素采样** 从固定屏幕区域读取血条/蓝条以及技能冷却指示器。  
- **小地图位置** 源自鼠标/光标检测（仅限于训练模式模拟）。  

### LLM + PPO 混合架构  

- **LLM**（`llama-cpp-python`）提供战略推理与上下文理解。  
- **PPO**（`rl_model.py`）学习基于 13 维状态向量输出离散动作的策略。  
- 两个组件异步工作：LLM 推理频率较低，而 PPO 以约 10 Hz 的频率选择动作。  

### 奖励函数  

奖励由以下因素计算：  
- **造成伤害**（正奖励）  
- **受到伤害**（负奖励）  
- **击杀**（高额正奖励）  
- **与敌人距离**（超出范围时惩罚）  
- **金币累积**（补刀正奖励）  

## 🧪 未来扩展方向  

目前 **尚未实现** 的潜在改进包括：  
- **基于 YOLO 的目标检测**，用于英雄、小兵、防御塔识别。  
- **基于边界框几何的距离估算**。  
- **基于计算机视觉的小地图路径规划**。  

欢迎在这些方面贡献代码！  

## 📄 许可证  

详见 [LICENSE](LICENSE) 文件。  

## 🤝 参与贡献  

欢迎提交 Issue 和 Pull Request。重大变更请先在 Issue 中讨论。  

## 📧 联系方式  

如有问题或合作意向，请在 GitHub 上提交 Issue。
