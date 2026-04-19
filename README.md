# League-Auto-Agent
An architecture of LLM+PPO module used for League
[中文](README_CN)
## 📌 Overview

League-Auto-Agent is an experimental framework that combines a Large Language Model (LLM) with Proximal Policy Optimization (PPO) reinforcement learning to create an autonomous agent for playing League of Legends. The system captures real-time game state via OCR and pixel-based UI monitoring, feeding a 13-dimensional state vector into a hybrid decision-making architecture that selects from 8 discrete actions at ~10 Hz.

## 🧠 Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Game Capture  │────▶│  State Encoding │────▶│  LLM + PPO      │
│ (OCR + Pixel)   │     │  (13-dim vector)│     │  Policy Network │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Game Control  │◀────│  Action Mapping │◀────│  Action Output  │
│  (Keyboard/Mouse)│     │  (8 actions)   │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Key Components

| File | Purpose |
|:-----|:--------|
| `main.py` | Entry point; checks admin privileges and launches the agent. |
| `agent.py` | Core agent loop integrating perception, decision, and control. |
| `env.py` | Environment wrapper (simulated or real) defining state/action spaces and reward logic. |
| `policy.py` | Policy network for action selection. |
| `rl_model.py` | PPO training logic and model definition. |
| `think.py` | LLM reasoning module for high-level strategic decisions. |
| `view.py` | Screen capture, OCR (Tesseract), and pixel-based UI parsing. |
| `control.py` | Keyboard/mouse input simulation (`pydirectinput`). |
| `action_space.py` | Action space definition (8 discrete actions). |
| `debug_view.py` | Visualization and debugging utilities. |
| `test_behavior.py` / `test_system.py` | Unit and integration tests. |

## 🎮 Action Space

The agent selects from **8 discrete actions** every ~100 ms (10 Hz):

| ID | Name | Description |
|:---|:-----|:------------|
| 0 | `MOVE` | Move toward enemy |
| 1 | `ATTACK` | Basic attack (if within range) |
| 2 | `Q` | Cast Q ability |
| 3 | `W` | Cast W ability |
| 4 | `E` | Cast E ability |
| 5 | `R` | Cast R ability (ultimate) |
| 6 | `D` | Use summoner spell 1 (e.g., Flash) |
| 7 | `F` | Use summoner spell 2 (e.g., Ignite) |

## 📊 State Representation

The environment encodes game state into a **13‑dimensional normalized vector** (range `[0, 1]`):

| Index | Feature | Normalization Factor |
|:------|:--------|:---------------------|
| 0 | Player HP | ÷ 150.0 |
| 1 | Player Mana | ÷ 100.0 |
| 2 | Enemy HP | ÷ 150.0 |
| 3 | Distance to enemy | ÷ 1200.0 |
| 4 | Q cooldown | ÷ 300.0 |
| 5 | W cooldown | ÷ 300.0 |
| 6 | E cooldown | ÷ 300.0 |
| 7 | R cooldown | ÷ 300.0 |
| 8 | Summoner D cooldown | ÷ 300.0 |
| 9 | Summoner F cooldown | ÷ 300.0 |
| 10 | Gold | ÷ 10000.0 |
| 11 | Minimap X (normalized) | Already [0, 1] |
| 12 | Minimap Y (normalized) | Already [0, 1] |

> **Note:** Gold extraction uses Tesseract OCR; if OCR is not configured, this dimension defaults to `0`.

## 🔧 Installation

### Prerequisites
- Windows 10/11
- Python 3.10+
- CUDA 12.6 (optional, for GPU‑accelerated training)
- League of Legends client
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for gold reading)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/albertm88/League-Auto-Agent.git
cd League-Auto-Agent

# 2. Create and activate conda environment
conda create -n league_agent python=3.10 -y
conda activate league_agent

# 3. Install Python dependencies
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
|:--------|:--------|:--------|
| `torch` | 2.10.0+cu126 | Deep learning (PPO) |
| `opencv-python` | 4.9.0.80 | Image processing & screen capture |
| `llama-cpp-python` | 0.3.36 | LLM inference |
| `numpy` | 1.26.4 | Numerical operations |
| `pydirectinput` | - | Keyboard/mouse simulation |
| `mss` | - | Fast screen capture |
| `pillow` | - | Image handling |
| `keyboard` | - | Global hotkeys (F10 emergency stop) |
| `pytesseract` | 0.3.13 | OCR for gold reading |

### Tesseract OCR Configuration

After installing Tesseract, specify the path in your code or environment variable:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## 🚀 Usage

### Running the Agent

```bash
conda activate league_agent
python main.py
```

> ⚠️ **Administrator rights required:** The script must be run as administrator to send inputs to the game client.

### Emergency Stop
Press **F10** at any time to immediately halt all operations (global hotkey).

### Training Mode

```python
from env import LoLEnv

env = LoLEnv()
state = env.reset()
for step in range(env.max_steps):
    action = policy.select_action(state)
    next_state, reward, done, _ = env.step(action)
    # training logic...
```

### Real Game Mode

```python
from agent import GameAgent

agent = GameAgent(real_game_mode=True)
agent.run()
```

## 📁 Project Structure

```
League-Auto-Agent/
├── main.py
├── agent.py
├── env.py
├── policy.py
├── rl_model.py
├── think.py
├── view.py              # Screen capture + OCR + pixel parsing
├── control.py
├── action_space.py
├── debug_view.py
├── test_behavior.py
├── test_system.py
├── requirements.txt
├── model/               # Trained models
├── debug_crops/         # Debug image crops
├── temp/
├── .gitignore
└── LICENSE
```

## 🔬 Technical Details

### Perception (Current Implementation)

Game state is extracted using:
- **OCR** (`pytesseract`) for gold and text-based UI elements.
- **Pixel sampling** to read health/mana bars and cooldown indicators from fixed screen regions.
- **Minimap position** derived from mouse/cursor detection (limited to training mode simulation).

### LLM + PPO Hybrid

- **LLM** (`llama-cpp-python`) provides strategic reasoning and context interpretation.
- **PPO** (`rl_model.py`) learns a policy that outputs discrete actions based on the 13‑dim state vector.
- The two components work asynchronously: LLM inferences occur less frequently, while PPO selects actions at ~10 Hz.

### Reward Function

Rewards are computed from:
- **Damage dealt** (positive)
- **Damage taken** (negative)
- **Kills** (large positive)
- **Distance to enemy** (penalty for being out of range)
- **Gold accumulation** (positive for farming)

## 🧪 Future Extensions

Potential improvements currently **not** implemented:
- **YOLO-based object detection** for champion, minion, and turret recognition.
- **Distance estimation** between entities using bounding‑box geometry.
- **Minimap path planning** via computer vision.

Contributions in these areas are welcome!

## 📄 License

See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Issues and pull requests are appreciated. Please discuss significant changes in an issue first.

## 📧 Contact

Open a GitHub issue for questions or collaboration inquiries.
