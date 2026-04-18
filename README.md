# League Auto Agent

**A Python-based LLM + PPO reinforcement learning agent for League of Legends.**

> ⚠️ This project is for **learning and research purposes only** (学习交流).  
> It does not modify the game client in any way and relies solely on screen capture.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          League Auto Agent                          │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Screen       │    │ Game Vision  │    │ LLM Agent            │  │
│  │ Capture      │───▶│ (OpenCV)     │───▶│ (llama-cpp-python)   │  │
│  │ (mss)        │    │              │    │                      │  │
│  └──────────────┘    └──────┬───────┘    └──────────┬───────────┘  │
│                             │ obs_frame              │ guidance     │
│                             ▼                        ▼              │
│                      ┌─────────────────────────────────────────┐   │
│                      │          PPO Agent (PyTorch)             │   │
│                      │  ┌──────────────┐  ┌────────────────┐   │   │
│                      │  │  CNN Encoder │  │  LLM Fusion FC │   │   │
│                      │  └──────┬───────┘  └────────┬───────┘   │   │
│                      │         └──────────┬─────────┘           │   │
│                      │              ┌─────▼──────┐              │   │
│                      │              │ Policy Head│              │   │
│                      │              │ Value Head │              │   │
│                      │              └─────┬──────┘              │   │
│                      └────────────────────┼────────────────────┘   │
│                                           │ action                  │
│                                           ▼                         │
│                              ┌─────────────────────┐               │
│                              │  Action Executor     │               │
│                              │  (pyautogui)         │               │
│                              └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Module | Technology | Role |
|---|---|---|
| `src/capture/screen_capture.py` | [mss](https://python-mss.readthedocs.io/) | Captures game frames from the display |
| `src/vision/game_vision.py` | [OpenCV](https://opencv.org/) | Extracts health/mana/minimap from frames |
| `src/llm/llm_agent.py` | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | Strategic reasoning from game state text |
| `src/ppo/network.py` | [PyTorch](https://pytorch.org/) | CNN + LLM fusion actor-critic network |
| `src/ppo/replay_buffer.py` | NumPy / PyTorch | GAE-λ rollout buffer for PPO |
| `src/ppo/ppo_agent.py` | PyTorch | PPO-clip training loop & checkpointing |
| `src/actions/action_executor.py` | [pyautogui](https://pyautogui.readthedocs.io/) | Keyboard / mouse input to the game |
| `src/utils/logger.py` | Python logging | Rotating file + console logging |

---

## Requirements

- Python ≥ 3.9
- A League of Legends client running on the **same machine** in windowed or borderless mode

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `llama-cpp-python` requires a C++ compiler and CMake.  
> See the [official build guide](https://github.com/abetlen/llama-cpp-python#installation) for GPU-accelerated builds.

---

## Setup

### 1. Download a GGUF model

Place a GGUF-format language model (e.g., a Llama-3 or Mistral 7B GGUF file) in the `models/` directory and update `config.yaml`:

```yaml
llm:
  model_path: "models/your-model.gguf"
```

The agent will still work without a model file – it will simply fall back to the default `FARM` strategy.

### 2. Adjust configuration

Edit `config.yaml` to match your screen resolution, monitor index, and preferred hyperparameters.

### 3. Run the agent

**Training mode** (PPO updates enabled):

```bash
python main.py --config config.yaml
```

**Resume from checkpoint**:

```bash
python main.py --config config.yaml --checkpoint checkpoints/ppo_update_000100.pt
```

**Inference only** (no PPO updates):

```bash
python main.py --config config.yaml --inference
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
League-Auto-Agent/
├── main.py                      # Entry point
├── config.yaml                  # All configuration
├── requirements.txt
├── src/
│   ├── capture/
│   │   └── screen_capture.py    # mss-based screen capture
│   ├── vision/
│   │   └── game_vision.py       # OpenCV game-state extraction
│   ├── llm/
│   │   └── llm_agent.py         # llama-cpp-python LLM reasoning
│   ├── ppo/
│   │   ├── network.py           # CNN + LLM fusion actor-critic
│   │   ├── replay_buffer.py     # GAE rollout buffer
│   │   └── ppo_agent.py         # PPO training & checkpointing
│   ├── actions/
│   │   └── action_executor.py   # pyautogui input execution
│   └── utils/
│       └── logger.py            # Rotating logger
├── tests/                       # pytest unit tests
├── models/                      # Place your .gguf model file here
├── checkpoints/                 # Auto-saved PPO checkpoints
└── logs/                        # Agent log files
```

---

## Disclaimer

This project is intended **strictly for learning and communication purposes** (学习交流).
Use it responsibly and in accordance with Riot Games' Terms of Service.
