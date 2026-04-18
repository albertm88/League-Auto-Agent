"""League Auto Agent – main entry point.

Usage
-----
Train mode (default):

    python main.py --config config.yaml

Resume from checkpoint:

    python main.py --config config.yaml --checkpoint checkpoints/ppo_update_000100.pt

Inference-only mode (no PPO updates):

    python main.py --config config.yaml --inference
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def obs_to_chw(bgr_frame: np.ndarray) -> np.ndarray:
    """Convert a HxWxC BGR uint8 frame to CxHxW float32 in [0, 1]."""
    return bgr_frame.transpose(2, 0, 1).astype(np.float32) / 255.0


def compute_reward(prev_state, curr_state) -> float:
    """Simple reward: positive for gaining health relative to previous step,
    negative for losing health.  Encourages staying alive.

    Extend this function with richer signals (kills, gold, objectives) once
    a proper game API is available.
    """
    if prev_state is None:
        return 0.0
    delta_health = curr_state.health_ratio - prev_state.health_ratio
    # Small living reward to encourage staying in game
    return float(delta_health * 10.0 + 0.01)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="League Auto Agent")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a PPO checkpoint to resume from.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run in inference-only mode (no PPO updates).",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #
    cfg = load_config(args.config)

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    from src.utils.logger import configure_from_config

    logger = configure_from_config(cfg)
    logger.info("League Auto Agent starting …")
    logger.info("Config loaded from '%s'.", args.config)

    # ------------------------------------------------------------------ #
    # Components                                                           #
    # ------------------------------------------------------------------ #
    from src.capture.screen_capture import ScreenCapture
    from src.vision.game_vision import GameVision
    from src.llm.llm_agent import LLMAgent
    from src.ppo.ppo_agent import PPOAgent
    from src.actions.action_executor import ActionExecutor

    capture_cfg = cfg.get("capture", {})
    vision_cfg = cfg.get("vision", {})
    llm_cfg = cfg.get("llm", {})
    ppo_cfg = cfg.get("ppo", {})
    actions_cfg = cfg.get("actions", {})

    screen_w = capture_cfg.get("resolution", {}).get("width", 1920)
    screen_h = capture_cfg.get("resolution", {}).get("height", 1080)

    screen_capture = ScreenCapture(
        monitor_index=capture_cfg.get("monitor_index", 1),
        fps=capture_cfg.get("fps", 5),
    )
    game_vision = GameVision(vision_cfg)
    llm_agent = LLMAgent(llm_cfg, logger=logger)
    ppo_agent = PPOAgent(ppo_cfg, logger=logger)
    action_executor = ActionExecutor(actions_cfg, screen_w, screen_h)

    # ------------------------------------------------------------------ #
    # Optional: load checkpoint / LLM model                               #
    # ------------------------------------------------------------------ #
    if args.checkpoint:
        ppo_agent.load_checkpoint(args.checkpoint)

    # Load LLM only if model file exists (graceful degradation otherwise)
    model_path = llm_cfg.get("model_path", "models/llm_model.gguf")
    if os.path.isfile(model_path):
        llm_agent.load()
    else:
        logger.warning(
            "LLM model not found at '%s'. "
            "Strategic guidance will default to FARM.",
            model_path,
        )

    # ------------------------------------------------------------------ #
    # Graceful shutdown on SIGINT / SIGTERM                               #
    # ------------------------------------------------------------------ #
    running = True

    def _shutdown(signum, frame):  # noqa: ANN001
        nonlocal running
        logger.info("Shutdown signal received – stopping agent.")
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ------------------------------------------------------------------ #
    # Main agent loop                                                      #
    # ------------------------------------------------------------------ #
    logger.info("Starting agent loop (inference=%s).", args.inference)
    prev_state = None

    with screen_capture:
        while running:
            try:
                # 1. Capture frame
                raw_frame = screen_capture.capture()

                # 2. Extract game state
                game_state = game_vision.process(raw_frame)

                # 3. LLM strategic guidance
                description = game_vision.describe(game_state)
                llm_guidance = llm_agent.reason(description)

                # 4. Prepare observation for PPO (CxHxW float32 [0,1])
                obs = obs_to_chw(game_state.obs_frame)

                # 5. Select action
                action, log_prob, value = ppo_agent.select_action(obs, llm_guidance)

                # 6. Execute action
                action_executor.execute(action)

                # 7. Compute reward
                reward = compute_reward(prev_state, game_state)
                done = game_state.health_ratio <= 0.0

                # 8. Store transition (training mode only)
                if not args.inference:
                    ppo_agent.store_transition(
                        obs=obs,
                        llm_features=llm_guidance,
                        action=action,
                        log_prob=log_prob,
                        reward=reward,
                        done=done,
                        value=value,
                    )

                    # PPO update when buffer is full
                    if ppo_agent.buffer.is_full():
                        ppo_agent.update(last_value=0.0 if done else value)

                prev_state = game_state

            except Exception as exc:
                logger.error("Error in agent loop: %s", exc, exc_info=True)
                time.sleep(1.0)

    logger.info("Agent loop ended.")

    # Final checkpoint
    if not args.inference:
        ppo_agent.save_checkpoint()
        logger.info("Final checkpoint saved.")


if __name__ == "__main__":
    main()
