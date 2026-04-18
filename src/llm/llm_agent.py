"""LLM reasoning module using llama-cpp-python.

Wraps a GGUF model loaded with ``llama_cpp.Llama`` and provides a simple
:meth:`LLMAgent.reason` interface that accepts a text game-state description
and returns a one-hot encoded strategic guidance vector consumed by the PPO
network.

Strategic actions recognised by the agent
------------------------------------------
0  FARM              – focus on last-hitting minions
1  ENGAGE            – initiate a fight
2  RETREAT           – fall back / disengage
3  ROAM              – roam to another lane
4  RECALL            – back to base
5  BUY_ITEMS         – buy items in shop
6  PUSH_LANE         – push minion wave
7  CONTEST_OBJECTIVE – contest dragon / baron / tower
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np


# Ordered list of supported strategic action tags
STRATEGY_TAGS: List[str] = [
    "FARM",
    "ENGAGE",
    "RETREAT",
    "ROAM",
    "RECALL",
    "BUY_ITEMS",
    "PUSH_LANE",
    "CONTEST_OBJECTIVE",
]

_TAG_PATTERN = re.compile(
    r"\[(" + "|".join(re.escape(t) for t in STRATEGY_TAGS) + r")\]",
    re.IGNORECASE,
)


class LLMAgent:
    """Wraps a llama-cpp-python model for game-state reasoning.

    Parameters
    ----------
    cfg:
        The ``llm`` section of the YAML configuration dictionary.
    logger:
        Optional logger instance.
    """

    def __init__(self, cfg: Dict, logger=None) -> None:
        self._cfg = cfg
        self._logger = logger
        self._llm = None  # Lazily loaded

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the GGUF model from disk.

        Importing :mod:`llama_cpp` is deferred to this method so that the
        rest of the codebase can be imported and tested without a compiled
        model available.
        """
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python"
            ) from exc

        model_path = self._cfg.get("model_path", "models/llm_model.gguf")
        n_ctx = self._cfg.get("n_ctx", 2048)
        n_threads = self._cfg.get("n_threads", 4)

        if self._logger:
            self._logger.info("Loading LLM model from '%s' …", model_path)

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

        if self._logger:
            self._logger.info("LLM model loaded.")

    def unload(self) -> None:
        """Release the loaded model from memory."""
        self._llm = None

    # ------------------------------------------------------------------
    # Core reasoning
    # ------------------------------------------------------------------

    def reason(self, game_description: str) -> np.ndarray:
        """Query the LLM and return a one-hot strategic guidance vector.

        If the model is not loaded or if LLM inference fails, the method
        falls back to the ``FARM`` strategy (index 0).

        Parameters
        ----------
        game_description:
            Plain-text description of the current game state (produced by
            :class:`~src.vision.game_vision.GameVision`).

        Returns
        -------
        numpy.ndarray
            Float32 one-hot vector of length ``len(STRATEGY_TAGS)``.
        """
        if self._llm is None:
            if self._logger:
                self._logger.debug("LLM not loaded – defaulting to FARM strategy.")
            return self._one_hot(0)

        prompt = self._build_prompt(game_description)
        try:
            response = self._llm(
                prompt,
                max_tokens=self._cfg.get("max_tokens", 256),
                temperature=self._cfg.get("temperature", 0.7),
                stop=["\n", "</s>"],
            )
            text = response["choices"][0]["text"]
        except Exception as exc:
            if self._logger:
                self._logger.warning("LLM inference failed: %s", exc)
            return self._one_hot(0)

        strategy_idx = self._parse_strategy(text)
        if self._logger:
            self._logger.debug(
                "LLM output: '%s' → strategy index %d (%s)",
                text.strip(),
                strategy_idx,
                STRATEGY_TAGS[strategy_idx],
            )
        return self._one_hot(strategy_idx)

    def get_strategy_name(self, guidance_vector: np.ndarray) -> str:
        """Return the strategy name corresponding to a guidance vector."""
        idx = int(np.argmax(guidance_vector))
        return STRATEGY_TAGS[idx]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, game_description: str) -> str:
        system_prompt = self._cfg.get("system_prompt", "")
        return (
            f"{system_prompt}\n\n"
            f"Current game state: {game_description}\n"
            f"Recommended action:"
        )

    @staticmethod
    def _parse_strategy(text: str) -> int:
        """Extract strategy index from LLM output text.

        Returns 0 (FARM) if no recognised tag is found.
        """
        match = _TAG_PATTERN.search(text)
        if match:
            tag = match.group(1).upper()
            if tag in STRATEGY_TAGS:
                return STRATEGY_TAGS.index(tag)
        return 0  # Default: FARM

    @staticmethod
    def _one_hot(index: int) -> np.ndarray:
        vec = np.zeros(len(STRATEGY_TAGS), dtype=np.float32)
        vec[index] = 1.0
        return vec
