"""Unit tests for the LLM agent module."""

from __future__ import annotations

import numpy as np
import pytest

from src.llm.llm_agent import LLMAgent, STRATEGY_TAGS


LLM_CFG = {
    "model_path": "models/llm_model.gguf",
    "n_ctx": 512,
    "n_threads": 1,
    "temperature": 0.5,
    "max_tokens": 64,
    "system_prompt": "You are a LoL assistant.",
}


class TestLLMAgent:
    def test_reason_without_load_returns_farm(self):
        """When the LLM is not loaded, the agent should default to FARM (idx 0)."""
        agent = LLMAgent(LLM_CFG)
        guidance = agent.reason("Player health: 80%.")

        assert isinstance(guidance, np.ndarray)
        assert guidance.shape == (len(STRATEGY_TAGS),)
        assert int(np.argmax(guidance)) == 0  # FARM

    def test_guidance_vector_is_one_hot(self):
        agent = LLMAgent(LLM_CFG)
        vec = agent.reason("some game state")

        assert np.sum(vec) == pytest.approx(1.0)
        assert vec.max() == pytest.approx(1.0)
        assert vec.min() == pytest.approx(0.0)

    def test_parse_strategy_known_tags(self):
        from src.llm.llm_agent import LLMAgent, STRATEGY_TAGS

        agent = LLMAgent(LLM_CFG)
        for idx, tag in enumerate(STRATEGY_TAGS):
            vec = agent._one_hot(idx)
            assert int(np.argmax(vec)) == idx

    def test_parse_strategy_from_text(self):
        from src.llm.llm_agent import LLMAgent

        agent = LLMAgent(LLM_CFG)
        assert agent._parse_strategy("[ENGAGE] the enemy.") == STRATEGY_TAGS.index("ENGAGE")
        assert agent._parse_strategy("[RETREAT] now!") == STRATEGY_TAGS.index("RETREAT")
        assert agent._parse_strategy("[RECALL] to base") == STRATEGY_TAGS.index("RECALL")
        assert agent._parse_strategy("no tag here") == 0  # default FARM

    def test_parse_strategy_case_insensitive(self):
        from src.llm.llm_agent import LLMAgent

        agent = LLMAgent(LLM_CFG)
        assert agent._parse_strategy("[farm] minions") == 0

    def test_get_strategy_name(self):
        agent = LLMAgent(LLM_CFG)
        vec = agent._one_hot(STRATEGY_TAGS.index("ROAM"))
        assert agent.get_strategy_name(vec) == "ROAM"

    def test_strategy_tags_length_matches_config(self):
        """STRATEGY_TAGS length must equal the llm_feature_dim in config."""
        assert len(STRATEGY_TAGS) == 8

    def test_reason_with_mocked_llm(self, monkeypatch):
        """Mock llama_cpp.Llama to verify the parsing path end-to-end."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.return_value = {
            "choices": [{"text": " I recommend [PUSH_LANE]."}]
        }

        agent = LLMAgent(LLM_CFG)
        agent._llm = mock_llm

        guidance = agent.reason("Player health: 50%.")
        assert agent.get_strategy_name(guidance) == "PUSH_LANE"
