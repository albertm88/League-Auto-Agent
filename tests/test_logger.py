"""Unit tests for the logger utility."""

from __future__ import annotations

import logging
import os

import pytest


class TestLogger:
    def test_get_logger_returns_logger(self, tmp_path):
        from src.utils.logger import get_logger

        logger = get_logger(
            name="test_logger",
            log_dir=str(tmp_path),
            log_file="test.log",
        )
        assert isinstance(logger, logging.Logger)

    def test_log_file_is_created(self, tmp_path):
        from src.utils.logger import get_logger

        log_dir = str(tmp_path / "logs")
        logger = get_logger(
            name="test_logger_file",
            log_dir=log_dir,
            log_file="out.log",
        )
        logger.info("test message")

        log_path = os.path.join(log_dir, "out.log")
        assert os.path.isfile(log_path)

    def test_log_message_written(self, tmp_path):
        from src.utils.logger import get_logger

        log_dir = str(tmp_path)
        logger = get_logger(
            name="test_logger_content",
            log_dir=log_dir,
            log_file="content.log",
        )
        logger.info("hello world")

        log_path = os.path.join(log_dir, "content.log")
        content = open(log_path).read()
        assert "hello world" in content

    def test_no_duplicate_handlers(self, tmp_path):
        from src.utils.logger import get_logger

        name = "test_no_dupe"
        logger1 = get_logger(name, log_dir=str(tmp_path))
        logger2 = get_logger(name, log_dir=str(tmp_path))

        # Both calls return the same logger instance
        assert logger1 is logger2
        # Handlers should not have been added twice
        assert len(logger1.handlers) == len(logger2.handlers)

    def test_configure_from_config(self, tmp_path):
        from src.utils.logger import configure_from_config

        cfg = {
            "logging": {
                "level": "DEBUG",
                "log_dir": str(tmp_path),
                "log_file": "cfg_test.log",
                "max_bytes": 1024,
                "backup_count": 1,
            }
        }
        logger = configure_from_config(cfg)
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.DEBUG
