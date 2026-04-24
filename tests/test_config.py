import os
import pytest
from unittest.mock import patch

from bridgekit.config import require_anthropic_api_key


class TestRequireAnthropicApiKey:
    """require_anthropic_api_key() returns the key when set, raises EnvironmentError otherwise."""

    def test_returns_key_when_set(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-value"}, clear=True):
            assert require_anthropic_api_key() == "sk-ant-test-value"

    def test_raises_environment_error_when_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError):
                require_anthropic_api_key()

    def test_raises_environment_error_when_empty_string(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=True):
            with pytest.raises(EnvironmentError):
                require_anthropic_api_key()

    def test_error_message_mentions_key_and_export_command(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                require_anthropic_api_key()
            with pytest.raises(EnvironmentError, match="export ANTHROPIC_API_KEY"):
                require_anthropic_api_key()
