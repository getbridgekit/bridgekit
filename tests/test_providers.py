import os
import pytest
from unittest.mock import patch

from bridgekit.config import parse_provider, require_api_key, Provider


class TestParseProvider:
    """parse_provider() returns the correct Provider enum."""

    def test_explicit_anthropic(self):
        assert parse_provider(provider="anthropic") == Provider.ANTHROPIC

    def test_explicit_openai(self):
        assert parse_provider(provider="openai") == Provider.OPENAI

    def test_explicit_gemini(self):
        assert parse_provider(provider="gemini") == Provider.GEMINI

    def test_infer_anthropic_from_model(self):
        assert parse_provider(model="claude-opus-4-6") == Provider.ANTHROPIC

    def test_infer_openai_from_model(self):
        assert parse_provider(model="gpt-4o") == Provider.OPENAI

    def test_infer_gemini_from_model(self):
        assert parse_provider(model="gemini-1.5-pro") == Provider.GEMINI

    def test_defaults_to_anthropic(self):
        assert parse_provider() == Provider.ANTHROPIC

    def test_raises_on_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            parse_provider(provider="cohere")

    def test_provider_takes_precedence_over_model(self):
        assert parse_provider(provider="openai", model="claude-opus-4-6") == Provider.OPENAI


class TestRequireApiKey:
    """require_api_key() returns the correct key or raises EnvironmentError."""

    def test_returns_anthropic_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            assert require_api_key(Provider.ANTHROPIC) == "sk-ant-test"

    def test_returns_openai_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-test"}, clear=True):
            assert require_api_key(Provider.OPENAI) == "sk-openai-test"

    def test_returns_gemini_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "gemini-test"}, clear=True):
            assert require_api_key(Provider.GEMINI) == "gemini-test"

    def test_raises_when_anthropic_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                require_api_key(Provider.ANTHROPIC)

    def test_raises_when_openai_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                require_api_key(Provider.OPENAI)

    def test_raises_when_gemini_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
                require_api_key(Provider.GEMINI)
