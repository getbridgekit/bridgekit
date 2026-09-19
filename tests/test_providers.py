import os
import sys
import types
import pytest
from unittest.mock import MagicMock, patch

from bridgekit.config import parse_provider, require_api_key, Provider
from bridgekit.providers import OllamaClient, OpenRouterClient, create_client


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

    def test_infer_openrouter_from_model(self):
        assert parse_provider(model="deepseek/deepseek-v4-flash-0731:free") == Provider.OPENROUTER

    def test_infer_ollama_from_model(self):
        assert parse_provider(model="llama3.2") == Provider.OLLAMA

    def test_infer_ollama_from_mistral_model(self):
        assert parse_provider(model="mistral") == Provider.OLLAMA

    def test_defaults_to_anthropic(self):
        assert parse_provider() == Provider.ANTHROPIC

    def test_raises_on_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            parse_provider(provider="cohere")

    def test_provider_takes_precedence_over_model(self):
        assert parse_provider(provider="openai", model="claude-opus-4-6") == Provider.OPENAI

    def test_explicit_ollama(self):
        assert parse_provider(provider="ollama") == Provider.OLLAMA

    def test_explicit_openrouter(self):
        assert parse_provider(provider="openrouter") == Provider.OPENROUTER


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

    def test_returns_openrouter_key(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
            assert require_api_key(Provider.OPENROUTER) == "sk-or-test"

    def test_raises_when_openrouter_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
                require_api_key(Provider.OPENROUTER)

    def test_ollama_does_not_require_a_key(self):
        env = {k: v for k, v in os.environ.items() if k != "OLLAMA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            assert require_api_key(Provider.OLLAMA) is None


class TestOllamaClient:
    """OllamaClient talks to a local Ollama server and needs no API key."""

    def _fake_ollama_module(self):
        fake_module = types.ModuleType("ollama")
        fake_module.Client = MagicMock()
        return fake_module

    def test_no_api_key_required(self):
        fake_ollama = self._fake_ollama_module()
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            client = OllamaClient()
            assert client.api_key is None

    def test_uses_default_host_when_env_var_not_set(self):
        fake_ollama = self._fake_ollama_module()
        env = {k: v for k, v in os.environ.items() if k != "OLLAMA_HOST"}
        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {"ollama": fake_ollama}):
            OllamaClient()
            fake_ollama.Client.assert_called_once_with(host="http://localhost:11434")

    def test_uses_ollama_host_env_var_when_set(self):
        fake_ollama = self._fake_ollama_module()
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://remote-box:11434"}), \
                patch.dict(sys.modules, {"ollama": fake_ollama}):
            OllamaClient()
            fake_ollama.Client.assert_called_once_with(host="http://remote-box:11434")

    def test_create_message_returns_response_content(self):
        fake_ollama = self._fake_ollama_module()
        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = {"message": {"content": "hello from llama"}}
        fake_ollama.Client.return_value = mock_client_instance

        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            client = OllamaClient()
            result = client.create_message("system prompt", "user message", model="llama3.2", max_tokens=50)

        assert result == "hello from llama"
        mock_client_instance.chat.assert_called_once_with(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user message"}
            ],
            options={"num_predict": 50}
        )


def _fake_openai_module():
    """A stand-in for the `openai` package so these tests don't require it
    to actually be installed - it's an optional extra, and CI only installs
    `.[dev]`."""
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = MagicMock()
    return fake_module


class TestOpenRouterClient:
    """OpenRouterClient reuses OpenAI's SDK pointed at OpenRouter's base URL."""

    def test_uses_openrouter_base_url(self):
        fake_openai = _fake_openai_module()
        with patch.dict(sys.modules, {"openai": fake_openai}):
            OpenRouterClient(api_key="or-test-key")
        fake_openai.OpenAI.assert_called_once_with(api_key="or-test-key", base_url="https://openrouter.ai/api/v1")

    def test_create_message_returns_response_content(self):
        fake_openai = _fake_openai_module()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "response text"
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        fake_openai.OpenAI.return_value = mock_client_instance

        with patch.dict(sys.modules, {"openai": fake_openai}):
            client = OpenRouterClient(api_key="or-test-key")
            result = client.create_message(
                "system prompt", "user message", model="deepseek/deepseek-v4-flash-0731:free", max_tokens=100
            )

        assert result == "response text"
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="deepseek/deepseek-v4-flash-0731:free",
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user message"}
            ],
            max_tokens=100
        )


class TestCreateClientRouting:
    """create_client() routes to the right client class for the new providers."""

    def test_routes_ollama(self):
        fake_ollama = types.ModuleType("ollama")
        fake_ollama.Client = MagicMock()
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            client = create_client(Provider.OLLAMA, model="llama3.2")
        assert isinstance(client, OllamaClient)

    def test_routes_openrouter(self):
        fake_openai = _fake_openai_module()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True), \
                patch.dict(sys.modules, {"openai": fake_openai}):
            client = create_client(Provider.OPENROUTER, model="deepseek/deepseek-v4-flash-0731:free")
        assert isinstance(client, OpenRouterClient)
