import os
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_message(text: str):
    """Build a minimal fake Anthropic message response."""
    content_block = MagicMock()
    content_block.text = text
    message = MagicMock()
    message.content = [content_block]
    return message


FAKE_RESPONSE = (
    "BRIDGEKIT ANALYSIS REVIEW\n"
    "─────────────────────────────────────────\n\n"
    "1. CLARITY\n"
    "✅ STRONG — The writeup is clear and jargon-free.\n\n"
    "2. AUDIENCE CLARITY\n"
    "✅ STRONG — Written for the right audience.\n\n"
    "3. STATISTICAL RIGOR\n"
    "⚠️  NEEDS WORK — Sample size is not mentioned.\n\n"
    "4. METHODOLOGY\n"
    "✅ STRONG — Approach is well explained.\n\n"
    "5. BUSINESS IMPACT\n"
    "❌ MISSING — No quantified outcomes.\n\n"
    "─────────────────────────────────────────\n"
    "BOTTOM LINE\n"
    "Add specific metrics to quantify business impact."
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvaluateReturnsString:
    """evaluate() should return a plain string."""

    def test_returns_string(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.reviewer import evaluate
                result = evaluate("We ran an A/B test on 500 users.")

        assert isinstance(result, str)

    def test_returns_non_empty_string(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.reviewer import evaluate
                result = evaluate("We ran an A/B test on 500 users.")

        assert len(result) > 0


class TestEvaluateOutputStructure:
    """evaluate() output should contain the required section headers."""

    def test_output_contains_clarity(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.reviewer import evaluate
                result = evaluate("Some analysis text.")

        assert "CLARITY" in result

    def test_output_contains_bottom_line(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.reviewer import evaluate
                result = evaluate("Some analysis text.")

        assert "BOTTOM LINE" in result

    def test_output_contains_both_required_sections(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.reviewer import evaluate
                result = evaluate("Some analysis text.")

        assert "CLARITY" in result and "BOTTOM LINE" in result


class TestEvaluateMissingApiKey:
    """evaluate() should raise EnvironmentError when the API key is absent."""

    def test_raises_environment_error_when_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.reviewer import evaluate
            with pytest.raises(EnvironmentError):
                evaluate("Some analysis text.")

    def test_error_message_mentions_key(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.reviewer import evaluate
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                evaluate("Some analysis text.")


class TestEvaluateEmptyInput:
    """evaluate() should raise ValueError for empty or whitespace-only input."""

    def test_empty_string_raises_value_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.reviewer import evaluate
            with pytest.raises(ValueError, match="empty"):
                evaluate("")

    def test_whitespace_only_raises_value_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.reviewer import evaluate
            with pytest.raises(ValueError, match="empty"):
                evaluate("   ")


class TestEvaluateApiCallShape:
    """evaluate() should pass the user text through to the Anthropic API."""

    def test_api_called_with_user_text(self):
        user_text = "Our conversion rate improved after the campaign."
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.reviewer import evaluate
                evaluate(user_text)

                call_kwargs = mock_client.messages.create.call_args
                # The user text should appear somewhere in the messages payload
                messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
                content = str(messages_arg)
                assert user_text in content
