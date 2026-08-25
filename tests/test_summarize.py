import os
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_message(text: str):
    content_block = MagicMock()
    content_block.text = text
    message = MagicMock()
    message.content = [content_block]
    return message


FAKE_RESPONSE = (
    "BRIDGEKIT SUMMARY\n"
    "─────────────────────────────────────────\n"
    "AUDIENCE: General Business Audience\n\n"
    "KEY TAKEAWAY\n"
    "The new onboarding flow increased upgrade rates by 3x within 30 days.\n\n"
    "WHAT WE FOUND\n"
    "- Users who engaged with reporting in week 1 were 3x more likely to upgrade\n"
    "- The effect held across acquisition channels\n"
    "- Sample size was 500 users\n\n"
    "SO WHAT\n"
    "Prioritize onboarding users to the reporting feature as a growth lever.\n"
    "─────────────────────────────────────────\n"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSummarizeReturnsString:
    """summarize() should return a non-empty string."""

    def test_returns_string(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.summarize import summarize
                result = summarize("We ran an A/B test on 500 users and saw a 3x lift in upgrades.")

        assert isinstance(result, str)
        assert len(result) > 0


class TestSummarizeOutputStructure:
    """summarize() output should contain the required section headers."""

    def test_output_contains_key_takeaway(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.summarize import summarize
                result = summarize("Some analysis text.")

        assert "KEY TAKEAWAY" in result

    def test_output_contains_so_what(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.summarize import summarize
                result = summarize("Some analysis text.")

        assert "SO WHAT" in result


class TestSummarizeMissingApiKey:
    """summarize() should raise EnvironmentError when the API key is absent."""

    def test_raises_environment_error_when_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.summarize import summarize
            with pytest.raises(EnvironmentError):
                summarize("Some analysis text.")

    def test_error_message_mentions_key(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.summarize import summarize
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                summarize("Some analysis text.")


class TestSummarizeEmptyInput:
    """summarize() should raise ValueError for empty or whitespace-only input."""

    def test_empty_string_raises_value_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.summarize import summarize
            with pytest.raises(ValueError, match="empty"):
                summarize("")

    def test_whitespace_only_raises_value_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.summarize import summarize
            with pytest.raises(ValueError, match="empty"):
                summarize("   ")


class TestSummarizeAudience:
    """summarize() should include a custom audience in the system prompt."""

    def test_custom_audience_reaches_system_prompt(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.summarize import summarize
                summarize("Some analysis text.", audience="VP of Marketing")

                call_kwargs = mock_client.messages.create.call_args
                assert "VP of Marketing" in call_kwargs.kwargs.get("system", "")

    def test_default_audience_used_when_not_specified(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.summarize import summarize
                summarize("Some analysis text.")

                call_kwargs = mock_client.messages.create.call_args
                assert "General Business Audience" in call_kwargs.kwargs.get("system", "")


class TestSummarizeCustomSystemPrompt:
    """summarize() should forward a custom system_prompt to the API, ignoring audience."""

    def test_custom_system_prompt_reaches_api(self):
        custom_prompt = "You are a data journalist writing a one-paragraph news brief."
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.summarize import summarize
                summarize("Some analysis text.", system_prompt=custom_prompt)

                call_kwargs = mock_client.messages.create.call_args
                assert call_kwargs.kwargs.get("system") == custom_prompt


class TestSummarizeApiCallShape:
    """summarize() should pass the user text through to the Anthropic API."""

    def test_api_called_with_user_text(self):
        user_text = "Our conversion rate improved after the campaign."
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.summarize import summarize
                summarize(user_text)

                call_kwargs = mock_client.messages.create.call_args
                messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
                content = str(messages_arg)
                assert user_text in content


class TestSummarizeMaxTokens:
    """summarize() should pass max_tokens through to the API."""

    def test_default_max_tokens_is_1024(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.summarize import summarize
                summarize("Some analysis text.")

                call_kwargs = mock_client.messages.create.call_args
                assert call_kwargs.kwargs.get("max_tokens") == 1024

    def test_custom_max_tokens_reaches_api(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.summarize import summarize
                summarize("Some analysis text.", max_tokens=2048)

                call_kwargs = mock_client.messages.create.call_args
                assert call_kwargs.kwargs.get("max_tokens") == 2048
