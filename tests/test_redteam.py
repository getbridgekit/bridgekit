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
    "BRIDGEKIT RED TEAM\n"
    "─────────────────────────────────────────\n"
    "STAKEHOLDER: Skeptical Senior Executive\n\n"
    "CRITIQUE 1: Sample Size\n"
    '❯ "How many users was this actually tested on?"\n'
    "WHY IT LANDS: No sample size is mentioned anywhere.\n"
    "TO ADDRESS: Report n for each group with a power calculation.\n\n"
    "CRITIQUE 2: Causation vs Correlation\n"
    '❯ "You\'re assuming the feature caused this lift — prove it."\n'
    "WHY IT LANDS: No control group is described.\n"
    "TO ADDRESS: Show the experimental design with random assignment.\n\n"
    "CRITIQUE 3: Business Impact\n"
    '❯ "What does a 5% lift actually mean in dollars?"\n'
    "WHY IT LANDS: Directional claims are not quantified.\n"
    "TO ADDRESS: Translate the metric into revenue or cost terms.\n\n"
    "─────────────────────────────────────────\n"
    "HARDEST QUESTION TO ANSWER\n"
    "What is the p-value and did you correct for multiple comparisons?"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRedteamReturnsString:
    """redteam() should return a non-empty string."""

    def test_returns_string(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.redteam import redteam
                result = redteam("We ran an A/B test and saw a 5% lift in conversions.")

        assert isinstance(result, str)
        assert len(result) > 0


class TestRedteamOutputStructure:
    """redteam() output should contain the required section headers."""

    def test_output_contains_critique(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.redteam import redteam
                result = redteam("We ran an A/B test and saw a 5% lift in conversions.")

        assert "CRITIQUE" in result

    def test_output_contains_hardest_question(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.redteam import redteam
                result = redteam("We ran an A/B test and saw a 5% lift in conversions.")

        assert "HARDEST QUESTION" in result


class TestRedteamMissingApiKey:
    """redteam() should raise EnvironmentError when ANTHROPIC_API_KEY is absent."""

    def test_raises_environment_error_when_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.redteam import redteam
            with pytest.raises(EnvironmentError):
                redteam("Some analysis text.")

    def test_error_message_mentions_key(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.redteam import redteam
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                redteam("Some analysis text.")


class TestRedteamEmptyInput:
    """redteam() should raise ValueError for empty or whitespace-only input."""

    def test_empty_string_raises_value_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.redteam import redteam
            with pytest.raises(ValueError, match="empty"):
                redteam("")

    def test_whitespace_only_raises_value_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.redteam import redteam
            with pytest.raises(ValueError, match="empty"):
                redteam("   ")


class TestRedteamStakeholder:
    """redteam() should include a custom stakeholder in the system prompt."""

    def test_custom_stakeholder_reaches_system_prompt(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.redteam import redteam
                redteam("Some analysis text.", stakeholder="VP of Finance")

                call_kwargs = mock_client.messages.create.call_args
                assert "VP of Finance" in call_kwargs.kwargs.get("system", "")


class TestRedteamCustomSystemPrompt:
    """redteam() should forward a custom system_prompt to the API, ignoring stakeholder."""

    def test_custom_system_prompt_reaches_api(self):
        custom_prompt = "You are a hostile regulator looking for compliance violations."
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.redteam import redteam
                redteam("Some analysis text.", system_prompt=custom_prompt)

                call_kwargs = mock_client.messages.create.call_args
                assert call_kwargs.kwargs.get("system") == custom_prompt
