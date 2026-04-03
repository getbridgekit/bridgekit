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
    "BRIDGEKIT ANALYSIS PLAN\n"
    "─────────────────────────────────────────\n\n"
    "RECOMMENDED APPROACH\n"
    "A/B test with a two-proportion z-test.\n\n"
    "WHY THIS APPROACH\n"
    "Random assignment handles confounding.\n\n"
    "KEY ASSUMPTIONS\n"
    "- Users were randomly assigned\n"
    "- Independence between users\n\n"
    "WATCH OUT FOR\n"
    "Peeking at results before the test reaches planned sample size.\n\n"
    "ALTERNATIVES\n"
    "Logistic regression if you need to control for covariates.\n"
    "─────────────────────────────────────────\n"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlanReturnsString:
    """plan() should return a non-empty string."""

    def test_returns_string(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.planner import plan
                result = plan("Does our new onboarding flow increase upgrade rates?")

        assert isinstance(result, str)
        assert len(result) > 0


class TestPlanOutputStructure:
    """plan() output should contain required section headers."""

    def test_output_contains_recommended_approach(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.planner import plan
                result = plan("Does our new onboarding flow increase upgrade rates?")

        assert "RECOMMENDED APPROACH" in result

    def test_output_contains_watch_out_for(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.planner import plan
                result = plan("Does our new onboarding flow increase upgrade rates?")

        assert "WATCH OUT FOR" in result

    def test_output_contains_alternatives(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.planner import plan
                result = plan("Does our new onboarding flow increase upgrade rates?")

        assert "ALTERNATIVES" in result


class TestPlanMissingApiKey:
    """plan() should raise EnvironmentError when the API key is absent."""

    def test_raises_environment_error_when_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.planner import plan
            with pytest.raises(EnvironmentError):
                plan("Does our new onboarding flow increase upgrade rates?")

    def test_error_message_mentions_key(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.planner import plan
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                plan("Does our new onboarding flow increase upgrade rates?")


class TestPlanEmptyInput:
    """plan() should raise ValueError for empty or whitespace-only questions."""

    def test_empty_string_raises_value_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.planner import plan
            with pytest.raises(ValueError, match="empty"):
                plan("")

    def test_whitespace_only_raises_value_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.planner import plan
            with pytest.raises(ValueError, match="empty"):
                plan("   ")


class TestPlanOptionalParameters:
    """plan() should work with and without optional parameters."""

    def test_question_only(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.planner import plan
                result = plan("Does our new onboarding flow increase upgrade rates?")

        assert isinstance(result, str)

    def test_with_all_parameters(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.planner import plan
                result = plan(
                    question="Does our new onboarding flow increase upgrade rates?",
                    data_description="5,000 users split 50/50.",
                    goal="causal inference"
                )

        assert isinstance(result, str)

    def test_all_parameters_included_in_api_call(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_RESPONSE)
                MockAnthropic.return_value = mock_client

                from bridgekit.planner import plan
                plan(
                    question="Does our new onboarding flow increase upgrade rates?",
                    data_description="5,000 users split 50/50.",
                    goal="causal inference"
                )

                call_kwargs = mock_client.messages.create.call_args
                messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
                content = str(messages_arg)
                assert "5,000 users split 50/50." in content
                assert "causal inference" in content
