import os
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_ANTHROPIC = (
    "BRIDGEKIT ANALYSIS REVIEW\n"
    "─────────────────────────────────────────\n\n"
    "1. CLARITY\n"
    "✅ STRONG — Clear and jargon-free.\n\n"
    "─────────────────────────────────────────\n"
    "BOTTOM LINE\n"
    "Add quantified business impact."
)

FAKE_OPENAI = (
    "BRIDGEKIT ANALYSIS REVIEW\n"
    "─────────────────────────────────────────\n\n"
    "1. CLARITY\n"
    "⚠️  NEEDS WORK — Too much jargon.\n\n"
    "─────────────────────────────────────────\n"
    "BOTTOM LINE\n"
    "Define your methodology more clearly."
)

FAKE_SUMMARY = "Both agreed on Clarity. Anthropic was harsher on Statistical Rigor."

DEFAULT_RESPONSES = {
    "anthropic": FAKE_ANTHROPIC,
    "openai": FAKE_OPENAI,
}


def _make_call_tool_side_effect(responses: dict):
    def side_effect(tool_name, text, provider, model, kwargs):
        return responses.get(provider, "fallback output")
    return side_effect


def _patches(responses=None):
    """Context manager stacking _call_tool and _synthesize patches."""
    if responses is None:
        responses = DEFAULT_RESPONSES

    class _Ctx:
        def __enter__(self):
            self._p1 = patch("bridgekit.compare._call_tool", side_effect=_make_call_tool_side_effect(responses))
            self._p2 = patch("bridgekit.compare._synthesize", return_value=FAKE_SUMMARY)
            self._p1.__enter__()
            self._p2.__enter__()
            return self

        def __exit__(self, *args):
            self._p2.__exit__(*args)
            self._p1.__exit__(*args)

    return _Ctx()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompareReturnsString:
    def test_returns_string(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert isinstance(result, str)

    def test_returns_non_empty_string(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert len(result) > 0


class TestCompareOutputStructure:
    def test_output_contains_tool_header(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert "BRIDGEKIT COMPARE: EVALUATE" in result

    def test_output_contains_summary_section(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert "SUMMARY" in result
        assert FAKE_SUMMARY in result

    def test_summary_appears_before_provider_outputs(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert result.index("SUMMARY") < result.index(FAKE_ANTHROPIC)
        assert result.index("SUMMARY") < result.index(FAKE_OPENAI)

    def test_output_contains_both_provider_labels(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert "ANTHROPIC" in result
        assert "OPENAI" in result

    def test_output_contains_both_responses(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert FAKE_ANTHROPIC in result
        assert FAKE_OPENAI in result

    def test_anthropic_appears_before_openai(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert result.index(FAKE_ANTHROPIC) < result.index(FAKE_OPENAI)

    def test_plan_tool_header(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("What caused churn?", tool="plan")
        assert "BRIDGEKIT COMPARE: PLAN" in result

    def test_redteam_tool_header(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.", tool="redteam")
        assert "BRIDGEKIT COMPARE: REDTEAM" in result


class TestCompareValidation:
    def test_empty_text_raises_value_error(self):
        from bridgekit.compare import compare
        with pytest.raises(ValueError, match="empty"):
            compare("")

    def test_whitespace_text_raises_value_error(self):
        from bridgekit.compare import compare
        with pytest.raises(ValueError, match="empty"):
            compare("   ")

    def test_invalid_tool_raises_value_error(self):
        from bridgekit.compare import compare
        with pytest.raises(ValueError, match="Unknown tool"):
            compare("Some analysis.", tool="summarize")

    def test_too_few_providers_raises_value_error(self):
        from bridgekit.compare import compare
        with pytest.raises(ValueError, match="exactly 2"):
            compare("Some analysis.", providers=["anthropic"])

    def test_too_many_providers_raises_value_error(self):
        from bridgekit.compare import compare
        with pytest.raises(ValueError, match="exactly 2"):
            compare("Some analysis.", providers=["anthropic", "openai", "gemini"])


class TestCompareModelOverrides:
    def test_model_a_appears_in_output(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.", model_a="claude-haiku-4-5-20251001")
        assert "claude-haiku-4-5-20251001" in result

    def test_model_b_appears_in_output(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.", model_b="gpt-4-turbo")
        assert "gpt-4-turbo" in result


class TestCompareDefaultProviders:
    def test_defaults_to_anthropic_and_openai(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert "ANTHROPIC" in result
        assert "OPENAI" in result

    def test_default_models_shown(self):
        with _patches():
            from bridgekit.compare import compare
            result = compare("Some analysis.")
        assert "claude-opus-4-8" in result
        assert "gpt-4o" in result


class TestCompareSynthesis:
    def test_synthesize_called_once(self):
        with patch("bridgekit.compare._call_tool", side_effect=_make_call_tool_side_effect(DEFAULT_RESPONSES)):
            with patch("bridgekit.compare._synthesize", return_value=FAKE_SUMMARY) as mock_syn:
                from bridgekit.compare import compare
                compare("Some analysis.")
        mock_syn.assert_called_once()

    def test_synthesize_receives_both_outputs(self):
        captured = []

        def capture_synth(results):
            captured.extend(results)
            return FAKE_SUMMARY

        with patch("bridgekit.compare._call_tool", side_effect=_make_call_tool_side_effect(DEFAULT_RESPONSES)):
            with patch("bridgekit.compare._synthesize", side_effect=capture_synth):
                from bridgekit.compare import compare
                compare("Some analysis.")

        providers_seen = {r[0] for r in captured}
        assert "anthropic" in providers_seen
        assert "openai" in providers_seen


class TestCompareApiCallShape:
    def test_both_providers_called(self):
        calls = []

        def capture(tool_name, text, provider, model, kwargs):
            calls.append(provider)
            return "output"

        with patch("bridgekit.compare._call_tool", side_effect=capture):
            with patch("bridgekit.compare._synthesize", return_value=FAKE_SUMMARY):
                from bridgekit.compare import compare
                compare("Some analysis.")

        assert sorted(calls) == ["anthropic", "openai"]

    def test_user_text_passed_to_both_calls(self):
        user_text = "Our conversion rate improved after the campaign."
        seen_texts = []

        def capture(tool_name, text, provider, model, kwargs):
            seen_texts.append(text)
            return "output"

        with patch("bridgekit.compare._call_tool", side_effect=capture):
            with patch("bridgekit.compare._synthesize", return_value=FAKE_SUMMARY):
                from bridgekit.compare import compare
                compare(user_text)

        assert len(seen_texts) == 2
        assert all(t == user_text for t in seen_texts)

    def test_tool_name_passed_to_both_calls(self):
        tool_names = []

        def capture(tool_name, text, provider, model, kwargs):
            tool_names.append(tool_name)
            return "output"

        with patch("bridgekit.compare._call_tool", side_effect=capture):
            with patch("bridgekit.compare._synthesize", return_value=FAKE_SUMMARY):
                from bridgekit.compare import compare
                compare("Some analysis.", tool="redteam")

        assert all(n == "redteam" for n in tool_names)

    def test_kwargs_forwarded_to_tool(self):
        received_kwargs = []

        def capture(tool_name, text, provider, model, kwargs):
            received_kwargs.append(kwargs)
            return "output"

        with patch("bridgekit.compare._call_tool", side_effect=capture):
            with patch("bridgekit.compare._synthesize", return_value=FAKE_SUMMARY):
                from bridgekit.compare import compare
                compare("Some analysis.", max_tokens=2048)

        assert all(kw.get("max_tokens") == 2048 for kw in received_kwargs)
