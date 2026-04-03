import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile


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


FAKE_ANSWER = "Based on the documents, the conversion rate increased by 12%."


def _make_mock_chromadb(chunks: list[str] | None = None):
    """
    Return a (mock_chromadb_module, mock_embedding_fn_class) pair whose
    collection.query() returns the supplied chunks as context.
    """
    returned_docs = chunks if chunks is not None else ["sample context chunk"]

    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [returned_docs]}

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    mock_chromadb = MagicMock()
    mock_chromadb.Client.return_value = mock_chroma_client

    mock_embedding_fn_class = MagicMock()
    mock_embedding_fn_class.return_value = MagicMock()

    return mock_chromadb, mock_embedding_fn_class


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAskReturnsString:
    """ask() should return a non-empty string."""

    def test_returns_string_with_text_input(self):
        mock_chromadb, mock_ef = _make_mock_chromadb()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic, \
                 patch("chromadb.Client", mock_chromadb.Client), \
                 patch(
                     "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                     mock_ef,
                 ):
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_ANSWER)
                MockAnthropic.return_value = mock_client

                from bridgekit.search import ask
                result = ask("What was the conversion rate?", text="The conversion rate increased by 12%.")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_non_empty_answer(self):
        mock_chromadb, mock_ef = _make_mock_chromadb()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic, \
                 patch("chromadb.Client", mock_chromadb.Client), \
                 patch(
                     "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                     mock_ef,
                 ):
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_ANSWER)
                MockAnthropic.return_value = mock_client

                from bridgekit.search import ask
                result = ask("What was the conversion rate?", text="The conversion rate increased by 12%.")

        assert result == FAKE_ANSWER


class TestAskMissingSourceAndText:
    """ask() should raise ValueError when neither source nor text is supplied."""

    def test_raises_value_error_with_no_inputs(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.search import ask
            with pytest.raises(ValueError, match="source"):
                ask("What happened?")

    def test_raises_value_error_message_mentions_text(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from bridgekit.search import ask
            with pytest.raises(ValueError):
                ask("What happened?", source=None, text=None)


class TestAskMissingApiKey:
    """ask() should raise EnvironmentError when ANTHROPIC_API_KEY is absent."""

    def test_raises_environment_error_when_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.search import ask
            with pytest.raises(EnvironmentError):
                ask("What happened?", text="Some text about results.")

    def test_error_message_mentions_key(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from bridgekit.search import ask
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                ask("What happened?", text="Some text about results.")


class TestAskWithTextInput:
    """ask() should work correctly when called with the text= parameter."""

    def test_text_input_reaches_api(self):
        raw_text = "Revenue grew 25% year-over-year driven by enterprise sales."
        mock_chromadb, mock_ef = _make_mock_chromadb([raw_text])

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic, \
                 patch("chromadb.Client", mock_chromadb.Client), \
                 patch(
                     "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                     mock_ef,
                 ):
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_ANSWER)
                MockAnthropic.return_value = mock_client

                from bridgekit.search import ask
                ask("What drove revenue growth?", text=raw_text)

                # Verify the Anthropic API was actually called once
                assert mock_client.messages.create.call_count == 1

    def test_text_input_included_in_context(self):
        raw_text = "Churn dropped from 8% to 3% after onboarding improvements."
        mock_chromadb, mock_ef = _make_mock_chromadb([raw_text])

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as MockAnthropic, \
                 patch("chromadb.Client", mock_chromadb.Client), \
                 patch(
                     "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                     mock_ef,
                 ):
                mock_client = MagicMock()
                mock_client.messages.create.return_value = _make_mock_message(FAKE_ANSWER)
                MockAnthropic.return_value = mock_client

                from bridgekit.search import ask
                ask("What happened to churn?", text=raw_text)

                call_kwargs = mock_client.messages.create.call_args
                messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
                # The retrieved chunk (raw_text) should appear in the prompt context
                assert raw_text in str(messages_arg)


class TestAskWithSourceFolder:
    """ask() should load .txt files from a folder and pass their content to the API."""

    def test_source_folder_with_txt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = Path(tmpdir) / "report.txt"
            sample_content = "The experiment showed a 15% lift in click-through rate."
            sample_file.write_text(sample_content, encoding="utf-8")

            mock_chromadb, mock_ef = _make_mock_chromadb([sample_content])

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                with patch("anthropic.Anthropic") as MockAnthropic, \
                     patch("chromadb.Client", mock_chromadb.Client), \
                     patch(
                         "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                         mock_ef,
                     ):
                    mock_client = MagicMock()
                    mock_client.messages.create.return_value = _make_mock_message(FAKE_ANSWER)
                    MockAnthropic.return_value = mock_client

                    from bridgekit.search import ask
                    result = ask("What was the lift?", source=tmpdir)

            assert isinstance(result, str)
            assert len(result) > 0

    def test_source_folder_calls_api_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "notes.txt").write_text(
                "User satisfaction scores improved by 20 points.", encoding="utf-8"
            )

            mock_chromadb, mock_ef = _make_mock_chromadb()

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                with patch("anthropic.Anthropic") as MockAnthropic, \
                     patch("chromadb.Client", mock_chromadb.Client), \
                     patch(
                         "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                         mock_ef,
                     ):
                    mock_client = MagicMock()
                    mock_client.messages.create.return_value = _make_mock_message(FAKE_ANSWER)
                    MockAnthropic.return_value = mock_client

                    from bridgekit.search import ask
                    ask("How did satisfaction change?", source=tmpdir)

                    assert mock_client.messages.create.call_count == 1

    def test_source_folder_empty_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Folder exists but has no supported files
            mock_chromadb, mock_ef = _make_mock_chromadb()

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                with patch("chromadb.Client", mock_chromadb.Client), \
                     patch(
                         "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                         mock_ef,
                     ):
                    from bridgekit.search import ask
                    with pytest.raises(ValueError, match="No content found"):
                        ask("What happened?", source=tmpdir)
