import sys
import pytest
from unittest.mock import patch


FAKE_PLAN = "BRIDGEKIT ANALYSIS PLAN\n─────\nRECOMMENDED APPROACH\nUse a t-test."
FAKE_REVIEW = "BRIDGEKIT ANALYSIS REVIEW\n─────\n1. CLARITY\n✅ STRONG Clear writing."
FAKE_REDTEAM = "BRIDGEKIT RED TEAM\n─────\nCRITIQUE 1: Sample Size\nHARDEST QUESTION TO ANSWER\nWhat is n?"
FAKE_SEARCH = "Based on the documents, the answer is 42."


class TestPlanCommand:
    def test_basic_question(self, capsys):
        with patch("bridgekit.cli.plan", return_value=FAKE_PLAN) as mock_plan:
            with patch("sys.argv", ["bridgekit", "plan", "should I use a t-test?"]):
                from bridgekit.cli import main
                main()
        mock_plan.assert_called_once_with(
            question="should I use a t-test?",
            data_description=None,
            goal=None,
            provider=None,
            model=None,
        )
        assert FAKE_PLAN in capsys.readouterr().out

    def test_with_data_and_goal(self, capsys):
        with patch("bridgekit.cli.plan", return_value=FAKE_PLAN) as mock_plan:
            with patch("sys.argv", ["bridgekit", "plan", "my question",
                                    "--data", "50 rows", "--goal", "compare means"]):
                from bridgekit.cli import main
                main()
        mock_plan.assert_called_once_with(
            question="my question",
            data_description="50 rows",
            goal="compare means",
            provider=None,
            model=None,
        )

    def test_with_provider_and_model(self):
        with patch("bridgekit.cli.plan", return_value=FAKE_PLAN) as mock_plan:
            with patch("sys.argv", ["bridgekit", "plan", "my question",
                                    "--provider", "openai", "--model", "gpt-4o"]):
                from bridgekit.cli import main
                main()
        mock_plan.assert_called_once_with(
            question="my question",
            data_description=None,
            goal=None,
            provider="openai",
            model="gpt-4o",
        )

    def test_missing_question_exits(self):
        with patch("sys.argv", ["bridgekit", "plan"]):
            from bridgekit.cli import main
            with pytest.raises(SystemExit):
                main()

    def test_environment_error_exits(self, capsys):
        with patch("bridgekit.cli.plan", side_effect=EnvironmentError("ANTHROPIC_API_KEY not found")):
            with patch("sys.argv", ["bridgekit", "plan", "my question"]):
                from bridgekit.cli import main
                with pytest.raises(SystemExit) as exc:
                    main()
        assert exc.value.code == 1
        assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


class TestReviewCommand:
    def test_basic_text(self, capsys):
        with patch("bridgekit.cli.evaluate", return_value=FAKE_REVIEW) as mock_evaluate:
            with patch("sys.argv", ["bridgekit", "review", "my analysis text"]):
                from bridgekit.cli import main
                main()
        mock_evaluate.assert_called_once_with(
            text="my analysis text",
            provider=None,
            model=None,
        )
        assert FAKE_REVIEW in capsys.readouterr().out

    def test_missing_text_exits(self):
        with patch("sys.argv", ["bridgekit", "review"]):
            from bridgekit.cli import main
            with pytest.raises(SystemExit):
                main()


class TestRedteamCommand:
    def test_basic_text(self, capsys):
        with patch("bridgekit.cli.redteam", return_value=FAKE_REDTEAM) as mock_redteam:
            with patch("sys.argv", ["bridgekit", "redteam", "my analysis text"]):
                from bridgekit.cli import main
                main()
        mock_redteam.assert_called_once_with(
            text="my analysis text",
            stakeholder=None,
            provider=None,
            model=None,
        )
        assert FAKE_REDTEAM in capsys.readouterr().out

    def test_with_stakeholder(self):
        with patch("bridgekit.cli.redteam", return_value=FAKE_REDTEAM) as mock_redteam:
            with patch("sys.argv", ["bridgekit", "redteam", "my analysis text",
                                    "--stakeholder", "VP of Finance"]):
                from bridgekit.cli import main
                main()
        mock_redteam.assert_called_once_with(
            text="my analysis text",
            stakeholder="VP of Finance",
            provider=None,
            model=None,
        )

    def test_missing_text_exits(self):
        with patch("sys.argv", ["bridgekit", "redteam"]):
            from bridgekit.cli import main
            with pytest.raises(SystemExit):
                main()


class TestSearchCommand:
    def test_with_source(self, capsys):
        with patch("bridgekit.cli.ask", return_value=FAKE_SEARCH) as mock_ask:
            with patch("sys.argv", ["bridgekit", "search", "my question",
                                    "--source", "./my_docs"]):
                from bridgekit.cli import main
                main()
        mock_ask.assert_called_once_with(
            question="my question",
            source="./my_docs",
            text=None,
            provider=None,
            model=None,
        )
        assert FAKE_SEARCH in capsys.readouterr().out

    def test_with_text(self):
        with patch("bridgekit.cli.ask", return_value=FAKE_SEARCH) as mock_ask:
            with patch("sys.argv", ["bridgekit", "search", "my question",
                                    "--text", "some raw text"]):
                from bridgekit.cli import main
                main()
        mock_ask.assert_called_once_with(
            question="my question",
            source=None,
            text="some raw text",
            provider=None,
            model=None,
        )

    def test_missing_source_and_text_exits(self, capsys):
        with patch("sys.argv", ["bridgekit", "search", "my question"]):
            from bridgekit.cli import main
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1
        assert "error" in capsys.readouterr().err

    def test_missing_question_exits(self):
        with patch("sys.argv", ["bridgekit", "search"]):
            from bridgekit.cli import main
            with pytest.raises(SystemExit):
                main()


class TestNoCommand:
    def test_no_subcommand_exits(self):
        with patch("sys.argv", ["bridgekit"]):
            from bridgekit.cli import main
            with pytest.raises(SystemExit):
                main()
