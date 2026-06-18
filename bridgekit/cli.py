import argparse
import sys

from .planner import plan
from .reviewer import evaluate
from .redteam import redteam
from .search import ask


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help='AI provider: "anthropic", "openai", or "gemini"')
    parser.add_argument("--model", help="Specific model to use (e.g. claude-opus-4-8, gpt-4o)")


def _cmd_plan(args: argparse.Namespace) -> None:
    result = plan(
        question=args.question,
        data_description=args.data,
        goal=args.goal,
        provider=args.provider,
        model=args.model,
    )
    print(result)


def _cmd_review(args: argparse.Namespace) -> None:
    result = evaluate(
        text=args.text,
        provider=args.provider,
        model=args.model,
    )
    print(result)


def _cmd_redteam(args: argparse.Namespace) -> None:
    result = redteam(
        text=args.text,
        stakeholder=args.stakeholder,
        provider=args.provider,
        model=args.model,
    )
    print(result)


def _cmd_search(args: argparse.Namespace) -> None:
    if not args.source and not args.text:
        print("error: provide --source or --text", file=sys.stderr)
        sys.exit(1)
    result = ask(
        question=args.question,
        source=args.source,
        text=args.text,
        provider=args.provider,
        model=args.model,
    )
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bridgekit",
        description="AI tools for data scientists",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # plan
    p_plan = sub.add_parser("plan", help="Recommend the right analytical approach")
    p_plan.add_argument("question", help="The analytical question you want to answer")
    p_plan.add_argument("--data", metavar="DESCRIPTION", help="Description of your available data")
    p_plan.add_argument("--goal", help='Goal of the analysis (e.g. "prediction", "hypothesis testing")')
    _add_provider_args(p_plan)
    p_plan.set_defaults(func=_cmd_plan)

    # review
    p_review = sub.add_parser("review", help="Evaluate a data science analysis writeup")
    p_review.add_argument("text", help="The analysis text to review")
    _add_provider_args(p_review)
    p_review.set_defaults(func=_cmd_review)

    # redteam
    p_redteam = sub.add_parser("redteam", help="Red-team an analysis from a skeptical stakeholder")
    p_redteam.add_argument("text", help="The analysis text to red-team")
    p_redteam.add_argument("--stakeholder", help='Stakeholder role (e.g. "VP of Finance")')
    _add_provider_args(p_redteam)
    p_redteam.set_defaults(func=_cmd_redteam)

    # search
    p_search = sub.add_parser("search", help="Ask a question across documents or text")
    p_search.add_argument("question", help="The question to answer")
    p_search.add_argument("--source", metavar="PATH", help="Folder of documents to search")
    p_search.add_argument("--text", help="Raw text to search instead of a folder")
    _add_provider_args(p_search)
    p_search.set_defaults(func=_cmd_search)

    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, EnvironmentError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
