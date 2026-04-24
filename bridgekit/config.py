import os

DEFAULT_MODEL = "claude-opus-4-6"


def require_anthropic_api_key() -> str:
    """Return the Anthropic API key from the environment, or raise a clear error.

    Each Bridgekit tool calls this before constructing an Anthropic client so
    users get the same friendly message instead of whatever the SDK surfaces
    when the key is missing.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found. Set it with: export ANTHROPIC_API_KEY=your_key_here"
        )
    return api_key
