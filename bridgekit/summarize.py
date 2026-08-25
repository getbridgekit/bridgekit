from .config import DEFAULT_MODEL, parse_provider, get_default_model
from .providers import create_message

DEFAULT_AUDIENCE = "a general business audience with no technical or data science background"

SYSTEM_PROMPT_TEMPLATE = """You are a senior data scientist preparing an executive summary of a technical analysis for {audience}.

Your job is to translate technical detail into what matters for decision-making. Cut jargon, cut methodology detail unless it's essential to trust the conclusion, and lead with the takeaway. Write for someone who will skim this in 30 seconds.

Format your response exactly like this:

BRIDGEKIT SUMMARY
─────────────────────────────────────────
AUDIENCE: {audience_label}

KEY TAKEAWAY
[1-2 sentences: the single most important finding or recommendation]

WHAT WE FOUND
[3-5 bullet points of the supporting findings, in plain language]

SO WHAT
[1-2 sentences on the business implication or recommended next step]

─────────────────────────────────────────
"""


def summarize(text: str, audience: str = None, provider: str = None, model: str = None, system_prompt: str = None, max_tokens: int = 1024) -> str:
    """
    Turn a long analysis, notebook, or report into a short executive summary.

    Args:
        text:          The long-form analysis writeup, notebook output, or report as a plain string.
        audience:      Optional. Who the summary is for (e.g. "VP of Marketing", "board",
                       "engineering team"). Defaults to a general business audience.
        provider:      Optional. The AI provider to use ("anthropic", "openai", "gemini").
                       If not specified, defaults to "anthropic" or infers from model.
        model:         Optional. The specific model to use. If not specified, uses the provider's default.
        system_prompt: Optional. A custom system prompt to fully override the default summarizer persona.
                       When provided, the audience parameter is ignored.
        max_tokens:    Optional. Maximum tokens in the response. Defaults to 1024.

    Returns:
        A short executive summary covering the key takeaway, supporting findings,
        and the business implication.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    # Parse provider and determine model
    provider_enum = parse_provider(provider, model)
    if model is None:
        model = get_default_model(provider_enum)

    if system_prompt is None:
        audience_label = audience if audience else "General Business Audience"
        audience_desc = audience if audience else DEFAULT_AUDIENCE
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            audience=audience_desc,
            audience_label=audience_label
        )

    user_message = f"Summarize this analysis:\n\n{text}"

    return create_message(
        provider=provider_enum,
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        max_tokens=max_tokens
    )
